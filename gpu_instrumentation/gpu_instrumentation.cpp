#include "gpu_instrumentation.h"
#include "utilities.h"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Helper to insert U32 to a buffer
void putU32(std::vector<std::uint8_t>& buf, const std::uint64_t offset,
	    const std::uint32_t value)
{
	const auto bytes{u32ToBuf(value)};
	std::memcpy(buf.data() + offset, bytes.data(), bytes.size());
}

namespace
{
//
// msgqTxHeader, ogkm/src/common/shared/msgq/inc/msgq/msgq_priv.h:49.
// Written into the head of each queue by msgqTxCreate (msgq.c:290).
//
constexpr std::uint64_t TX_VERSION{0};
constexpr std::uint64_t TX_SIZE{4};
constexpr std::uint64_t TX_MSG_SIZE{8};
constexpr std::uint64_t TX_MSG_COUNT{12};
constexpr std::uint64_t TX_WRITE_PTR{16};
constexpr std::uint64_t TX_FLAGS{20};
constexpr std::uint64_t TX_RX_HDR_OFF{24};
constexpr std::uint64_t TX_ENTRY_OFF{28};

// msgq.h:39
constexpr std::uint32_t MSGQ_FLAGS_SWAP_RX{1};
// msgq_priv.h:38
constexpr std::uint32_t MSGQ_VERSION{0};

//
// GSP_MSG_QUEUE_ELEMENT, ogkm/src/nvidia/inc/kernel/gpu/gsp/message_queue_priv.h:43.
// authTagBuffer[16] | aadBuffer[16] | checkSum | seqNum | elemCount | pad | rpc
//
constexpr std::uint64_t ELEM_CHECKSUM{32};
constexpr std::uint64_t ELEM_SEQNUM{36};
// GSP_MSG_QUEUE_ELEMENT_HDR_SIZE == NV_OFFSETOF(GSP_MSG_QUEUE_ELEMENT, rpc)
constexpr std::uint64_t ELEM_HDR_SIZE{48};
// rpc_message_header_v03_00.length is the third NvU32 of the rpc header.
constexpr std::uint64_t ELEM_RPC_LENGTH{ELEM_HDR_SIZE + 8};
// sizeof(GSP_MSG_QUEUE_ELEMENT): header plus the 32 byte rpc header.
constexpr std::uint64_t ELEM_MIN_SIZE{ELEM_HDR_SIZE + 32};

// GSP_MSG_QUEUE_ELEMENT_SIZE_MIN, message_queue_priv.h:91
constexpr std::uint32_t DEFAULT_MSG_SIZE{4096};
// msgqTxCreate: NV_ALIGN_UP(rxHdrOff + 4, 1 << 12), msgq.c:227
constexpr std::uint32_t DEFAULT_ENTRY_OFF{4096};
// msgqTxCreate: NV_ALIGN_UP(sizeof(msgqTxHeader), 1 << 4), msgq.c:226
constexpr std::uint32_t DEFAULT_RX_HDR_OFF{32};

struct Geometry {
	std::uint32_t msgSize{};
	std::uint32_t msgCount{};
	std::uint32_t entryOff{};
	std::uint32_t rxHdrOff{};
	std::uint32_t writePtr{};
	std::uint32_t flags{};
	bool live{}; // the shared header validated, values are the GSP's
};

//
// _checkSum32, message_queue_priv.h:112.  XOR of the NvU64 words covering
// [0, len), then hi32 ^ lo32.  Bytes past the end of @buf are read as zero,
// which matches the driver only when the element we place already covers the
// whole checksummed span; when rpc.length reaches past it the checksum cannot
// be solved and the read at message_queue_cpu.c:684 is the point of interest
// anyway, since it happens before the result is compared.
//
std::uint32_t checkSum32(const std::vector<std::uint8_t>& buf, const std::uint64_t len)
{
	std::uint64_t sum{};
	for (std::uint64_t i{0}; i < len; i += 8) {
		std::uint64_t word{};
		for (std::uint64_t b{0}; b < 8; ++b) {
			const std::uint64_t idx{i + b};
			if (idx < buf.size()) {
				word |= static_cast<std::uint64_t>(buf[idx]) << (8 * b);
			}
		}
		sum ^= word;
	}
	return static_cast<std::uint32_t>(sum >> 32) ^
	       static_cast<std::uint32_t>(sum & 0xffffffffu);
}

//
// Read the status queue's msgqTxHeader.  msgqRxLink (msgq.c:360-385) validated
// these once at link time and kept a private copy, so a header the fuzzer has
// since trampled tells us nothing; fall back to the values msgqTxCreate must
// have produced for a 0x40000 byte, 4096 byte-per-element queue.
//
Geometry readGeometry(const GspMsgQueue::Info& info)
{
	Geometry geo{};
	const std::uint64_t base{info.status_queue_offset};

	const auto version{injectorReadMemoryU32(base + TX_VERSION)};
	const auto size{injectorReadMemoryU32(base + TX_SIZE)};
	const auto msgSize{injectorReadMemoryU32(base + TX_MSG_SIZE)};
	const auto msgCount{injectorReadMemoryU32(base + TX_MSG_COUNT)};
	const auto writePtr{injectorReadMemoryU32(base + TX_WRITE_PTR)};
	const auto flags{injectorReadMemoryU32(base + TX_FLAGS)};
	const auto rxHdrOff{injectorReadMemoryU32(base + TX_RX_HDR_OFF)};
	const auto entryOff{injectorReadMemoryU32(base + TX_ENTRY_OFF)};

	if (version && size && msgSize && msgCount && writePtr && flags && rxHdrOff &&
	    entryOff && *version == MSGQ_VERSION && *size == info.status_queue_size &&
	    *msgSize == DEFAULT_MSG_SIZE && *entryOff > 0 &&
	    *entryOff < info.status_queue_size && *rxHdrOff >= DEFAULT_RX_HDR_OFF &&
	    *rxHdrOff + 4 <= *entryOff &&
	    *msgCount == (info.status_queue_size - *entryOff) / *msgSize &&
	    *msgCount >= 2) {
		geo.msgSize = *msgSize;
		geo.msgCount = *msgCount;
		geo.entryOff = *entryOff;
		geo.rxHdrOff = *rxHdrOff;
		geo.writePtr = *writePtr;
		geo.flags = *flags;
		geo.live = true;
		return geo;
	}

	geo.msgSize = DEFAULT_MSG_SIZE;
	geo.entryOff = DEFAULT_ENTRY_OFF;
	geo.rxHdrOff = DEFAULT_RX_HDR_OFF;
	geo.msgCount = static_cast<std::uint32_t>(
	    (info.status_queue_size - DEFAULT_ENTRY_OFF) / DEFAULT_MSG_SIZE);
	geo.writePtr = writePtr ? *writePtr : 0;
	// Assume the GSP agreed to MSGQ_FLAGS_SWAP_RX; see the note in readReadPtr.
	geo.flags = MSGQ_FLAGS_SWAP_RX;
	geo.live = false;
	return geo;
}

//
// The slot the driver will read next.  msgqRxMarkConsumed (msgq.c:682) mirrors
// the driver's private rxReadPtr into pReadOutgoing on every consume, and
// msgqRxLink (msgq.c:416) zeroes it at link time, so the mirror is always
// current.  With MSGQ_FLAGS_SWAP_RX agreed on both sides that mirror lives in
// the command queue (msgq.c:398); otherwise in the status queue (msgq.c:403).
//
std::uint32_t readReadPtr(const GspMsgQueue::Info& info, const Geometry& geo)
{
	const auto cmdFlags{injectorReadMemoryU32(info.cmd_queue_offset + TX_FLAGS)};
	const auto cmdRxHdrOff{
	    injectorReadMemoryU32(info.cmd_queue_offset + TX_RX_HDR_OFF)};

	const bool swapped{cmdFlags && (*cmdFlags & MSGQ_FLAGS_SWAP_RX) &&
			   (geo.flags & MSGQ_FLAGS_SWAP_RX)};

	std::uint64_t off{info.status_queue_offset + geo.rxHdrOff};
	if (swapped && cmdRxHdrOff && *cmdRxHdrOff >= DEFAULT_RX_HDR_OFF &&
	    *cmdRxHdrOff + 4 <= info.cmd_queue_size) {
		off = info.cmd_queue_offset + *cmdRxHdrOff;
	}

	const auto rp{injectorReadMemoryU32(off)};
	if (rp && *rp < geo.msgCount) {
		return *rp;
	}
	// Mirror unreadable or out of range: the queue is idle iff writePtr == readPtr.
	return geo.writePtr < geo.msgCount ? geo.writePtr : 0;
}
} // namespace

/**
 * insertBuffer: insert a random buffer at a specific region of the message queue; intentionally unsafe to serve as a quick example/primitive
 * @buffer: the buffer to be inserted
 * @size: the size of the buffer, in bytes
 * @offset: offset from the kernel virtual address of the message queue base
 *
 * static constraint: @offset must land in the status queue.  See
 * gpu_instrumentation.txt.
 */
int insertBuffer(std::uint8_t* buffer, const std::uint32_t size, const std::uint64_t offset)
{
	if (buffer == nullptr || size == 0) {
		errno = EINVAL;
		return -1;
	}

	const auto info{prepareInjection()};
	if (!info) {
		errno = ENODEV;
		return -1;
	}

	if (offset < info->status_queue_offset) {
		errno = EINVAL;
		return -1;
	}

	// create a message
	std::vector<std::uint8_t> message(buffer, buffer + size);

	if (!injectorWriteMemory(offset, message)) {
		errno = EIO;
		return -1;
	}

	return 0;
}

/**
 * gspStatusMsg: place one GSP -> driver status queue message where the driver
 * will read it next, and advertise it.
 *
 * @elem:     the element bytes, laid out as GSP_MSG_QUEUE_ELEMENT
 * @size:     length of @elem, in bytes
 * @avail:    how many elements the queue should advertise as readable
 * @seqDelta: 0..32, applied as (derived sequence number + seqDelta - 16)
 *
 * static:  every byte of @elem apart from checkSum and seqNum, plus @avail and
 *          @seqDelta.
 * dynamic: the target slot, seqNum, checkSum and the status queue writePtr,
 *          all read from or written to live shared memory.
 */
int gspStatusMsg(const std::uint8_t* elem, const std::uint32_t size,
		 const std::uint32_t avail, const std::uint32_t seqDelta)
{
	if (elem == nullptr || size < ELEM_MIN_SIZE) {
		errno = EINVAL;
		return -1;
	}

	const auto info{prepareInjection()};
	if (!info) {
		errno = ENODEV;
		return -1;
	}
	if (info->status_queue_offset == 0 || info->status_queue_size == 0) {
		errno = ENODEV;
		return -1;
	}

	const Geometry geo{readGeometry(*info)};
	if (geo.msgSize == 0 || geo.msgCount < 2) {
		errno = EIO;
		return -1;
	}

	const std::uint32_t slot{readReadPtr(*info, geo)};

	// One message may not wrap onto itself.
	const std::uint32_t nSlots{(size + geo.msgSize - 1) / geo.msgSize};
	if (nSlots > geo.msgCount - 1) {
		errno = EINVAL;
		return -1;
	}

	std::uint32_t nAvail{avail};
	if (nAvail == 0) {
		nAvail = 1;
	}
	if (nAvail > geo.msgCount - 1) {
		nAvail = geo.msgCount - 1;
	}

	std::vector<std::uint8_t> buf(elem, elem + size);

	//
	// GspMsgQueueReceiveStatus accepts an element only when its seqNum equals
	// pMQI->rxSeqNum (message_queue_cpu.c:697), a driver-private counter.  The
	// element the driver consumed last is still sitting in the slot before the
	// read pointer and carries rxSeqNum - 1.
	//
	const std::uint32_t prevSlot{(slot + geo.msgCount - 1) % geo.msgCount};
	const std::uint64_t prevOff{info->status_queue_offset + geo.entryOff +
				    static_cast<std::uint64_t>(prevSlot) * geo.msgSize};
	const auto prevSeq{injectorReadMemoryU32(prevOff + ELEM_SEQNUM)};
	const std::uint32_t seqNum{(prevSeq ? *prevSeq : 0u) + 1u + seqDelta - 16u};
	putU32(buf, ELEM_SEQNUM, seqNum);

	//
	// checkSum is defined as the value that makes _checkSum32 over
	// GSP_MSG_QUEUE_ELEMENT_HDR_SIZE + rpc.length come out zero
	// (message_queue_cpu.c:684).  It sits in the low half of the fifth NvU64,
	// so with the field zeroed the accumulated hi32 ^ lo32 is exactly the
	// value the field has to hold.
	//
	putU32(buf, ELEM_CHECKSUM, 0);
	const std::uint32_t rpcLength{bufToU32(std::vector<std::uint8_t>(
	    buf.begin() + ELEM_RPC_LENGTH, buf.begin() + ELEM_RPC_LENGTH + 4))};
	const std::uint64_t csLen{std::min<std::uint64_t>(
	    ELEM_HDR_SIZE + static_cast<std::uint64_t>(rpcLength), buf.size())};
	const std::uint32_t checkSum{checkSum32(buf, std::max<std::uint64_t>(csLen, ELEM_HDR_SIZE))};
	putU32(buf, ELEM_CHECKSUM, checkSum);

	//
	// Write the element across consecutive slots, wrapping.  The driver's
	// staging copy (message_queue_cpu.c:652) concatenates slots in read order,
	// so a wrap in the queue is still contiguous once staged.
	//
	std::uint64_t done{0};
	for (std::uint32_t i{0}; i < nSlots; ++i) {
		const std::uint32_t s{(slot + i) % geo.msgCount};
		const std::uint64_t off{info->status_queue_offset + geo.entryOff +
					static_cast<std::uint64_t>(s) * geo.msgSize};
		const std::uint64_t chunk{
		    std::min<std::uint64_t>(geo.msgSize, buf.size() - done)};
		std::vector<std::uint8_t> part(buf.begin() + done,
					       buf.begin() + done + chunk);
		if (!injectorWriteMemory(off, part)) {
			errno = EIO;
			return -1;
		}
		done += chunk;
	}

	//
	// Advertise it.  msgqRxGetReadAvailable (msgq.c:610) ignores the queue
	// entirely when writePtr >= msgCount, and reports
	// (writePtr + msgCount - rxReadPtr) % msgCount otherwise.
	//
	const std::uint32_t writePtr{(slot + nAvail) % geo.msgCount};
	if (!injectorWriteMemoryU32(info->status_queue_offset + TX_WRITE_PTR, writePtr)) {
		errno = EIO;
		return -1;
	}

	fprintf(stderr,
		"[GPU INSTRUMENTATION] status msg: hdr=%s slot=%u nslots=%u writePtr=%u "
		"msgCount=%u prevSeq=%u seqNum=%u elemCount=%u fn=0x%x rpcLen=0x%x size=%u\n",
		geo.live ? "live" : "fallback", slot, nSlots, writePtr, geo.msgCount,
		prevSeq ? *prevSeq : 0u, seqNum,
		bufToU32(std::vector<std::uint8_t>(buf.begin() + 40, buf.begin() + 44)),
		bufToU32(std::vector<std::uint8_t>(buf.begin() + ELEM_HDR_SIZE + 12,
						   buf.begin() + ELEM_HDR_SIZE + 16)),
		rpcLength, size);

	return 0;
}
