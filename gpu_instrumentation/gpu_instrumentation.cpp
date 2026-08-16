#include "gpu_instrumentation.h"
#include "utilities.h"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
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
// rpc_message_header_v03_00.function is the fourth NvU32 of the rpc header.
constexpr std::uint64_t ELEM_RPC_FUNCTION{ELEM_HDR_SIZE + 12};
// sizeof(GSP_MSG_QUEUE_ELEMENT): header plus the 32 byte rpc header.
constexpr std::uint64_t ELEM_MIN_SIZE{ELEM_HDR_SIZE + 32};

//
// NV_VGPU_MSG_EVENT_FIRST_EVENT, ogkm/src/nvidia/inc/kernel/vgpu/rpc_global_enums.h:238.
// GSP -> CPU messages below this are returns for a CPU -> GSP RPC, which
// _kgspRpcRecvPoll (kernel_gsp.c:2063) may be blocked on; at or above it they
// are asynchronous events that nothing waits for.
//
constexpr std::uint32_t RPC_FIRST_EVENT{0x1000};

// How long gspStatusMsg waits for the GSP to leave a message in the queue.
constexpr std::uint64_t SUBSTITUTE_WAIT_NS{50ull * 1000 * 1000};

std::uint64_t monotonicNs()
{
	struct timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 0;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<std::uint64_t>(ts.tv_sec) * 1000000000ull +
	       static_cast<std::uint64_t>(ts.tv_nsec);
}

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

//
// The GSP's own write pointer, re-read live.  msgqTxSubmitBuffers (msgq.c:526)
// stores the GSP's private tx.writePtr here on every submit, so this is the
// only honest statement of how many status queue slots actually hold a message.
//
std::uint32_t readWritePtr(const GspMsgQueue::Info& info)
{
	const auto wp{injectorReadMemoryU32(info.status_queue_offset + TX_WRITE_PTR)};
	return wp ? *wp : ~0u;
}

std::uint64_t slotOffset(const GspMsgQueue::Info& info, const Geometry& geo,
			 const std::uint32_t slot)
{
	return info.status_queue_offset + geo.entryOff +
	       static_cast<std::uint64_t>(slot) * geo.msgSize;
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
 * gspStatusMsg: overwrite a GSP -> driver status queue message that the GSP has
 * already produced and the driver has not yet consumed.
 *
 * @elem:     the element bytes, laid out as GSP_MSG_QUEUE_ELEMENT
 * @size:     length of @elem, in bytes
 * @seqDelta: 0..32, applied as (victim sequence number + seqDelta - 16)
 *
 * static:  every byte of @elem apart from checkSum and seqNum - authTag, aad,
 *          elemCount, the whole rpc header and the payload - plus @seqDelta.
 * dynamic: the target slot and the victim's seqNum, both read live, and
 *          checkSum, solved from the resulting bytes.
 *
 * This call substitutes; it never appends.  Round 3's crash
 * f5b7286d "NVRM: GSP RPC timeout" came from appending: the old code raised the
 * status queue writePtr (msgq.c:526 is the GSP's only writer of that field) so
 * the driver consumed a slot the GSP had not filled yet.  msgqRxMarkConsumed
 * (msgq.c:675) then left the driver's private rxReadPtr one slot ahead of the
 * GSP's private tx.writePtr, the GSP's next message landed in a slot the driver
 * had already passed, msgqRxGetReadAvailable (msgq.c:615) reported 0 for it and
 * _kgspRpcRecvPoll (kernel_gsp.c:2097) spun out its 45s timeout on the reply
 * that message was.  Overwriting an already-produced element advances neither
 * pointer, so the two sides stay in step.
 *
 * The victim must also be an asynchronous event (rpc.function >=
 * NV_VGPU_MSG_EVENT_FIRST_EVENT).  Destroying an RPC return would strand
 * whatever _kgspRpcRecvPoll is polling for and produce the same timeout without
 * any pointer skew.
 */
int gspStatusMsg(const std::uint8_t* elem, const std::uint32_t size,
		 const std::uint32_t seqDelta)
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

	// One message may not wrap onto itself.
	const std::uint32_t nSlots{(size + geo.msgSize - 1) / geo.msgSize};
	if (nSlots > geo.msgCount - 1) {
		errno = EINVAL;
		return -1;
	}

	//
	// Wait for the GSP to leave an event in the queue.  Both pointers are
	// re-read every pass: the driver drains from its own poll loop and from
	// the interrupt bottom half, so an availability read goes stale fast.
	//
	std::uint32_t slot{0};
	std::uint32_t victimSeq{0};
	std::uint32_t victimFn{0};
	std::uint32_t nWrite{0};
	std::uint32_t lastAvail{0};
	const std::uint64_t deadline{monotonicNs() + SUBSTITUTE_WAIT_NS};

	while (monotonicNs() < deadline) {
		const std::uint32_t wp{readWritePtr(*info)};
		if (wp >= geo.msgCount) {
			continue;
		}
		const std::uint32_t rp{readReadPtr(*info, geo)};
		if (rp >= geo.msgCount) {
			continue;
		}
		const std::uint32_t nAvail{(wp + geo.msgCount - rp) % geo.msgCount};
		lastAvail = nAvail;
		if (nAvail == 0) {
			continue;
		}

		//
		// Take the longest run of event slots starting at the read
		// pointer, capped by what the GSP produced and by what @elem
		// actually fills.
		//
		std::uint32_t n{0};
		const std::uint32_t want{std::min(nSlots, nAvail)};
		for (; n < want; ++n) {
			const std::uint64_t off{
			    slotOffset(*info, geo, (rp + n) % geo.msgCount)};
			const auto fn{injectorReadMemoryU32(off + ELEM_RPC_FUNCTION)};
			if (!fn || *fn < RPC_FIRST_EVENT) {
				break;
			}
			if (n == 0) {
				victimFn = *fn;
			}
		}
		if (n == 0) {
			continue;
		}

		const auto seq{
		    injectorReadMemoryU32(slotOffset(*info, geo, rp) + ELEM_SEQNUM)};
		if (!seq) {
			continue;
		}

		slot = rp;
		victimSeq = *seq;
		nWrite = n;
		break;
	}

	if (nWrite == 0) {
		fprintf(stderr,
			"[GPU INSTRUMENTATION] status msg: no event pending, avail=%u "
			"msgCount=%u size=%u\n",
			lastAvail, geo.msgCount, size);
		errno = EAGAIN;
		return -1;
	}

	// Only the slots we are allowed to overwrite get written.
	std::vector<std::uint8_t> buf(
	    elem, elem + std::min<std::uint64_t>(
			     size, static_cast<std::uint64_t>(nWrite) * geo.msgSize));

	//
	// GspMsgQueueReceiveStatus accepts an element only when its seqNum equals
	// pMQI->rxSeqNum (message_queue_cpu.c:697), a driver-private counter.  The
	// element we are overwriting has not been consumed yet, so its own seqNum
	// is exactly the value the driver is about to demand.
	//
	const std::uint32_t seqNum{victimSeq + seqDelta - 16u};
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
	// so a wrap in the queue is still contiguous once staged.  writePtr is
	// deliberately left alone; see the note above.
	//
	std::uint64_t done{0};
	for (std::uint32_t i{0}; i < nWrite && done < buf.size(); ++i) {
		const std::uint64_t off{
		    slotOffset(*info, geo, (slot + i) % geo.msgCount)};
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

	fprintf(stderr,
		"[GPU INSTRUMENTATION] status msg: hdr=%s slot=%u nwrite=%u nslots=%u "
		"avail=%u msgCount=%u victimFn=0x%x victimSeq=%u seqNum=%u elemCount=%u "
		"fn=0x%x rpcLen=0x%x size=%u\n",
		geo.live ? "live" : "fallback", slot, nWrite, nSlots, lastAvail,
		geo.msgCount, victimFn, victimSeq, seqNum,
		bufToU32(std::vector<std::uint8_t>(buf.begin() + 40, buf.begin() + 44)),
		bufToU32(std::vector<std::uint8_t>(buf.begin() + ELEM_RPC_FUNCTION,
						   buf.begin() + ELEM_RPC_FUNCTION + 4)),
		rpcLength, size);

	return 0;
}
