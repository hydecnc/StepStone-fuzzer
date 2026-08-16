#ifndef GPU_INSTRUMENTATION_H
#define GPU_INSTRUMENTATION_H

#include <cstdint>
#include <string>

int insertBuffer(std::uint8_t* buffer, const std::uint32_t size, const std::uint64_t offset);

int gspStatusMsg(const std::uint8_t* elem, const std::uint32_t size,
		 const std::uint32_t avail, const std::uint32_t seqDelta);

#endif
