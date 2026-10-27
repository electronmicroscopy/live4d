#pragma once

// Frame layout and stateless helpers shared by the imagemux RDMA source operators.
// Protocol 2 has one ring; protocol 3 adds the ring index to the frame header and immediate.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "constants.h"

namespace holoscan::ops::imagemux {

constexpr size_t kFrameAlign = 128;
constexpr size_t kFrameHeaderBytes = 128;
constexpr uint32_t kFrameMagic = 0x46584D49U;  // "IMXF" in little-endian wire order.
constexpr uint16_t kFrameStart = 1;
constexpr uint16_t kFrameImage = 2;
constexpr uint16_t kFrameEnd = 3;
constexpr uint16_t kFrameWrap = 0xFFFF;
constexpr uint16_t kDataTypeUint16 = 2;
constexpr uint16_t kCompressionNone = 0;
constexpr size_t kImageBytes = dectris::kDetectorPixels * sizeof(uint16_t);
constexpr size_t kMaxStartBytes = 16 * 1024;

inline uint16_t load_le16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

inline uint32_t load_le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline uint64_t load_le64(const uint8_t* p) {
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= static_cast<uint64_t>(p[i]) << (8 * i);
  return value;
}

// One mapped-host gather kernel per translation unit; device code is not shared across files.
static __global__ void gather_mapped_bytes(const uint8_t* source, uint8_t* destination,
                                           size_t bytes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < bytes) destination[index] = source[index];
}

inline Error make_error(ErrorCode code, std::string message) {
  return Error{code, std::move(message)};
}

inline std::string cuda_detail(cudaError_t status) {
  return std::string(cudaGetErrorString(status));
}

inline std::string sdk_detail(const Error& error) {
  std::string detail(error_code_name(error.code));
  if (!error.message.empty()) detail += ": " + error.message;
  return detail;
}

inline std::string loan_detail(const TensorOutputLoan& loan) {
  return "device_id=" + std::to_string(loan.device_id().value) +
         ", memory_kind=" + std::to_string(static_cast<unsigned>(loan.memory_kind())) +
         ", byte_size=" + std::to_string(loan.byte_size());
}

struct StreamSyncOnAbandon {
  cudaStream_t stream{};
  bool& work_enqueued;
  bool& published;

  ~StreamSyncOnAbandon() {
    if (work_enqueued && !published) cudaStreamSynchronize(stream);
  }
};

}  // namespace holoscan::ops::imagemux
