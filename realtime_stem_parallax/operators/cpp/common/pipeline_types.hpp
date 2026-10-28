#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <matx.h>

namespace holoscan::ops {

// Bytes per pixel for a configured dtype string. Shared by ZmqRxOp and DecompressGpuOp;
// extend both this map and DecompressGpuOp's output-tensor dispatch to add a dtype.
inline std::size_t pixel_dtype_elem_size(const std::string& dtype) {
  if (dtype == "uint16") return 2;
  if (dtype == "uint32") return 4;
  throw std::runtime_error("Unsupported pixel dtype '" + dtype +
                           "' (expected 'uint16' or 'uint32')");
}

struct CompressedBatch {
  // Points into ZmqRxOp's circular pinned host pool. The basic pipeline relies
  // on downstream operators consuming a slot before ZmqRxOp reuses it; a
  // production follow-up should replace this with explicit slot ownership or
  // backpressure.
  uint8_t* h_pinned_data = nullptr;
  std::shared_ptr<matx::tensor_t<std::size_t, 1>> sizes;
  std::shared_ptr<matx::tensor_t<std::size_t, 1>> offsets;
  std::shared_ptr<matx::tensor_t<uint64_t, 1>> image_ids;
  std::size_t num_images = 0;
  uint64_t batch_id = 0;

  CompressedBatch() = default;

  CompressedBatch(uint8_t* data,
                  std::shared_ptr<matx::tensor_t<std::size_t, 1>> sz,
                  std::shared_ptr<matx::tensor_t<std::size_t, 1>> off,
                  std::shared_ptr<matx::tensor_t<uint64_t, 1>> ids,
                  std::size_t num,
                  uint64_t bid)
      : h_pinned_data(data),
        sizes(std::move(sz)),
        offsets(std::move(off)),
        image_ids(std::move(ids)),
        num_images(num),
        batch_id(bid) {}
};

}  // namespace holoscan::ops
