#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <matx.h>

namespace holoscan::ops {

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
