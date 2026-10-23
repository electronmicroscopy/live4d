#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/logger/logger.hpp>
#include "constants.h"
#include "batch_window.hpp"
#include "pipeline_types.hpp"

namespace holoscan::ops {
struct DecompressGpuParams {
  uint64_t publication_window{};
  uint64_t input_window{};
};
HOLOSCAN_PARAM_SCHEMA(
    DecompressGpuParams,
    holoscan::param_schema("holoscan.dectris.DecompressGpuParams", 1),
    holoscan::field<&DecompressGpuParams::publication_window>("publication_window"),
    holoscan::field<&DecompressGpuParams::input_window>("input_window"));

class DecompressGpuOp final : public Operator<DecompressGpuParams> {
 public:
  ~DecompressGpuOp() override;
  static constexpr size_t MAX_BLOCKS_PER_IMAGE = 256;
  void setup(OperatorSpec& spec) override;
  Contract contract() const override;
  void start() override;
  expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;
  Input<CompressedBatch> input;
  Output<DecodedBatch> output;
 private:
  uint8_t* d_compressed_data_ = nullptr;
  uint8_t* d_intermediate_data_ = nullptr;
  const void** d_compressed_ptrs_ = nullptr;
  size_t* d_compressed_sizes_ = nullptr;
  int* h_pinned_num_blocks_ = nullptr;
  nvcompStatus_t* h_pinned_statuses_ = nullptr;
  cudaStream_t stream_ = nullptr;
  int64_t last_capture_ns_ = -1;
  ::dectris::BatchWindow* publication_window_{};
  ::dectris::BatchWindow* input_window_{};
  size_t max_compressed_size_per_image_ = 0;
  size_t max_compressed_batch_size_ = 0;
  uint32_t active_width_{static_cast<uint32_t>(DETECTOR_WIDTH)};
  uint32_t active_height_{static_cast<uint32_t>(DETECTOR_HEIGHT)};
  uint64_t active_series_id_{};
  bool active_series_{};
  void allocate_gpu_resources();
  void free_gpu_resources();
  bool decompress_batch_gpu(const std::vector<const uint8_t*>& compressed_ptrs,
                            const std::vector<size_t>& compressed_sizes,
                            size_t num_images, size_t image_pixels_per_frame,
                            uint16_t* output_buffer);
};
}  // namespace holoscan::ops
