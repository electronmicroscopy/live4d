#pragma once

#include <cuda_runtime.h>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "batch_window.hpp"
#include "constants.h"
#include "pipeline_types.hpp"

namespace holoscan::ops {

// Uploads one lane's uncompressed uint16 batches from pinned host memory to its GPU. It takes the
// same CompressedBatch messages as DecompressGpuOp, ignores batches tagged for other lanes, and
// emits the DecodedBatch the scan and detector operators consume.
struct RawUploadGpuParams {
  uint64_t publication_window{};
  uint64_t input_window{};
  uint32_t destination{};
  int64_t device{};
};
HOLOSCAN_PARAM_SCHEMA(
    RawUploadGpuParams, holoscan::param_schema("holoscan.dectris.RawUploadGpuParams", 1),
    holoscan::field<&RawUploadGpuParams::publication_window>("publication_window"),
    holoscan::field<&RawUploadGpuParams::input_window>("input_window"),
    holoscan::field<&RawUploadGpuParams::destination>("destination"),
    holoscan::field<&RawUploadGpuParams::device>("device").range(int64_t{0}, int64_t{255}));

class RawUploadGpuOp final : public Operator<RawUploadGpuParams> {
 public:
  ~RawUploadGpuOp() override;
  void setup(OperatorSpec& spec) override;
  Contract contract() const override;
  void start() override;
  expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;

  Input<CompressedBatch> input;
  Output<DecodedBatch> output;

 private:
  int64_t last_capture_ns_ = -1;
  uint64_t batch_id_{};
  int device_{-1};
  ::dectris::BatchWindow* publication_window_{};
  ::dectris::BatchWindow* input_window_{};
  uint32_t active_width_{static_cast<uint32_t>(DETECTOR_WIDTH)};
  uint32_t active_height_{static_cast<uint32_t>(DETECTOR_HEIGHT)};
  uint64_t active_series_id_{};
  bool active_series_{};
};

}  // namespace holoscan::ops
