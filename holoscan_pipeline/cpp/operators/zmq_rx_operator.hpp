#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <zmq.h>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/execution_context.hpp>
#include "constants.h"
#include "batch_window.hpp"
#include "pipeline_types.hpp"

namespace holoscan::ops {
struct ZmqRxParams {
  std::string endpoint{dectris::kDefaultZmqEndpoint};
  int64_t batch_size{static_cast<int64_t>(dectris::kMaxBatchSize)};
  uint64_t publication_window{};
};
HOLOSCAN_PARAM_SCHEMA(
    ZmqRxParams, holoscan::param_schema("holoscan.dectris.ZmqRxParams", 1),
    holoscan::field<&ZmqRxParams::endpoint>("endpoint"),
    holoscan::field<&ZmqRxParams::batch_size>("batch_size").range(int64_t{1}, int64_t{1000}),
    holoscan::field<&ZmqRxParams::publication_window>("publication_window"));

class ZmqRxOp final : public Operator<ZmqRxParams> {
 public:
  ~ZmqRxOp() override;
  void setup(OperatorSpec& spec) override;
  Contract contract() const override;
  void start() override;
  expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;
  Output<CompressedBatch> output;
 private:
  expected<void, Error> emit_batch(bool start = false, bool end = false, bool ready = false);
  void cleanup_zmq();
  void* zmq_ctx_ = nullptr;
  void* zmq_socket_ = nullptr;
  zmq_msg_t msg_;
  bool msg_initialized_ = false;
  bool ready_ = false;
  bool active_ = false;
  bool pending_end_ = false;
  uint64_t series_images_ = 0;
  size_t active_image_bytes_ = dectris::kDetectorPixels * sizeof(uint16_t);
  uint32_t active_width_ = static_cast<uint32_t>(dectris::kDetectorWidth);
  uint32_t active_height_ = static_cast<uint32_t>(dectris::kDetectorHeight);
  std::vector<uint8_t> staging_;
  CompressedBatch metadata_;
  uint64_t batch_id_ = 0;
  ::dectris::BatchWindow* publication_window_{};
};
}  // namespace holoscan::ops
