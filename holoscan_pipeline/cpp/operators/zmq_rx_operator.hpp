#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
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
  // TCP connections to the sender. The DCU paces each connection (about 60k binned beam-on
  // frames/s), so 120 kHz needs several. With more than one, the sender's round robin can
  // reorder messages across connections; the receiver then tolerates that (see compute_multi).
  int64_t n_connections{1};
};
HOLOSCAN_PARAM_SCHEMA(
    ZmqRxParams, holoscan::param_schema("holoscan.dectris.ZmqRxParams", 1),
    holoscan::field<&ZmqRxParams::endpoint>("endpoint"),
    holoscan::field<&ZmqRxParams::batch_size>("batch_size").range(int64_t{1}, int64_t{1000}),
    holoscan::field<&ZmqRxParams::publication_window>("publication_window"),
    holoscan::field<&ZmqRxParams::n_connections>("n_connections").range(int64_t{1},
                                                                         int64_t{16}));

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
  expected<void, Error> compute_multi(ExecutionContext& context, size_t target);
  expected<void, Error> open_series(const uint8_t* data, size_t bytes);
  expected<void, Error> stage_image(const uint8_t* data, size_t bytes);
  expected<void, Error> finish_series();
  expected<void, Error> emit_end();
  int next_message();
  bool hold_message();
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
  // Several connections (n_connections > 1): the series' END can overtake its last images, the
  // next series' messages can overtake this series' END, and a START can arrive after its
  // first images. Messages of another series are held until this one is closed.
  bool end_seen_ = false;
  uint64_t announced_images_ = 0;     // number_of_images of the START (0 = not announced)
  bool has_finished_ = false;         // finished_series_ is valid
  uint64_t finished_series_ = 0;      // the last series closed: its stragglers are dropped
  uint64_t dropped_stragglers_ = 0;
  std::chrono::steady_clock::time_point last_own_message_{};
  std::deque<zmq_msg_t> held_;        // messages of another series, in arrival order
  std::deque<zmq_msg_t> replay_;      // held messages to process before the socket
  size_t held_bytes_ = 0;
};
}  // namespace holoscan::ops
