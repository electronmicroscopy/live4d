#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "batch_window.hpp"
#include "imxrx.h"
#include "pipeline_types.hpp"

namespace holoscan::ops {

struct RdmaRingsHostRxParams {
  uint64_t qp_handle{};
  uint64_t ring_address{};
  uint64_t ring_bytes{};
  int64_t ring_count{1};
  uint64_t credit_address{};
  uint64_t credit_rkey{};
  std::vector<uint64_t> publication_windows{};
  std::vector<uint64_t> acquisition_windows{};
  uint64_t acquisition_limit{};
  int64_t batch_size{128};
  int64_t max_batch{512};
};

HOLOSCAN_PARAM_SCHEMA(
    RdmaRingsHostRxParams, holoscan::param_schema("holoscan.dectris.RdmaRingsHostRxParams", 1),
    holoscan::field<&RdmaRingsHostRxParams::qp_handle>("qp_handle"),
    holoscan::field<&RdmaRingsHostRxParams::ring_address>("ring_address"),
    holoscan::field<&RdmaRingsHostRxParams::ring_bytes>("ring_bytes"),
    holoscan::field<&RdmaRingsHostRxParams::ring_count>("ring_count").range(int64_t{1}, int64_t{256}),
    holoscan::field<&RdmaRingsHostRxParams::credit_address>("credit_address"),
    holoscan::field<&RdmaRingsHostRxParams::credit_rkey>("credit_rkey"),
    holoscan::field<&RdmaRingsHostRxParams::publication_windows>("publication_windows"),
    holoscan::field<&RdmaRingsHostRxParams::acquisition_windows>("acquisition_windows"),
    holoscan::field<&RdmaRingsHostRxParams::acquisition_limit>("acquisition_limit"),
    holoscan::field<&RdmaRingsHostRxParams::batch_size>("batch_size").range(int64_t{1}, int64_t{1000}),
    holoscan::field<&RdmaRingsHostRxParams::max_batch>("max_batch").range(int64_t{1}, int64_t{1024}));

class RdmaRingsHostRxOp final : public Operator<RdmaRingsHostRxParams> {
 public:
  ~RdmaRingsHostRxOp() override;
  void setup(OperatorSpec& spec) override;
  Contract contract() const override;
  void start() override;
  expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;

  Output<CompressedBatch> output;

 private:
  struct WireHeader {
    uint16_t kind{};
    uint64_t sequence{};
    uint64_t position{};
    uint64_t data_size{};
    uint64_t series_id{};
    uint64_t image_id{};
    uint32_t image_count{};
    uint32_t dim_x{};
    uint32_t dim_y{};
    uint16_t dtype{};
    uint16_t compression{};
    uint64_t uncompressed_size{};
    uint16_t channel{};
    uint16_t channel_count{};
    uint16_t ring{};
  };
  struct FrameState {
    WireHeader header{};
    size_t ring{};
    size_t offset{};
    uint32_t images_consumed{};
  };
  // Each ring is an independent frame stream with its own sequence, positions, and credit.
  struct RingState {
    uint64_t next_sequence{};
    uint64_t reader_position{};
    uint64_t safe_credit_position{};
    uint64_t sent_credit_position{};
    std::deque<uint32_t> immediates;  // Low sequence bits, in completion order.
  };
  enum class ControlKind : uint8_t { kNone, kReady, kStart, kEnd };

  expected<void, Error> poll_completions();
  expected<void, Error> retry_credit_writes();
  expected<void, Error> repost_recvs(int consumed);
  expected<void, Error> check_series_continues() const;
  expected<void, Error> read_next_frame(size_t ring, uint32_t sequence_bits);
  expected<void, Error> emit_control(ControlKind kind);
  expected<void, Error> emit_batch(uint64_t series_id);
  expected<void, Error> record_credit(size_t ring, uint64_t position);
  bool series_admitted() const;
  const uint8_t* ring_base(size_t ring) const {
    return reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ring_address_)) +
           ring * ring_bytes_;
  }
  void clear_state() noexcept;

  imx_qp* qp_{};  // Borrowed from the Python QueuePair owner.
  uint64_t ring_address_{};
  size_t ring_bytes_{};
  uint64_t credit_address_{};
  uint32_t credit_rkey_{};
  size_t batch_size_{};
  size_t max_batch_{};
  uint64_t acquisition_limit_{};
  size_t lanes_{1};
  size_t active_image_bytes_{};
  uint32_t active_width_{};
  uint32_t active_height_{};

  std::vector<RingState> rings_;
  size_t current_ring_{};  // Ring of the active series, or of the next START.
  uint64_t next_batch_id_{};
  uint64_t series_ordinal_{};  // STARTs read so far; the next series goes to lane % lanes_.
  uint32_t active_destination_{};
  std::vector<::dectris::BatchWindow*> publication_windows_, acquisition_windows_;
  uint64_t active_series_id_{};
  uint64_t active_series_images_{};
  int recv_deficit_{};
  bool active_series_{};
  bool ready_emitted_{};

  std::vector<uint32_t> immediate_scratch_;
  std::optional<FrameState> frame_;
  ControlKind pending_control_{ControlKind::kNone};
  uint64_t pending_control_series_id_{};
  uint64_t pending_control_images_{};
  std::vector<uint8_t> pending_start_metadata_;
  std::vector<uint8_t> staging_;
  CompressedBatch metadata_;
};

}  // namespace holoscan::ops
