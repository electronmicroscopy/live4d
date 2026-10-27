#pragma once

#include <cstdint>
#include <climits>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "batch_window.hpp"
#include "imxrx.h"
#include "pipeline_types.hpp"

namespace holoscan::ops {

// imagemux protocol 3: the sender writes series round robin into ring_count equal rings.
// Ring i occupies [ring_address + i * ring_bytes, +ring_bytes) of one registered allocation,
// and its credit word is at credit_address + 8 * i.
struct RdmaRingsRxParams {
  uint64_t qp_handle{};
  uint64_t ring_address{};
  uint64_t ring_device_address{};
  uint64_t ring_bytes{};
  int64_t ring_count{1};
  std::string ring_memory{"gpu"};
  uint64_t credit_address{};
  uint64_t credit_rkey{};
  uint64_t publication_window{};
  int64_t batch_size{128};
  int64_t max_batch{512};
  int64_t device{};
};

HOLOSCAN_PARAM_SCHEMA(
    RdmaRingsRxParams, holoscan::param_schema("holoscan.dectris.RdmaRingsRxParams", 1),
    holoscan::field<&RdmaRingsRxParams::qp_handle>("qp_handle"),
    holoscan::field<&RdmaRingsRxParams::ring_address>("ring_address"),
    holoscan::field<&RdmaRingsRxParams::ring_device_address>("ring_device_address"),
    holoscan::field<&RdmaRingsRxParams::ring_bytes>("ring_bytes"),
    holoscan::field<&RdmaRingsRxParams::ring_count>("ring_count").range(int64_t{1}, int64_t{256}),
    holoscan::field<&RdmaRingsRxParams::ring_memory>("ring_memory"),
    holoscan::field<&RdmaRingsRxParams::credit_address>("credit_address"),
    holoscan::field<&RdmaRingsRxParams::credit_rkey>("credit_rkey"),
    holoscan::field<&RdmaRingsRxParams::publication_window>("publication_window"),
    holoscan::field<&RdmaRingsRxParams::batch_size>("batch_size").range(int64_t{1}, int64_t{1000}),
    holoscan::field<&RdmaRingsRxParams::max_batch>("max_batch").range(int64_t{1}, int64_t{1024}),
    holoscan::field<&RdmaRingsRxParams::device>("device").range(int64_t{0}, INT64_MAX));

class RdmaRingsRxOp final : public Operator<RdmaRingsRxParams> {
 public:
  ~RdmaRingsRxOp() override;
  void setup(OperatorSpec& spec) override;
  Contract contract() const override;
  void start() override;
  expected<void, Error> compute(ExecutionContext& context) override;
  void stop() override;

  Output<DecodedBatch> output;

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
  struct CudaEvent {
    cudaEvent_t handle{};
    ~CudaEvent();
  };
  struct FrameState {
    WireHeader header{};
    size_t ring{};
    size_t offset{};
    uint32_t images_consumed{};
    std::shared_ptr<CudaEvent> last_read;
  };
  struct CreditState {
    uint64_t position{};
    std::shared_ptr<CudaEvent> event;
  };
  // Each ring is an independent frame stream with its own sequence, positions, and credit.
  struct RingState {
    uint64_t next_sequence{};
    uint64_t reader_position{};
    uint64_t safe_credit_position{};
    uint64_t sent_credit_position{};
    std::deque<uint32_t> immediates;  // Low sequence bits, in completion order.
    std::deque<CreditState> credits;
  };
  enum class ControlKind : uint8_t { kNone, kReady, kStart, kEnd };

  expected<void, Error> poll_completions();
  expected<void, Error> poll_credits();
  expected<void, Error> retry_credit_writes();
  expected<void, Error> repost_recvs(int consumed);
  expected<void, Error> check_series_continues() const;
  expected<void, Error> read_next_frame(size_t ring, uint32_t sequence_bits);
  expected<void, Error> emit_control(ExecutionContext& context, ControlKind kind);
  expected<void, Error> record_credit(size_t ring, uint64_t position,
                                      const std::shared_ptr<CudaEvent>& event = {});
  expected<void, Error> flush_gpudirect_writes();
  uint64_t ring_host_address(size_t ring) const { return ring_address_ + ring * ring_bytes_; }
  uint64_t ring_device_address(size_t ring) const {
    return ring_device_address_ + ring * ring_bytes_;
  }
  void clear_state() noexcept;

  imx_qp* qp_{};  // Borrowed from the Python QueuePair owner.
  uint64_t ring_address_{};
  uint64_t ring_device_address_{};
  size_t ring_bytes_{};
  bool gpu_ring_{};
  uint64_t credit_address_{};
  uint32_t credit_rkey_{};
  int device_{-1};
  size_t batch_size_{};
  size_t max_batch_{};
  size_t active_image_bytes_{};
  uint32_t active_width_{};
  uint32_t active_height_{};

  std::vector<RingState> rings_;
  size_t current_ring_{};  // Ring of the active series, or of the next START.
  size_t pending_credit_count_{};
  uint64_t next_batch_id_{};
  ::dectris::BatchWindow* publication_window_{};
  int64_t last_capture_ns_{-1};
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
};

}  // namespace holoscan::ops
