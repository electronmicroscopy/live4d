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
#include "rdma_session.hpp"

namespace holoscan::ops {

struct RdmaRxParams {
  uint64_t qp_handle{};
  uint64_t ring_address{};
  uint64_t ring_device_address{};
  uint64_t ring_bytes{};
  std::string ring_memory{"gpu"};
  uint64_t credit_address{};
  uint64_t credit_rkey{};
  uint64_t publication_window{};
  uint64_t session_handle{};
  int64_t batch_size{128};
  int64_t max_batch{512};
  int64_t device{};
};

HOLOSCAN_PARAM_SCHEMA(
    RdmaRxParams, holoscan::param_schema("holoscan.dectris.RdmaRxParams", 2),
    holoscan::field<&RdmaRxParams::qp_handle>("qp_handle"),
    holoscan::field<&RdmaRxParams::ring_address>("ring_address"),
    holoscan::field<&RdmaRxParams::ring_device_address>("ring_device_address"),
    holoscan::field<&RdmaRxParams::ring_bytes>("ring_bytes"),
    holoscan::field<&RdmaRxParams::ring_memory>("ring_memory"),
    holoscan::field<&RdmaRxParams::credit_address>("credit_address"),
    holoscan::field<&RdmaRxParams::credit_rkey>("credit_rkey"),
    holoscan::field<&RdmaRxParams::publication_window>("publication_window"),
    holoscan::field<&RdmaRxParams::session_handle>("session_handle"),
    holoscan::field<&RdmaRxParams::batch_size>("batch_size").range(int64_t{1}, int64_t{1000}),
    holoscan::field<&RdmaRxParams::max_batch>("max_batch").range(int64_t{1}, int64_t{1024}),
    holoscan::field<&RdmaRxParams::device>("device").range(int64_t{0}, INT64_MAX));

class RdmaRxOp final : public Operator<RdmaRxParams> {
 public:
  ~RdmaRxOp() override;
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
  };
  struct CudaEvent {
    cudaEvent_t handle{};
    ~CudaEvent();
  };
  struct FrameState {
    WireHeader header{};
    size_t offset{};
    size_t span{};
    uint32_t images_consumed{};
    std::shared_ptr<CudaEvent> last_read;
  };
  struct CreditState {
    uint64_t position{};
    std::shared_ptr<CudaEvent> event;
  };
  enum class ControlKind : uint8_t { kNone, kReady, kStart, kEnd };

  expected<void, Error> poll_completions();
  expected<void, Error> poll_credits();
  expected<void, Error> retry_credit_write();
  expected<void, Error> repost_recvs(int consumed);
  expected<void, Error> read_next_frame(uint32_t immediate);
  expected<void, Error> emit_control(ExecutionContext& context, ControlKind kind);
  expected<void, Error> record_credit(uint64_t position,
                                      const std::shared_ptr<CudaEvent>& event = {});
  expected<void, Error> flush_gpudirect_writes();
  expected<void, Error> session_control();
  expected<void, Error> drain_session();
  void reset_connection_state();
  void clear_state() noexcept;

  imx_qp* qp_{};  // Borrowed from the Python QueuePair owner.
  ::dectris::RdmaSession* session_{};  // Owner outlives the runtime session.
  uint64_t session_generation_{};
  bool session_end_emitted_{};
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

  uint64_t next_sequence_{};
  uint64_t reader_position_{};
  uint64_t safe_credit_position_{};
  uint64_t sent_credit_position_{};
  uint64_t next_batch_id_{};
  ::dectris::BatchWindow* publication_window_{};
  int64_t last_capture_ns_{-1};
  uint64_t active_series_id_{};
  uint64_t active_series_images_{};
  int recv_deficit_{};
  bool active_series_{};
  bool ready_emitted_{};

  std::vector<uint32_t> immediate_scratch_;
  std::deque<uint32_t> pending_immediates_;
  std::deque<CreditState> pending_credits_;
  std::optional<FrameState> frame_;
  ControlKind pending_control_{ControlKind::kNone};
  uint64_t pending_control_series_id_{};
  uint64_t pending_control_images_{};
  std::vector<uint8_t> pending_start_metadata_;
};

}  // namespace holoscan::ops
