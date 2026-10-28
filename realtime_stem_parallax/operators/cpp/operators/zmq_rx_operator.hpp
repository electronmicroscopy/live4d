#pragma once

#include <zmq.h>
#include <chrono>
#include <string>

#include "matx.h"
#include <holoscan/holoscan.hpp>

#include "constants.h"
#include "pipeline_types.hpp"

namespace holoscan::ops {

class ZmqRxOp : public Operator {
  public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ZmqRxOp)

    ZmqRxOp();
    ~ZmqRxOp();

    void initialize() override;
    void start() override;
    void setup(OperatorSpec& spec) override;
    void compute(InputContext& op_input, OutputContext& op_output,
                ExecutionContext& context) override;
    void stop() override;

private:
    Parameter<std::string> endpoint_;
    Parameter<int64_t> batch_size_;
    Parameter<int64_t> detector_width_;
    Parameter<int64_t> detector_height_;
    Parameter<std::string> pixel_dtype_;
    Parameter<int64_t> n_connections_;
    Parameter<int64_t> expected_images_;

    // Derived from the parameters in start(); every incoming image is validated against
    // these (and the first image's dims) so a misconfigured geometry fails loudly instead
    // of decompressing into wrong-sized buffers.
    size_t elem_size_ = 0;
    size_t expected_image_bytes_ = 0;
    bool dims_checked_ = false;

    void* zmq_socket_ = nullptr;  // the process-wide stream connection (not owned, see .cu)
    zmq_msg_t msg_;
    bool msg_initialized_ = false;

    holoscan::Application* app_ptr_ = nullptr;

    size_t batch_count_{0};
    size_t pending_bytes_{0};  // compressed bytes of the current, not yet emitted batch
    size_t total_images_emitted_{0};  // Total number of images emitted across all batches
    size_t total_batches_emitted_{0};  // Total batches emitted

    // Series bookkeeping. One run of this operator handles one series (start, images, end).
    bool end_seen_{false};       // `end` received; still draining the other connections
    bool next_seen_{false};      // a message of the next series arrived (held for the next run)
    size_t leftover_messages_{0};  // messages of a series an earlier run already finished
    bool start_seen_{false};     // this series' `start` message was received
    bool series_known_{false};   // series_id_ is valid (from `start` or the first image)
    uint64_t series_id_{0};
    uint64_t announced_images_{0};  // number_of_images from `start` (0 = not announced)
    size_t skipped_messages_{0};  // messages that could not be decoded and were dropped
    size_t replay_left_{0};       // held messages (of this series) still to process first
    std::chrono::steady_clock::time_point last_own_message_;  // last message of this series

    // Pinned host memory for compressed data (persistent buffer)
    uint8_t* h_pinned_compressed_ = nullptr;
    size_t max_compressed_size_per_image_ = 0;
    size_t max_compressed_batch_size_ = 0;

    // Circular buffer pools for metadata.
    matx::tensor_t<size_t, 2> compressed_sizes_pool_;
    matx::tensor_t<size_t, 2> compressed_offsets_pool_;
    matx::tensor_t<uint64_t, 2> image_ids_pool_;

    size_t current_pool_index_ = 0;

    size_t configured_batch_size() const;
    void emit_current_batch(OutputContext& op_output, size_t pool_buffer_offset,
                            size_t valid_images, size_t total_compressed_size);
    void cleanup_zmq();
    void handle_start(const uint8_t* msg_data, size_t msg_size);
    bool next_message(bool* timed_out);
    bool hold_message();
    bool series_complete() const;
};

} // namespace holoscan::ops
