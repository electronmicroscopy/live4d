#pragma once

#include <zmq.h>
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

    void* zmq_ctx_ = nullptr;
    void* zmq_socket_ = nullptr;
    zmq_msg_t msg_;
    bool msg_initialized_ = false;

    holoscan::Application* app_ptr_ = nullptr;

    size_t batch_count_{0};
    size_t total_images_emitted_{0};  // Total number of images emitted across all batches
    size_t total_batches_emitted_{0};  // Total batches emitted

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
};

} // namespace holoscan::ops
