#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <vector>

#include <zmq.h>
#include <chrono>
#include <stdexcept>
#include <thread>

#include <holoscan/holoscan.hpp>
#include <nvtx3/nvToolsExt.h>

#include "constants.h"
#include "zmq_rx_operator.hpp"
#include "fast_cbor_parse.h"


namespace holoscan::ops {

ZmqRxOp::ZmqRxOp() {}

ZmqRxOp::~ZmqRxOp()
{
  cleanup_zmq();
  if (h_pinned_compressed_) {
    cudaFreeHost(h_pinned_compressed_);
    h_pinned_compressed_ = nullptr;
  }
}

void ZmqRxOp::initialize() {
    Operator::initialize();
}

void ZmqRxOp::setup(OperatorSpec& spec)
{
  app_ptr_ = spec.fragment()->application();

  spec.param(endpoint_,
             "endpoint",
             "ZMQ endpoint",
             "DECTRIS Stream2 ZMQ PUSH endpoint to connect to.",
             std::string(dectris::kDefaultZmqEndpoint));
  spec.param(batch_size_,
             "batch_size",
             "Batch size",
             "Number of image messages to group per compressed batch.",
             static_cast<int64_t>(dectris::kMaxBatchSize));
  spec.output<CompressedBatch>("output");
}

void ZmqRxOp::start() {
  batch_count_ = 0;
  total_images_emitted_ = 0;
  total_batches_emitted_ = 0;
  current_pool_index_ = 0;
  const size_t batch_size = configured_batch_size();

  zmq_ctx_ = zmq_ctx_new();
  if (zmq_ctx_ == nullptr) {
    throw std::runtime_error("ZmqRxOp: failed to create ZMQ context");
  }
  zmq_socket_ = zmq_socket(zmq_ctx_, ZMQ_PULL);
  if (zmq_socket_ == nullptr) {
    throw std::runtime_error("ZmqRxOp: failed to create ZMQ PULL socket");
  }
  int rc = zmq_connect(zmq_socket_, endpoint_.get().c_str());
  if (rc != 0) {
    throw std::runtime_error(std::string("ZmqRxOp: failed to connect to ") + endpoint_.get() +
                             ": " + zmq_strerror(zmq_errno()));
  }
  zmq_msg_init(&msg_);
  msg_initialized_ = true;

  // Calculate worst-case compressed size per image
  // For BSLZ4: Dectris header (12 bytes) + worst-case LZ4 (slightly larger than uncompressed)
  size_t uncompressed_size = dectris::kDetectorHeight * dectris::kDetectorWidth * sizeof(uint16_t);
  max_compressed_size_per_image_ = 12 + uncompressed_size + (uncompressed_size / 255) + 16;
  max_compressed_batch_size_ = max_compressed_size_per_image_ * dectris::kMaxBatchSize;

  // Allocate pinned host memory for compressed data (POOL_SIZE sections for circular buffer)
  size_t total_pinned_size = max_compressed_batch_size_ * dectris::kPoolSize;
  cudaError_t err = cudaMallocHost(&h_pinned_compressed_, total_pinned_size);
  if (err != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Failed to allocate pinned memory: {}", cudaGetErrorString(err));
    throw std::runtime_error("Failed to allocate pinned memory");
  }

  HOLOSCAN_LOG_INFO("ZmqRxOp: Allocated {} MB pinned memory for compressed data (POOL_SIZE={}, {} MB per slot)",
                    total_pinned_size / (1024 * 1024), dectris::kPoolSize,
                    max_compressed_batch_size_ / (1024 * 1024));

  // Allocate circular buffer pools for metadata
  matx::make_tensor(compressed_sizes_pool_,
                    {dectris::kPoolSize, dectris::kMaxBatchSize},
                    matx::MATX_HOST_MEMORY);
  matx::make_tensor(compressed_offsets_pool_,
                    {dectris::kPoolSize, dectris::kMaxBatchSize},
                    matx::MATX_HOST_MEMORY);
  matx::make_tensor(image_ids_pool_,
                    {dectris::kPoolSize, dectris::kMaxBatchSize},
                    matx::MATX_HOST_MEMORY);

  // Initialize with zeros
  (compressed_sizes_pool_ = 0).run();
  (compressed_offsets_pool_ = 0).run();
  (image_ids_pool_ = 0).run();

  HOLOSCAN_LOG_INFO("ZmqRxOp: Start endpoint={} batch_size={} max_batch={}",
                    endpoint_.get(), batch_size, dectris::kMaxBatchSize);
}

void ZmqRxOp::stop() {
  if (h_pinned_compressed_) {
    cudaFreeHost(h_pinned_compressed_);
    h_pinned_compressed_ = nullptr;
  }
  cleanup_zmq();

  HOLOSCAN_LOG_INFO("ZmqRxOp: Stop");
}

void ZmqRxOp::compute(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context)
{
  const uint8_t* msg_data;
  size_t msg_size;
  const size_t batch_size = configured_batch_size();

  // Track compressed data offset in pinned buffer for this pool slot
  // Each pool slot has its own section of the pinned buffer
  size_t pool_buffer_offset = current_pool_index_ * max_compressed_batch_size_;
  size_t total_compressed_size = 0;
  size_t valid_images = 0;
  bool saw_end = false;

  // Loop until we receive a full batch or stop/end message.
  while (batch_count_ < batch_size) {
    {
      nvtx3::scoped_range range("ZmqRxOp::compute: receive msg", nvtx3::rgb{255, 0, 0}); // Red
      int recv_rc = zmq_msg_recv(&msg_, zmq_socket_, 0);
      if (recv_rc < 0) {
        HOLOSCAN_LOG_WARN("{}: ZMQ receive failed: {}", name(), zmq_strerror(zmq_errno()));
        continue;
      }
      msg_data = (const uint8_t*)zmq_msg_data(&msg_);
      msg_size = zmq_msg_size(&msg_);
    }

    {
      nvtx3::scoped_range range("ZmqRxOp::compute: process msg", nvtx3::rgb{0, 255, 0}); // Green

      // Use fast CBOR parser for message type detection
      zmq_message_type_t msg_type = fast_detect_message_type(msg_data, msg_size);

      if (msg_type == ZMQ_MSG_START) {
        continue; // Continue to next message
      }

      if (msg_type == ZMQ_MSG_END) {
        saw_end = true;
        break;
      }

      if (msg_type == ZMQ_MSG_IMAGE) {
        // Extract compressed data from CBOR using fast parser
        const uint8_t* compressed_data = nullptr;
        size_t compressed_size = 0;
        size_t original_size = 0;
        size_t elem_size = 0;
        uint64_t image_id = 0;

        int result = fast_extract_compressed_bslz4(
            msg_data, msg_size,
            &compressed_data, &compressed_size,
            &original_size, &elem_size, &image_id);

        if (result != 0) {
          HOLOSCAN_LOG_WARN("{}: Failed to extract compressed data from message", name());
          continue;
        }

        // Check buffer space
        if (total_compressed_size + compressed_size > max_compressed_batch_size_) {
          HOLOSCAN_LOG_ERROR("{}: Compressed data exceeds buffer size", name());
          break;
        }

        // Store metadata in circular buffer pools
        compressed_offsets_pool_(current_pool_index_, batch_count_) = total_compressed_size;
        compressed_sizes_pool_(current_pool_index_, batch_count_) = compressed_size;
        image_ids_pool_(current_pool_index_, batch_count_) = image_id;

        // Copy compressed data to this pool slot's section of the pinned buffer
        std::memcpy(h_pinned_compressed_ + pool_buffer_offset + total_compressed_size,
                    compressed_data, compressed_size);
        total_compressed_size += compressed_size;

        batch_count_++;
        valid_images++;
      } else {
        HOLOSCAN_LOG_WARN("{}: Unknown message type received (size: {} bytes)", name(), msg_size);
      }
    }
  }

  if (batch_count_ > 0) {
    emit_current_batch(op_output, pool_buffer_offset, valid_images, total_compressed_size);
  }

  if (saw_end) {
    HOLOSCAN_LOG_INFO("{}: Stream ended after emitting {} images in {} batches",
                      name(), total_images_emitted_, total_batches_emitted_);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    if (app_ptr_ != nullptr) {
      app_ptr_->stop_execution();
    }
  }
}

size_t ZmqRxOp::configured_batch_size() const {
  const int64_t requested = batch_size_.get();
  if (requested <= 0) {
    throw std::runtime_error("ZmqRxOp: batch_size must be positive");
  }
  return std::min(static_cast<size_t>(requested), dectris::kMaxBatchSize);
}

void ZmqRxOp::emit_current_batch(OutputContext& op_output,
                                 size_t pool_buffer_offset,
                                 size_t valid_images,
                                 size_t total_compressed_size) {
  nvtx3::scoped_range range("ZmqRxOp::compute: emit batch", nvtx3::rgb{255, 255, 0});
  matx::index_t idx = static_cast<matx::index_t>(current_pool_index_);

  auto sizes_1d = matx::slice<1>(
      matx::slice(compressed_sizes_pool_, {idx, 0}, {idx + 1, matx::matxEnd}),
      {0, 0}, {matx::matxDropDim, matx::matxEnd});

  auto offsets_1d = matx::slice<1>(
      matx::slice(compressed_offsets_pool_, {idx, 0}, {idx + 1, matx::matxEnd}),
      {0, 0}, {matx::matxDropDim, matx::matxEnd});

  auto image_ids_1d = matx::slice<1>(
      matx::slice(image_ids_pool_, {idx, 0}, {idx + 1, matx::matxEnd}),
      {0, 0}, {matx::matxDropDim, matx::matxEnd});

  CompressedBatch output_batch(
      h_pinned_compressed_ + pool_buffer_offset,
      std::make_shared<decltype(sizes_1d)>(sizes_1d),
      std::make_shared<decltype(offsets_1d)>(offsets_1d),
      std::make_shared<decltype(image_ids_1d)>(image_ids_1d),
      valid_images,
      total_batches_emitted_ + 1);

  op_output.emit(output_batch, "output");

  total_images_emitted_ += valid_images;
  total_batches_emitted_++;

  HOLOSCAN_LOG_INFO("{}: Emitting batch {} ({} images, {} bytes compressed, pool_idx: {})",
                    name(), total_batches_emitted_, valid_images, total_compressed_size,
                    current_pool_index_);

  current_pool_index_ = (current_pool_index_ + 1) % dectris::kPoolSize;
  batch_count_ = 0;
}

void ZmqRxOp::cleanup_zmq() {
  if (msg_initialized_) {
    zmq_msg_close(&msg_);
    msg_initialized_ = false;
  }
  if (zmq_socket_ != nullptr) {
    zmq_close(zmq_socket_);
    zmq_socket_ = nullptr;
  }
  if (zmq_ctx_ != nullptr) {
    zmq_ctx_term(zmq_ctx_);
    zmq_ctx_ = nullptr;
  }
}

} // namespace holoscan::ops
