#include "zmq_rx_operator.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include "fast_cbor_parse.h"
#include "header_appendix_log.hpp"

namespace holoscan::ops {
namespace {
constexpr size_t kMaxImageBytes = dectris::kMaxDetectorPixels * sizeof(uint16_t);
constexpr size_t kBslz4BlockBytes = 8192;
constexpr size_t kMaxBslz4Blocks = (kMaxImageBytes + kBslz4BlockBytes - 1) / kBslz4BlockBytes;
constexpr size_t kMaxCompressedImage =
    12 + kMaxImageBytes + kMaxImageBytes / 255 + kMaxBslz4Blocks * (16 + 4);
constexpr size_t kMaxCompressedBatch = kMaxCompressedImage * dectris::kMaxBatchSize;
}
ZmqRxOp::~ZmqRxOp() { cleanup_zmq(); }

void ZmqRxOp::setup(OperatorSpec& spec) {
  spec.output(output, "output").produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kPinnedHost,
                         .dtype = DLDataType{kDLUInt, 8, 1}, .rank = 1U},
      .bounds = tensor_bounds(kMaxCompressedBatch),
      .storage = TensorOutputStorage::kRuntimePool});
}

Contract ZmqRxOp::contract() const {
  // ZMQ's edge-triggered FD is not a level-readable source. Poll nonblocking at
  // a bounded cadence and retain partial batches between activations instead.
  Contract result;
  result.trigger(OnClock{.period = std::chrono::microseconds(100)});
  return result;
}

void ZmqRxOp::start() {
  publication_window_ = ::dectris::access(params().publication_window);
  ready_ = active_ = pending_end_ = false;
  active_image_bytes_ = dectris::kDetectorPixels * sizeof(uint16_t);
  active_width_ = static_cast<uint32_t>(dectris::kDetectorWidth);
  active_height_ = static_cast<uint32_t>(dectris::kDetectorHeight);
  batch_id_ = 0;
  metadata_ = {};
  metadata_.image_ids.reserve(dectris::kMaxBatchSize);
  metadata_.sizes.reserve(dectris::kMaxBatchSize);
  metadata_.offsets.reserve(dectris::kMaxBatchSize);
  staging_.clear();
  staging_.reserve(kMaxCompressedBatch);
  zmq_ctx_ = zmq_ctx_new();
  if (!zmq_ctx_) throw std::runtime_error("ZmqRxOp: cannot create context");
  zmq_socket_ = zmq_socket(zmq_ctx_, ZMQ_PULL);
  if (!zmq_socket_) throw std::runtime_error("ZmqRxOp: cannot create socket");
  int linger = 0;
  zmq_setsockopt(zmq_socket_, ZMQ_LINGER, &linger, sizeof(linger));
  if (zmq_connect(zmq_socket_, params().endpoint.c_str()) != 0)
    throw std::runtime_error("ZmqRxOp: cannot connect to endpoint");
  if (zmq_msg_init(&msg_) != 0) throw std::runtime_error("ZmqRxOp: cannot initialize message");
  msg_initialized_ = true;
}

expected<void, Error> ZmqRxOp::compute(ExecutionContext& context) {
  if (context.cancellation().stop_requested()) return {};
  if (publication_window_ && !publication_window_->admits(batch_id_ + 1)) return {};
  if (!ready_) {
    ready_ = true;
    return emit_batch(false, false, true);
  }
  if (pending_end_) {
    pending_end_ = false;
    active_ = false;
    return emit_batch(false, true);
  }
  const auto target = static_cast<size_t>(context.params<ZmqRxParams>().batch_size);
  // Bound control/unknown messages too, so idle and cancellation stay responsive.
  for (size_t received = 0; received < dectris::kMaxBatchSize + 2; ++received) {
    if (context.cancellation().stop_requested()) return {};
    if (zmq_msg_recv(&msg_, zmq_socket_, ZMQ_DONTWAIT) < 0) {
      if (zmq_errno() == EAGAIN || zmq_errno() == EINTR) return {};
      return make_unexpected(Error{ErrorCode::kFailure, "ZMQ receive failed"});
    }
    const auto* data = static_cast<const uint8_t*>(zmq_msg_data(&msg_));
    const size_t bytes = zmq_msg_size(&msg_);
    const auto type = fast_detect_message_type(data, bytes);
    if (type == ZMQ_MSG_START) {
      dectris_start_metadata_t start_metadata{};
      if (active_ || bytes > 16384 ||
          fast_extract_start_metadata(data, bytes, &start_metadata) != 0)
        return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid START image geometry"});
      metadata_.series_id = start_metadata.series_id;
      active_width_ = start_metadata.image_size_x;
      active_height_ = start_metadata.image_size_y;
      active_image_bytes_ = static_cast<size_t>(active_width_) * active_height_ * sizeof(uint16_t);
      active_ = true;
      series_images_ = 0;
      metadata_.start_metadata.assign(data, data + bytes);
      ::dectris::log_header_appendix(start_metadata.series_id, data, bytes);
      return emit_batch(true);
    }
    if (type == ZMQ_MSG_END) {
      if (!active_)
        return make_unexpected(Error{ErrorCode::kProtocolError, "END has no matching START"});
      pending_end_ = true;
      if (!metadata_.image_ids.empty()) return emit_batch();
      pending_end_ = false;
      active_ = false;
      return emit_batch(false, true);
    }
    if (type != ZMQ_MSG_IMAGE)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Unknown Stream2 message"});
    if (!active_)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Image has no matching START"});
    const uint8_t* compressed = nullptr;
    size_t size = 0, original_size = 0, elem_size = 0;
    uint32_t image_width = 0, image_height = 0;
    uint64_t image_id = 0;
    if (fast_extract_compressed_bslz4(data, bytes, &compressed, &size,
                                    &original_size, &elem_size, &image_width, &image_height,
                                    &image_id) != 0 ||
        original_size != active_image_bytes_ || elem_size != sizeof(uint16_t) ||
        image_width != active_width_ || image_height != active_height_ ||
        size > kMaxCompressedImage || size < 12)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid BSLZ4 image"});
    metadata_.offsets.push_back(staging_.size());
    metadata_.sizes.push_back(size);
    metadata_.image_ids.push_back(image_id);
    ++series_images_;
    staging_.insert(staging_.end(), compressed, compressed + size);
    if (metadata_.image_ids.size() == target) return emit_batch();
  }
  return {};
}

expected<void, Error> ZmqRxOp::emit_batch(bool start, bool end, bool ready) {
  // Empty acquisitions and exact batch multiples carry an explicit final marker.
  const std::array<int64_t, 1> shape{static_cast<int64_t>(std::max(size_t{1}, staging_.size()))};
  HOLOSCAN_TRY(auto loan, output.allocate_tensor(shape));
  HOLOSCAN_TRY(loan.write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
    if (!staging_.empty()) std::memcpy(writer.data(), staging_.data(), staging_.size());
  }));
  metadata_.batch_id = ++batch_id_;
  metadata_.start_of_series = start;
  metadata_.end_of_series = end;
  metadata_.pipeline_ready = ready;
  metadata_.series_images = series_images_;
  HOLOSCAN_TRY(output.emit_tensor(std::move(loan), metadata_));
  staging_.clear();
  metadata_.image_ids.clear();
  metadata_.sizes.clear();
  metadata_.offsets.clear();
  metadata_.start_metadata.clear();
  return {};
}

void ZmqRxOp::stop() { cleanup_zmq(); }
void ZmqRxOp::cleanup_zmq() {
  if (msg_initialized_) { zmq_msg_close(&msg_); msg_initialized_ = false; }
  if (zmq_socket_) { zmq_close(zmq_socket_); zmq_socket_ = nullptr; }
  if (zmq_ctx_) { zmq_ctx_term(zmq_ctx_); zmq_ctx_ = nullptr; }
}
}  // namespace holoscan::ops
