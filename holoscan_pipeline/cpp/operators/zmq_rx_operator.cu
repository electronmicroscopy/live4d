#include "zmq_rx_operator.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include "fast_cbor_parse.h"

namespace holoscan::ops {
namespace {
constexpr size_t kImageBytes = dectris::kDetectorPixels * sizeof(uint16_t);
constexpr size_t kMaxCompressedImage = 12 + kImageBytes + kImageBytes / 255 + 16;
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
  ended_ = false;
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
  if (ended_) return {};
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
    if (type == ZMQ_MSG_START) continue;
    if (type == ZMQ_MSG_END) {
      ended_ = true;
      return emit_batch(true);
    }
    if (type != ZMQ_MSG_IMAGE)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Unknown Stream2 message"});
    const uint8_t* compressed = nullptr;
    size_t size = 0, original_size = 0, elem_size = 0;
    uint64_t image_id = 0;
    if (fast_extract_compressed_bslz4(data, bytes, &compressed, &size,
                                    &original_size, &elem_size, &image_id) != 0 ||
        original_size != kImageBytes || elem_size != sizeof(uint16_t) ||
        size > kMaxCompressedImage || size < 12)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid 96x96 uint16 BSLZ4 image"});
    metadata_.offsets.push_back(staging_.size());
    metadata_.sizes.push_back(size);
    metadata_.image_ids.push_back(image_id);
    staging_.insert(staging_.end(), compressed, compressed + size);
    if (metadata_.image_ids.size() == target) return emit_batch(false);
  }
  return {};
}

expected<void, Error> ZmqRxOp::emit_batch(bool end_of_stream) {
  // Empty acquisitions and exact batch multiples carry an explicit final marker.
  const std::array<int64_t, 1> shape{static_cast<int64_t>(std::max(size_t{1}, staging_.size()))};
  HOLOSCAN_TRY(auto loan, output.allocate_tensor(shape));
  HOLOSCAN_TRY(loan.write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
    if (!staging_.empty()) std::memcpy(writer.data(), staging_.data(), staging_.size());
  }));
  metadata_.batch_id = ++batch_id_;
  metadata_.end_of_stream = end_of_stream;
  HOLOSCAN_TRY(output.emit_tensor(std::move(loan), metadata_));
  staging_.clear();
  metadata_.image_ids.clear();
  metadata_.sizes.clear();
  metadata_.offsets.clear();
  return {};
}

void ZmqRxOp::stop() { cleanup_zmq(); }
void ZmqRxOp::cleanup_zmq() {
  if (msg_initialized_) { zmq_msg_close(&msg_); msg_initialized_ = false; }
  if (zmq_socket_) { zmq_close(zmq_socket_); zmq_socket_ = nullptr; }
  if (zmq_ctx_) { zmq_ctx_term(zmq_ctx_); zmq_ctx_ = nullptr; }
}
}  // namespace holoscan::ops
