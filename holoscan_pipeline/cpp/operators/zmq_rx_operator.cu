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
// Several connections: a series counts as complete once every announced image is in, or once
// no message of it has arrived for this long after its END (the other connections are drained).
constexpr auto kDrainQuiet = std::chrono::milliseconds(100);
// Messages of another series held at most, before the current series is closed anyway.
constexpr size_t kMaxHeldMessages = 65536;
constexpr size_t kMaxHeldBytes = size_t{512} << 20;
constexpr int kMultiRecvQueue = 50000;

// Move the front message of `from` to the back of `to` (zmq_msg_t must not be copied).
void move_front(std::deque<zmq_msg_t>& from, std::deque<zmq_msg_t>& to) {
  to.emplace_back();
  zmq_msg_init(&to.back());
  zmq_msg_move(&to.back(), &from.front());
  zmq_msg_close(&from.front());
  from.pop_front();
}
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
  end_seen_ = has_finished_ = false;
  announced_images_ = finished_series_ = dropped_stragglers_ = 0;
  held_bytes_ = 0;
  const int64_t connections = params().n_connections;
  if (connections > 1) {
    // A deep queue per connection absorbs the stream while a batch is being published.
    int queue = kMultiRecvQueue;
    zmq_setsockopt(zmq_socket_, ZMQ_RCVHWM, &queue, sizeof(queue));
  }
  // Each connect is its own TCP connection; the sender's PUSH round-robins across them.
  for (int64_t i = 0; i < connections; ++i)
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
  if (context.params<ZmqRxParams>().n_connections > 1)
    return compute_multi(context, static_cast<size_t>(context.params<ZmqRxParams>().batch_size));
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

// ── several connections ───────────────────────────────────────────────────────────────────
// The sender round-robins one message stream across the connections, so the order between them
// is lost: END can arrive before the last images of its series, the next series' START (and
// images) before this series' END, and a series' first images before its START. Messages of
// another series are held and replayed once the current series is closed; a series closes when
// every image its START announced is in, or when nothing of it arrived for kDrainQuiet after
// its END (or, without an END, after the next series began). Images of a series that was
// already closed are dropped with a warning. Downstream still sees START, images, END in order.

expected<void, Error> ZmqRxOp::compute_multi(ExecutionContext& context, size_t target) {
  if (pending_end_) {
    pending_end_ = false;
    return emit_end();
  }
  const auto now = std::chrono::steady_clock::now();
  for (size_t received = 0; received < dectris::kMaxBatchSize + 2; ++received) {
    if (context.cancellation().stop_requested()) return {};
    const int got = next_message();
    if (got < 0) return make_unexpected(Error{ErrorCode::kFailure, "ZMQ receive failed"});
    if (got == 0) {
      // Idle. Close the series once its connections have been quiet after its END, or after
      // the next series began without one.
      if (active_ && (end_seen_ || !held_.empty()) && now - last_own_message_ > kDrainQuiet) {
        if (!end_seen_)
          HOLOSCAN_LOG_WARN("ZmqRxOp: series {} sent no END before the next series; closing it "
                            "with {} images", metadata_.series_id, series_images_);
        return finish_series();
      }
      return {};
    }
    const auto* data = static_cast<const uint8_t*>(zmq_msg_data(&msg_));
    const size_t bytes = zmq_msg_size(&msg_);
    const auto type = fast_detect_message_type(data, bytes);
    if (type == ZMQ_MSG_UNKNOWN)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Unknown Stream2 message"});
    uint64_t series = 0;
    if (fast_extract_series_id(data, bytes, &series) != 0)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Stream2 message has no series_id"});
    if (has_finished_ && series == finished_series_ && !(active_ && series == metadata_.series_id)) {
      if (dropped_stragglers_++ == 0)
        HOLOSCAN_LOG_WARN("ZmqRxOp: dropping messages of series {}, which is already closed",
                          series);
      continue;
    }
    if (!active_) {
      if (type == ZMQ_MSG_START) {
        last_own_message_ = now;
        return open_series(data, bytes);
      }
      // Its START is still on another connection.
      if (!hold_message())
        return make_unexpected(Error{ErrorCode::kProtocolError,
                                     "Images without START exceed the hold limit"});
      continue;
    }
    if (series != metadata_.series_id) {
      // The next series overtook this one's END or last images.
      if (!hold_message()) {
        HOLOSCAN_LOG_WARN("ZmqRxOp: hold limit reached; closing series {} with {} images",
                          metadata_.series_id, series_images_);
        return finish_series();
      }
      continue;
    }
    last_own_message_ = now;
    if (type == ZMQ_MSG_START)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Repeated START for one series"});
    if (type == ZMQ_MSG_END) {
      end_seen_ = true;
      if (announced_images_ == 0 || series_images_ >= announced_images_) return finish_series();
      continue;
    }
    if (auto staged = stage_image(data, bytes); !staged) return staged;
    if (end_seen_ && series_images_ >= announced_images_) return finish_series();
    if (metadata_.image_ids.size() == target) return emit_batch();
  }
  return {};
}

expected<void, Error> ZmqRxOp::open_series(const uint8_t* data, size_t bytes) {
  dectris_start_metadata_t start_metadata{};
  if (bytes > 16384 || fast_extract_start_metadata(data, bytes, &start_metadata) != 0)
    return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid START image geometry"});
  metadata_.series_id = start_metadata.series_id;
  active_width_ = start_metadata.image_size_x;
  active_height_ = start_metadata.image_size_y;
  active_image_bytes_ = static_cast<size_t>(active_width_) * active_height_ * sizeof(uint16_t);
  active_ = true;
  end_seen_ = false;
  series_images_ = 0;
  if (fast_extract_uint_field(data, bytes, "number_of_images", &announced_images_) != 0)
    announced_images_ = 0;
  metadata_.start_metadata.assign(data, data + bytes);
  ::dectris::log_header_appendix(start_metadata.series_id, data, bytes);
  // Held messages (images of this series that overtook its START, or of a later series) are
  // processed first, in arrival order. Messages of an earlier series whose START never came
  // (the pipeline started mid-series) can no longer be used: series ids increase per arm.
  if (!held_.empty()) {
    std::deque<zmq_msg_t> keep;
    size_t stale = 0;
    while (!held_.empty()) {
      uint64_t series = 0;
      const auto* held_data = static_cast<const uint8_t*>(zmq_msg_data(&held_.front()));
      if (fast_extract_series_id(held_data, zmq_msg_size(&held_.front()), &series) == 0 &&
          series < start_metadata.series_id) {
        zmq_msg_close(&held_.front());
        held_.pop_front();
        ++stale;
      } else {
        move_front(held_, keep);
      }
    }
    if (stale > 0)
      HOLOSCAN_LOG_WARN("ZmqRxOp: dropped {} messages of an earlier series whose START never "
                        "arrived (the pipeline started during that series)", stale);
    while (!replay_.empty()) move_front(replay_, keep);
    replay_.swap(keep);
    held_bytes_ = 0;
  }
  return emit_batch(true);
}

expected<void, Error> ZmqRxOp::stage_image(const uint8_t* data, size_t bytes) {
  const uint8_t* compressed = nullptr;
  size_t size = 0, original_size = 0, elem_size = 0;
  uint32_t image_width = 0, image_height = 0;
  uint64_t image_id = 0;
  if (fast_extract_compressed_bslz4(data, bytes, &compressed, &size, &original_size, &elem_size,
                                    &image_width, &image_height, &image_id) != 0 ||
      original_size != active_image_bytes_ || elem_size != sizeof(uint16_t) ||
      image_width != active_width_ || image_height != active_height_ ||
      size > kMaxCompressedImage || size < 12)
    return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid BSLZ4 image"});
  metadata_.offsets.push_back(staging_.size());
  metadata_.sizes.push_back(size);
  metadata_.image_ids.push_back(image_id);
  ++series_images_;
  staging_.insert(staging_.end(), compressed, compressed + size);
  return {};
}

expected<void, Error> ZmqRxOp::finish_series() {
  if (!metadata_.image_ids.empty()) {
    pending_end_ = true;          // the END follows in the next activation
    return emit_batch();
  }
  return emit_end();
}

expected<void, Error> ZmqRxOp::emit_end() {
  active_ = false;
  end_seen_ = false;
  has_finished_ = true;
  finished_series_ = metadata_.series_id;
  if (dropped_stragglers_ > 0)
    HOLOSCAN_LOG_WARN("ZmqRxOp: dropped {} late messages", dropped_stragglers_);
  dropped_stragglers_ = 0;
  // What was held (the next series) is processed before new messages.
  while (!held_.empty()) move_front(held_, replay_);
  held_bytes_ = 0;
  return emit_batch(false, true);
}

int ZmqRxOp::next_message() {
  // 1 = a message in msg_, 0 = none waiting, -1 = error. Held messages come first.
  if (!replay_.empty()) {
    zmq_msg_move(&msg_, &replay_.front());
    zmq_msg_close(&replay_.front());
    replay_.pop_front();
    return 1;
  }
  if (zmq_msg_recv(&msg_, zmq_socket_, ZMQ_DONTWAIT) >= 0) return 1;
  return (zmq_errno() == EAGAIN || zmq_errno() == EINTR) ? 0 : -1;
}

bool ZmqRxOp::hold_message() {
  held_bytes_ += zmq_msg_size(&msg_);
  held_.emplace_back();
  zmq_msg_init(&held_.back());
  zmq_msg_move(&held_.back(), &msg_);
  return held_.size() < kMaxHeldMessages && held_bytes_ < kMaxHeldBytes;
}

void ZmqRxOp::stop() { cleanup_zmq(); }
void ZmqRxOp::cleanup_zmq() {
  for (auto* queue : {&held_, &replay_}) {
    for (auto& m : *queue) zmq_msg_close(&m);
    queue->clear();
  }
  held_bytes_ = 0;
  if (msg_initialized_) { zmq_msg_close(&msg_); msg_initialized_ = false; }
  if (zmq_socket_) { zmq_close(zmq_socket_); zmq_socket_ = nullptr; }
  if (zmq_ctx_) { zmq_ctx_term(zmq_ctx_); zmq_ctx_ = nullptr; }
}
}  // namespace holoscan::ops
