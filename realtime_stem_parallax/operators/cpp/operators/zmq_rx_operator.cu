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
#include <deque>
#include <stdexcept>

#include <holoscan/holoscan.hpp>
#include <nvtx3/nvToolsExt.h>

#include "constants.h"
#include "zmq_rx_operator.hpp"
#include "fast_cbor_parse.h"


namespace holoscan::ops {

namespace {

// Receive timeout: how long compute() waits for a message before returning to the scheduler,
// and how long all connections must be quiet after `end` before the series counts as complete.
constexpr int kRecvTimeoutMs = 100;
// Messages of the next series held for the next run at most (beyond this, the run ends).
constexpr size_t kMaxHeldMessages = 65536;
constexpr size_t kMaxHeldBytes = size_t{512} << 20;
// Receive queue per connection, in messages: absorbs the stream while the next run starts.
constexpr int kRecvQueueMessages = 50000;

// The connection to the stream outlives one run of the operator. In continuous mode every scan
// builds a new application; closing the socket in between would throw away every message that
// already arrived for the next scan. So the socket stays open for the whole process, and
// messages of the next series that arrive before a run ends are held for the next run.
struct StreamConnection {
  void* ctx = nullptr;
  void* socket = nullptr;
  std::string endpoint;
  int64_t n_connections = 0;
  std::deque<zmq_msg_t> held;
  size_t held_bytes = 0;
  // The series the last run finished: its stragglers arriving later are leftovers.
  bool has_finished = false;
  uint64_t finished_series = 0;

  void close() {
    for (auto& m : held) zmq_msg_close(&m);
    held.clear();
    held_bytes = 0;
    has_finished = false;
    if (socket != nullptr) zmq_close(socket);
    if (ctx != nullptr) zmq_ctx_term(ctx);
    socket = ctx = nullptr;
  }
  ~StreamConnection() { close(); }
};

StreamConnection& stream_connection() {
  static StreamConnection connection;
  return connection;
}

// The shared socket for this endpoint, connected on first use.
void* open_stream(const std::string& endpoint, int64_t n_connections) {
  StreamConnection& c = stream_connection();
  if (c.socket != nullptr && c.endpoint == endpoint && c.n_connections == n_connections) {
    return c.socket;
  }
  c.close();
  c.ctx = zmq_ctx_new();
  if (c.ctx == nullptr) {
    throw std::runtime_error("ZmqRxOp: failed to create ZMQ context");
  }
  c.socket = zmq_socket(c.ctx, ZMQ_PULL);
  if (c.socket == nullptr) {
    throw std::runtime_error("ZmqRxOp: failed to create ZMQ PULL socket");
  }
  // Receive with a timeout so compute() can return while the stream is idle; otherwise a
  // stream that never sends `end` (aborted scan, idle start) blocks stop and Ctrl-C forever.
  const int recv_timeout_ms = kRecvTimeoutMs;
  zmq_setsockopt(c.socket, ZMQ_RCVTIMEO, &recv_timeout_ms, sizeof(recv_timeout_ms));
  const int recv_queue = kRecvQueueMessages;
  zmq_setsockopt(c.socket, ZMQ_RCVHWM, &recv_queue, sizeof(recv_queue));
  const int linger_ms = 0;
  zmq_setsockopt(c.socket, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));
  for (int64_t i = 0; i < n_connections; ++i) {
    // Each connect creates its own TCP connection (pipe); the sender's PUSH
    // round-robins across them, so aggregate rate scales past the sender's
    // per-connection message-rate ceiling.
    int rc = zmq_connect(c.socket, endpoint.c_str());
    if (rc != 0) {
      const std::string reason = zmq_strerror(zmq_errno());
      c.close();
      throw std::runtime_error("ZmqRxOp: failed to connect to " + endpoint + ": " + reason);
    }
  }
  c.endpoint = endpoint;
  c.n_connections = n_connections;
  return c.socket;
}

}  // namespace

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
             static_cast<int64_t>(dectris::kDefaultBatchSize));
  spec.param(detector_width_,
             "detector_width",
             "Detector width",
             "Frame width in pixels; validated against the stream.",
             static_cast<int64_t>(dectris::kDefaultDetectorWidth));
  spec.param(detector_height_,
             "detector_height",
             "Detector height",
             "Frame height in pixels; validated against the stream.",
             static_cast<int64_t>(dectris::kDefaultDetectorHeight));
  spec.param(pixel_dtype_,
             "dtype",
             "Pixel dtype",
             "Pixel data type, 'uint16' or 'uint32'; validated against the stream.",
             std::string(dectris::kDefaultPixelDtype));
  spec.param(n_connections_,
             "n_connections",
             "Stream connections",
             "Number of TCP connections to the sender. The DCU stream service paces "
             "each connection at roughly 10 us per message (~90-100k msg/s) regardless "
             "of frame size, so sustaining 120 kHz needs >= 2. ZMQ fair-queues the "
             "pipes back into this one socket; image_ids make cross-pipe ordering "
             "irrelevant downstream.",
             static_cast<int64_t>(1));
  spec.param(expected_images_,
             "expected_images",
             "Expected images",
             "Number of images the scan should deliver (Ny*Nx), or 0 if unknown. Compared with "
             "the `number_of_images` of the stream's start message, so a wrong scan size is "
             "reported before the scan runs.",
             static_cast<int64_t>(0));
  spec.output<CompressedBatch>("output");
}

void ZmqRxOp::start() {
  batch_count_ = 0;
  pending_bytes_ = 0;
  end_seen_ = false;
  next_seen_ = false;
  leftover_messages_ = 0;
  start_seen_ = false;
  series_known_ = false;
  series_id_ = 0;
  skipped_messages_ = 0;
  announced_images_ = 0;
  last_own_message_ = std::chrono::steady_clock::now();
  total_images_emitted_ = 0;
  total_batches_emitted_ = 0;
  current_pool_index_ = 0;
  dims_checked_ = false;
  const size_t batch_size = configured_batch_size();
  if (static_cast<size_t>(batch_size_.get()) > dectris::kMaxBatchSize) {
    HOLOSCAN_LOG_WARN("ZmqRxOp: requested batch_size {} exceeds the compiled ceiling "
                      "kMaxBatchSize={}; clamping to {}. Raise kMaxBatchSize in constants.h "
                      "and rebuild to allow larger batches.",
                      batch_size_.get(), dectris::kMaxBatchSize, batch_size);
  }

  if (detector_width_.get() <= 0 || detector_height_.get() <= 0) {
    throw std::runtime_error("ZmqRxOp: detector_width/detector_height must be positive");
  }
  elem_size_ = pixel_dtype_elem_size(pixel_dtype_.get());
  expected_image_bytes_ = static_cast<size_t>(detector_width_.get()) *
                          static_cast<size_t>(detector_height_.get()) * elem_size_;

  const int64_t n_conn = n_connections_.get();
  if (n_conn < 1) {
    throw std::runtime_error("ZmqRxOp: n_connections must be >= 1");
  }
  zmq_socket_ = open_stream(endpoint_.get(), n_conn);
  replay_left_ = stream_connection().held.size();
  if (replay_left_ > 0) {
    HOLOSCAN_LOG_INFO("ZmqRxOp: starting with {} messages of the next series that arrived "
                      "during the previous run", replay_left_);
  }
  zmq_msg_init(&msg_);
  msg_initialized_ = true;

  // Calculate worst-case compressed size per image
  // For BSLZ4: 12-byte Dectris header plus, per LZ4 block (nominally 8192 bytes), a
  // 4-byte length prefix and LZ4's incompressible-data bound.
  size_t n_blocks =
      (expected_image_bytes_ + dectris::kNominalBlockSize - 1) / dectris::kNominalBlockSize;
  max_compressed_size_per_image_ =
      12 + expected_image_bytes_ + n_blocks * (4 + dectris::kNominalBlockSize / 255 + 16);
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

  HOLOSCAN_LOG_INFO("ZmqRxOp: Start endpoint={} batch_size={} max_batch={} "
                    "detector={}x{} dtype={} connections={}",
                    endpoint_.get(), batch_size, dectris::kMaxBatchSize,
                    detector_height_.get(), detector_width_.get(), pixel_dtype_.get(),
                    n_connections_.get());
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
  size_t total_compressed_size = pending_bytes_;  // a partial batch may carry over
  bool saw_end = false;

  // Loop until we receive a full batch or the series is over.
  while (batch_count_ < batch_size) {
    {
      nvtx3::scoped_range range("ZmqRxOp::compute: receive msg", nvtx3::rgb{255, 0, 0}); // Red
      bool timed_out = false;
      if (!next_message(&timed_out)) {
        if (!timed_out) {
          continue;
        }
        if (end_seen_ || next_seen_) {
          // `end` (or the next series) arrived earlier and every connection has now been quiet
          // for the receive timeout: the frames still queued on other connections are all in.
          saw_end = true;
          break;
        }
        // Idle: return to the scheduler (so stop/Ctrl-C can take effect) and keep the
        // partial batch for the next call.
        pending_bytes_ = total_compressed_size;
        return;
      }
      msg_data = (const uint8_t*)zmq_msg_data(&msg_);
      msg_size = zmq_msg_size(&msg_);
    }

    {
      nvtx3::scoped_range range("ZmqRxOp::compute: process msg", nvtx3::rgb{0, 255, 0}); // Green

      // Use fast CBOR parser for message type detection
      zmq_message_type_t msg_type = fast_detect_message_type(msg_data, msg_size);

      // Every Stream2 message carries the series_id; the checks below keep one run to one
      // series. Messages without it (e.g. hand-made test streams) skip those checks.
      uint64_t msg_series = 0;
      const bool has_series =
          msg_type != ZMQ_MSG_UNKNOWN &&
          fast_extract_uint_field(msg_data, msg_size, "series_id", &msg_series) == 0;
      const bool other_series = has_series && series_known_ && msg_series != series_id_;
      const size_t received = total_images_emitted_ + batch_count_;
      const auto now = std::chrono::steady_clock::now();

      const StreamConnection& conn = stream_connection();
      if (has_series && conn.has_finished && msg_series == conn.finished_series &&
          !(series_known_ && msg_series == series_id_)) {
        // A straggler of the series the previous run finished.
        if (leftover_messages_++ == 0) {
          HOLOSCAN_LOG_WARN("{}: dropping messages of series {}, which the previous run already "
                            "finished", name(), msg_series);
        }
        continue;
      }

      // Messages of the next series can arrive (on one connection) while frames of this one are
      // still queued on the others; with several connections even its `start` can overtake this
      // series' `end`. Hold them for the next run and keep draining until this series has been
      // quiet for the receive timeout. (A `start` before any frame just replaces the series.)
      const bool adopt_start = msg_type == ZMQ_MSG_START && !end_seen_ && received == 0;
      if (!adopt_start && (other_series || (end_seen_ && msg_type == ZMQ_MSG_START))) {
        next_seen_ = true;
        const bool room_left = hold_message();
        if (now - last_own_message_ > std::chrono::milliseconds(kRecvTimeoutMs) || !room_left) {
          saw_end = true;
          break;
        }
        continue;
      }

      if (msg_type == ZMQ_MSG_START) {
        if (has_series) {
          series_id_ = msg_series;
          series_known_ = true;
        }
        start_seen_ = true;
        last_own_message_ = now;
        handle_start(msg_data, msg_size);
        continue; // Continue to next message
      }

      if (msg_type == ZMQ_MSG_END) {
        if (!start_seen_ && total_images_emitted_ + batch_count_ == 0) {
          // Nothing of this series was seen: an `end` left over from an earlier run.
          HOLOSCAN_LOG_WARN("{}: ignoring an `end` message (series {}) that came before any "
                            "start or frame, left over from an earlier run", name(),
                            has_series ? std::to_string(msg_series) : std::string("?"));
          continue;
        }
        // With several connections the sender round-robins, so `end` can overtake frames still
        // queued on the other connections. Keep reading until they are quiet (see EAGAIN),
        // unless every announced image is already in.
        if (!end_seen_) {
          end_seen_ = true;
          last_own_message_ = now;
          HOLOSCAN_LOG_INFO("{}: end received; draining the remaining connections", name());
        }
        if (series_complete()) {
          saw_end = true;
          break;
        }
        continue;
      }

      if (msg_type == ZMQ_MSG_IMAGE) {
        if (!series_known_ && has_series) {
          series_id_ = msg_series;
          series_known_ = true;
        }

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
          if (skipped_messages_ == 0) {
            HOLOSCAN_LOG_ERROR("{}: could not read a bslz4-compressed frame from an image "
                               "message; skipping it. If every frame fails, set the DCU's "
                               "stream compression to bslz4.", name());
          }
          skipped_messages_++;
          continue;
        }

        // Validate the stream against the configured geometry BEFORE the data enters the
        // pipeline — a mismatch would otherwise decompress into wrong-sized buffers and
        // emit silently wrong frames.
        if (original_size != expected_image_bytes_ || elem_size != elem_size_) {
          HOLOSCAN_LOG_ERROR(
              "{}: stream image is {} bytes with {}-byte elements, but the pipeline is "
              "configured for {}x{} {} ({} bytes, {}-byte elements). Set detector_width/"
              "detector_height/dtype to match the acquisition.",
              name(), original_size, elem_size, detector_height_.get(),
              detector_width_.get(), pixel_dtype_.get(), expected_image_bytes_, elem_size_);
          throw std::runtime_error("ZmqRxOp: stream does not match configured detector "
                                   "geometry/dtype");
        }
        if (!dims_checked_) {
          // Dims are validated once per stream: original_size cannot distinguish
          // transposed dimensions of a non-square detector.
          uint64_t dim0 = 0, dim1 = 0;
          if (fast_extract_image_dims(msg_data, msg_size, &dim0, &dim1) == 0) {
            if (dim0 != static_cast<uint64_t>(detector_height_.get()) ||
                dim1 != static_cast<uint64_t>(detector_width_.get())) {
              HOLOSCAN_LOG_ERROR(
                  "{}: stream reports {}x{} images but the pipeline is configured for "
                  "{}x{} (height x width).",
                  name(), dim0, dim1, detector_height_.get(), detector_width_.get());
              throw std::runtime_error("ZmqRxOp: stream does not match configured "
                                       "detector geometry");
            }
            HOLOSCAN_LOG_INFO("{}: stream validated: {}x{} {}", name(), dim0, dim1,
                              pixel_dtype_.get());
          } else {
            HOLOSCAN_LOG_WARN("{}: could not read image dims from the stream; relying on "
                              "size/elem-size checks only", name());
          }
          dims_checked_ = true;
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
        last_own_message_ = now;
        if (series_complete()) {
          saw_end = true;
          break;
        }
      } else {
        HOLOSCAN_LOG_WARN("{}: Unknown message type received (size: {} bytes)", name(), msg_size);
      }
    }
  }

  if (batch_count_ > 0) {
    emit_current_batch(op_output, pool_buffer_offset, batch_count_, total_compressed_size);
  }
  pending_bytes_ = 0;

  if (saw_end) {
    HOLOSCAN_LOG_INFO("{}: Stream ended after emitting {} images in {} batches",
                      name(), total_images_emitted_, total_batches_emitted_);
    if (!start_seen_ && total_images_emitted_ > 0) {
      HOLOSCAN_LOG_WARN("{}: the start message of series {} never arrived. If the pipeline "
                        "joined this series part-way (frames left over from an earlier run, or "
                        "started during a scan), the result is incomplete; the frame count "
                        "tells.", name(), series_id_);
    }
    if (!end_seen_) {
      HOLOSCAN_LOG_WARN("{}: series {} never sent its `end`: the next series began. Ending this "
                        "run with the frames received so far.", name(), series_id_);
    }
    if (series_known_) {
      StreamConnection& c = stream_connection();
      c.has_finished = true;
      c.finished_series = series_id_;
    }
    if (skipped_messages_ > 0) {
      HOLOSCAN_LOG_WARN("{}: {} image messages could not be decoded and were skipped",
                        name(), skipped_messages_);
    }
    // Stop ONLY this operator. Application::stop_execution() would retire every operator at
    // once, before the batch emitted just above is processed (the final partial batch was
    // lost that way). The decompressor and the sink finish everything already emitted, and
    // the scheduler then ends the application because nothing is left to run.
    stop_execution();
  }
}

void ZmqRxOp::handle_start(const uint8_t* msg_data, size_t msg_size) {
  // Report the series and compare what the DCU announces with the configuration, so a wrong
  // scan size or detector mode shows up when the detector is armed, not after the scan.
  uint64_t n_images = 0, size_x = 0, size_y = 0;
  const bool has_n =
      fast_extract_uint_field(msg_data, msg_size, "number_of_images", &n_images) == 0;
  announced_images_ = has_n ? n_images : 0;
  const bool has_xy =
      fast_extract_uint_field(msg_data, msg_size, "image_size_x", &size_x) == 0 &&
      fast_extract_uint_field(msg_data, msg_size, "image_size_y", &size_y) == 0;
  HOLOSCAN_LOG_INFO("{}: start of series {}: {} images of {}x{}", name(),
                    series_known_ ? std::to_string(series_id_) : std::string("?"),
                    has_n ? std::to_string(n_images) : std::string("?"),
                    has_xy ? std::to_string(size_y) : std::string("?"),
                    has_xy ? std::to_string(size_x) : std::string("?"));
  const int64_t expected = expected_images_.get();
  if (has_n && expected > 0 && n_images != static_cast<uint64_t>(expected)) {
    HOLOSCAN_LOG_WARN("{}: the detector will send {} images, but the scan has {} positions. "
                      "Check the scan size, or whether the scan adds extra (e.g. flyback) frames.",
                      name(), n_images, expected);
  }
  if (has_xy && (size_x != static_cast<uint64_t>(detector_width_.get()) ||
                 size_y != static_cast<uint64_t>(detector_height_.get()))) {
    HOLOSCAN_LOG_WARN("{}: the detector will send {}x{} frames, but the pipeline is configured "
                      "for {}x{} (height x width); the first frame will be refused.",
                      name(), size_y, size_x, detector_height_.get(), detector_width_.get());
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
  zmq_socket_ = nullptr;  // the connection itself stays open for the next run
}

bool ZmqRxOp::next_message(bool* timed_out) {
  // Held messages first (they arrived first), then the socket.
  *timed_out = false;
  StreamConnection& c = stream_connection();
  if (replay_left_ > 0 && !c.held.empty()) {
    c.held_bytes -= zmq_msg_size(&c.held.front());
    zmq_msg_move(&msg_, &c.held.front());
    zmq_msg_close(&c.held.front());
    c.held.pop_front();
    replay_left_--;
    return true;
  }
  if (zmq_msg_recv(&msg_, zmq_socket_, 0) >= 0) {
    return true;
  }
  if (zmq_errno() == EAGAIN) {
    *timed_out = true;
  } else {
    HOLOSCAN_LOG_WARN("{}: ZMQ receive failed: {}", name(), zmq_strerror(zmq_errno()));
  }
  return false;
}

bool ZmqRxOp::hold_message() {
  // Keep the current message (of the next series) for the next run. False once the held
  // messages reach their limit.
  StreamConnection& c = stream_connection();
  c.held_bytes += zmq_msg_size(&msg_);
  c.held.emplace_back();
  zmq_msg_init(&c.held.back());
  zmq_msg_move(&c.held.back(), &msg_);
  return c.held.size() < kMaxHeldMessages && c.held_bytes < kMaxHeldBytes;
}

bool ZmqRxOp::series_complete() const {
  // After `end`: every image the start message announced is in, so nothing is left to drain.
  return end_seen_ && announced_images_ > 0 &&
         total_images_emitted_ + batch_count_ >= announced_images_;
}

} // namespace holoscan::ops
