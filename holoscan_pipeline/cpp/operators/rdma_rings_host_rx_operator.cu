#include "rdma_rings_host_rx_operator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <endian.h>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include "constants.h"
#include "fast_cbor_parse.h"
#include "header_appendix_log.hpp"
#include "imagemux_wire.hpp"

namespace holoscan::ops {
namespace {

using namespace imagemux;

constexpr uint16_t kProtocolVersion = 3;
// The immediate carries the ring index above the low 24 bits of the frame's ring sequence.
constexpr unsigned kSequenceBits = 24;
constexpr uint32_t kSequenceMask = (1U << kSequenceBits) - 1U;
constexpr size_t kMaxRings = 256;
constexpr uint64_t kCreditBytes = 8;
constexpr int kMaxReceiveDeficit = 256;
constexpr size_t kMaxBatchBytes =
    dectris::kMaxBatchSize * dectris::kMaxDetectorPixels * sizeof(uint16_t);

Error stream_error(std::string detail) {
  return make_error(ErrorCode::kProtocolError, "Invalid RDMA stream: " + std::move(detail));
}

}  // namespace

RdmaRingsHostRxOp::~RdmaRingsHostRxOp() { stop(); }

void RdmaRingsHostRxOp::setup(OperatorSpec& spec) {
  spec.output(output, "output").produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kPinnedHost,
                         .dtype = DLDataType{kDLUInt, 8, 1}, .rank = 1U},
      .bounds = tensor_bounds(kMaxBatchBytes),
      .storage = TensorOutputStorage::kRuntimePool});
}

Contract RdmaRingsHostRxOp::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::microseconds(100)});
  return result;
}

void RdmaRingsHostRxOp::start() {
  const auto& config = params();
  const bool counts_valid = config.ring_count >= 1 &&
                            static_cast<uint64_t>(config.ring_count) <= kMaxRings;
  const uint64_t ring_count = counts_valid ? static_cast<uint64_t>(config.ring_count) : 1;
  if (config.qp_handle == 0 || config.ring_address == 0 || config.credit_address == 0 ||
      !counts_valid || config.ring_bytes < 2 * kFrameAlign ||
      config.ring_bytes % kFrameAlign != 0 || config.ring_address % kFrameAlign != 0 ||
      config.ring_bytes > UINT64_MAX / ring_count ||
      config.ring_address > UINT64_MAX - config.ring_bytes * ring_count ||
      config.credit_address > UINT64_MAX - kCreditBytes * ring_count ||
      config.credit_rkey > std::numeric_limits<uint32_t>::max() ||
      config.publication_windows.empty() ||
      (!config.acquisition_windows.empty() &&
       config.acquisition_windows.size() != config.publication_windows.size()) ||
      config.batch_size < 1 || config.batch_size > 1000 || config.max_batch < 1 ||
      config.max_batch > 1024) {
    throw std::runtime_error("Invalid RDMA receiver configuration");
  }

  qp_ = reinterpret_cast<imx_qp*>(static_cast<uintptr_t>(config.qp_handle));
  ring_address_ = config.ring_address;
  ring_bytes_ = static_cast<size_t>(config.ring_bytes);
  credit_address_ = config.credit_address;
  credit_rkey_ = static_cast<uint32_t>(config.credit_rkey);
  batch_size_ = static_cast<size_t>(config.batch_size);
  max_batch_ = static_cast<size_t>(config.max_batch);
  acquisition_limit_ = config.acquisition_limit;
  publication_windows_.clear();
  acquisition_windows_.clear();
  for (const auto handle : config.publication_windows) {
    if (!handle) throw std::runtime_error("RdmaRingsHostRxOp: null publication window");
    publication_windows_.push_back(::dectris::access(handle));
  }
  for (const auto handle : config.acquisition_windows) {
    if (!handle) throw std::runtime_error("RdmaRingsHostRxOp: null acquisition window");
    acquisition_windows_.push_back(::dectris::access(handle));
  }
  lanes_ = publication_windows_.size();
  active_width_ = static_cast<uint32_t>(dectris::kDetectorWidth);
  active_height_ = static_cast<uint32_t>(dectris::kDetectorHeight);
  active_image_bytes_ = kImageBytes;

  rings_.assign(static_cast<size_t>(ring_count), RingState{});
  current_ring_ = 0;
  next_batch_id_ = 0;
  series_ordinal_ = 0;
  active_destination_ = 0;
  active_series_id_ = 0;
  active_series_images_ = 0;
  recv_deficit_ = 0;
  active_series_ = false;
  ready_emitted_ = false;
  pending_control_ = ControlKind::kNone;
  pending_control_series_id_ = 0;
  pending_control_images_ = 0;
  pending_start_metadata_.clear();
  immediate_scratch_.assign(max_batch_, 0);
  staging_.clear();
  staging_.reserve(kMaxBatchBytes);
  metadata_ = {};
  metadata_.image_ids.reserve(dectris::kMaxBatchSize);
  metadata_.sizes.reserve(dectris::kMaxBatchSize);
  metadata_.offsets.reserve(dectris::kMaxBatchSize);
  frame_.reset();
}

// The next series is admitted when the lane it is bound for has finished its previous one, and
// while finite mode has not yet used its acquisition. Until then its START stays in the ring and
// the sender is held back by the missing credit.
bool RdmaRingsHostRxOp::series_admitted() const {
  if (acquisition_limit_ && series_ordinal_ >= acquisition_limit_) return false;
  if (acquisition_windows_.empty()) return true;
  const size_t lane = static_cast<size_t>(series_ordinal_ % acquisition_windows_.size());
  return acquisition_windows_[lane]->admits(series_ordinal_ / acquisition_windows_.size() + 1);
}

expected<void, Error> RdmaRingsHostRxOp::compute(ExecutionContext& context) {
  if (!qp_) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA receiver configuration"));
  }
  if (context.cancellation().stop_requested()) return {};
  HOLOSCAN_TRY(retry_credit_writes());
  HOLOSCAN_TRY(poll_completions());
  // Every lane receives every batch, so each lane's window must admit the next one.
  for (const auto* window : publication_windows_) {
    if (!window->admits(next_batch_id_ + 1)) return {};
  }
  if (pending_control_ != ControlKind::kNone) return emit_control(pending_control_);

  metadata_.image_ids.clear();
  metadata_.sizes.clear();
  metadata_.offsets.clear();
  staging_.clear();
  size_t image_count = 0;
  uint64_t series_id = 0;

  while (image_count < batch_size_) {
    if (context.cancellation().stop_requested()) return {};
    HOLOSCAN_TRY(retry_credit_writes());

    if (frame_) {
      FrameState& frame = *frame_;
      const size_t frame_remaining = frame.header.image_count - frame.images_consumed;
      const size_t batch_remaining = batch_size_ - image_count;
      const size_t take = std::min(frame_remaining, batch_remaining);
      if (take == 0) return make_unexpected(stream_error("empty image frame remainder"));

      const size_t source_offset = frame.offset + kFrameHeaderBytes +
                                   static_cast<size_t>(frame.images_consumed) * active_image_bytes_;
      const size_t copy_bytes = take * active_image_bytes_;
      const uint8_t* source = ring_base(frame.ring) + source_offset;
      staging_.insert(staging_.end(), source, source + copy_bytes);
      for (size_t i = 0; i < take; ++i) {
        metadata_.image_ids.push_back(frame.header.image_id + frame.images_consumed + i);
        metadata_.offsets.push_back((image_count + i) * active_image_bytes_);
        metadata_.sizes.push_back(active_image_bytes_);
      }
      if (image_count == 0) series_id = frame.header.series_id;
      image_count += take;
      frame.images_consumed += static_cast<uint32_t>(take);

      // The bytes are in host staging now, so the ring space is free to reuse.
      if (frame.images_consumed == frame.header.image_count) {
        HOLOSCAN_TRY(record_credit(frame.ring, rings_[frame.ring].reader_position));
        frame_.reset();
      }
      continue;
    }

    if (pending_control_ != ControlKind::kNone) break;
    if (!active_series_ && !series_admitted()) break;
    RingState& ring = rings_[current_ring_];
    if (!ring.immediates.empty()) {
      const uint32_t sequence_bits = ring.immediates.front();
      ring.immediates.pop_front();
      HOLOSCAN_TRY(read_next_frame(current_ring_, sequence_bits));
      continue;
    }

    HOLOSCAN_TRY(poll_completions());
    if (pending_control_ != ControlKind::kNone) break;
    if (!rings_[current_ring_].immediates.empty()) continue;
    HOLOSCAN_TRY(check_series_continues());
    if (image_count == 0) return {};
    if (context.cancellation().stop_requested()) return {};
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  if (context.cancellation().stop_requested()) return {};
  if (image_count == 0) {
    if (pending_control_ != ControlKind::kNone) return emit_control(pending_control_);
    return {};
  }
  if (metadata_.image_ids.size() != image_count ||
      staging_.size() != image_count * active_image_bytes_) {
    return make_unexpected(stream_error("incomplete output batch"));
  }
  return emit_batch(series_id);
}

expected<void, Error> RdmaRingsHostRxOp::emit_batch(uint64_t series_id) {
  const std::array<int64_t, 1> shape{static_cast<int64_t>(std::max(size_t{1}, staging_.size()))};
  HOLOSCAN_TRY(auto loan, output.allocate_tensor(shape));
  HOLOSCAN_TRY(loan.write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
    if (!staging_.empty()) std::memcpy(writer.data(), staging_.data(), staging_.size());
  }));
  metadata_.batch_id = ++next_batch_id_;
  metadata_.series_id = series_id;
  metadata_.destination = active_destination_;
  metadata_.start_of_series = false;
  metadata_.end_of_series = false;
  metadata_.pipeline_ready = false;
  metadata_.series_images = 0;
  metadata_.start_metadata.clear();
  HOLOSCAN_TRY(output.emit_tensor(std::move(loan), metadata_));
  staging_.clear();
  metadata_.image_ids.clear();
  metadata_.sizes.clear();
  metadata_.offsets.clear();
  return {};
}

expected<void, Error> RdmaRingsHostRxOp::emit_control(ControlKind kind) {
  const std::array<int64_t, 1> shape{1};
  HOLOSCAN_TRY(auto loan, output.allocate_tensor(shape));
  HOLOSCAN_TRY(loan.write_host_and_commit([&](const TensorOutputWriteGuard& writer) {
    static_cast<uint8_t*>(writer.data())[0] = 0;
  }));
  metadata_.image_ids.clear();
  metadata_.sizes.clear();
  metadata_.offsets.clear();
  metadata_.batch_id = ++next_batch_id_;
  metadata_.series_id = 0;
  metadata_.destination = 0;
  metadata_.start_of_series = false;
  metadata_.end_of_series = false;
  metadata_.pipeline_ready = false;
  metadata_.series_images = 0;
  metadata_.start_metadata.clear();
  if (kind == ControlKind::kReady) {
    metadata_.pipeline_ready = true;
  } else if (kind == ControlKind::kStart) {
    metadata_.start_of_series = true;
    metadata_.series_id = pending_control_series_id_;
    metadata_.destination = active_destination_;
    metadata_.start_metadata = pending_start_metadata_;
  } else if (kind == ControlKind::kEnd) {
    metadata_.end_of_series = true;
    metadata_.series_id = pending_control_series_id_;
    metadata_.destination = active_destination_;
    metadata_.series_images = pending_control_images_;
  }
  HOLOSCAN_TRY(output.emit_tensor(std::move(loan), metadata_));
  if (kind == ControlKind::kReady) ready_emitted_ = true;
  if (kind == ControlKind::kStart) pending_start_metadata_.clear();
  pending_control_ = ControlKind::kNone;
  pending_control_series_id_ = 0;
  pending_control_images_ = 0;
  return {};
}

expected<void, Error> RdmaRingsHostRxOp::poll_completions() {
  if (recv_deficit_ > 0) HOLOSCAN_TRY(repost_recvs(0));

  imx_poll_result result{};
  const int rc = imx_qp_poll(qp_, immediate_scratch_.data(), static_cast<int>(max_batch_), &result);
  if (rc != 0) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      std::string("RDMA connection failed: ") + imx_last_error()));
  }
  if (result.n_failed > 0) {
    const char* details = imx_qp_errors(qp_);
    return make_unexpected(make_error(
        ErrorCode::kFailure,
        std::string("RDMA connection failed") +
            (details && details[0] ? std::string(": ") + details : std::string{})));
  }
  if (result.n_imm < 0 || static_cast<size_t>(result.n_imm) > max_batch_) {
    return make_unexpected(stream_error("receive completion count out of range"));
  }

  if (result.n_imm > 0) {
    HOLOSCAN_TRY(repost_recvs(result.n_imm));
    for (int i = 0; i < result.n_imm; ++i) {
      const uint32_t immediate = immediate_scratch_[i];
      const size_t ring = immediate >> kSequenceBits;
      if (ring >= rings_.size()) {
        return make_unexpected(stream_error("immediate names ring " + std::to_string(ring) +
                                            " of " + std::to_string(rings_.size())));
      }
      rings_[ring].immediates.push_back(immediate & kSequenceMask);
    }
  }
  if (!ready_emitted_ && pending_control_ == ControlKind::kNone)
    pending_control_ = ControlKind::kReady;
  return {};
}

// imagemux writes one series at a time and moves to the next ring when the series ID changes.
// Receive completions arrive in write order, so a frame in another ring while the current ring
// has none queued means the sender dropped the rest of the current ring's series.
expected<void, Error> RdmaRingsHostRxOp::check_series_continues() const {
  if (!rings_[current_ring_].immediates.empty()) return {};
  for (size_t ring = 0; ring < rings_.size(); ++ring) {
    if (ring == current_ring_ || rings_[ring].immediates.empty()) continue;
    std::string expected = active_series_
                               ? "END of series " + std::to_string(active_series_id_)
                               : "the next START";
    return make_unexpected(stream_error("ring " + std::to_string(ring) +
                                        " received a frame before " + expected + " in ring " +
                                        std::to_string(current_ring_) +
                                        "; the sender dropped frames"));
  }
  return {};
}

expected<void, Error> RdmaRingsHostRxOp::retry_credit_writes() {
  for (size_t index = 0; index < rings_.size(); ++index) {
    RingState& ring = rings_[index];
    if (ring.safe_credit_position <= ring.sent_credit_position) continue;
    const uint64_t value = htole64(ring.safe_credit_position);
    const int rc = imx_qp_write_inline(qp_, &value, sizeof(value),
                                       credit_address_ + kCreditBytes * index, credit_rkey_);
    if (rc < 0) {
      return make_unexpected(make_error(ErrorCode::kFailure,
                                        std::string("RDMA connection failed: ") +
                                            imx_last_error()));
    }
    if (rc == 0) break;  // Send queue full; retried on the next poll.
    ring.sent_credit_position = ring.safe_credit_position;
  }
  return {};
}

expected<void, Error> RdmaRingsHostRxOp::repost_recvs(int consumed) {
  const int want = consumed + recv_deficit_;
  if (want <= 0) return {};
  const int posted = imx_qp_post_recvs(qp_, want);
  if (posted < 0) {
    return make_unexpected(make_error(
        ErrorCode::kFailure,
        std::string("RDMA receive queue could not be replenished: ") + imx_last_error()));
  }
  recv_deficit_ = want - posted;
  if (recv_deficit_ > kMaxReceiveDeficit) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "RDMA receive queue could not be replenished"));
  }
  return {};
}

expected<void, Error> RdmaRingsHostRxOp::record_credit(size_t ring_index, uint64_t position) {
  RingState& ring = rings_[ring_index];
  if (position <= ring.safe_credit_position) {
    return make_unexpected(stream_error("ring credit position did not advance"));
  }
  ring.safe_credit_position = position;
  return {};
}

expected<void, Error> RdmaRingsHostRxOp::read_next_frame(size_t ring_index,
                                                         uint32_t sequence_bits) {
  RingState& ring = rings_[ring_index];
  if (sequence_bits != (ring.next_sequence & kSequenceMask)) {
    return make_unexpected(stream_error(
        "ring " + std::to_string(ring_index) + " announced sequence " +
        std::to_string(sequence_bits) + ", expected " +
        std::to_string(ring.next_sequence & kSequenceMask)));
  }
  const size_t offset = static_cast<size_t>(ring.reader_position % ring_bytes_);
  const uint8_t* frame = ring_base(ring_index) + offset;
  std::array<uint8_t, kFrameHeaderBytes> raw{};
  std::memcpy(raw.data(), frame, raw.size());
  const bool valid = load_le32(raw.data()) == kFrameMagic &&
                     load_le16(raw.data() + 4) == kProtocolVersion &&
                     load_le64(raw.data() + 8) == ring.next_sequence &&
                     load_le64(raw.data() + 16) == ring.reader_position &&
                     load_le16(raw.data() + 84) == ring_index;
  if (!valid) {
    return make_unexpected(stream_error(
        "frame header in ring " + std::to_string(ring_index) + " does not match sequence " +
        std::to_string(ring.next_sequence) + " at position " +
        std::to_string(ring.reader_position)));
  }

  const uint16_t kind = load_le16(raw.data() + 6);
  WireHeader header{};
  header.kind = kind;
  header.sequence = load_le64(raw.data() + 8);
  header.position = load_le64(raw.data() + 16);
  header.data_size = load_le64(raw.data() + 24);
  header.series_id = load_le64(raw.data() + 32);
  header.image_id = load_le64(raw.data() + 40);
  header.image_count = load_le32(raw.data() + 56);
  header.dim_x = load_le32(raw.data() + 60);
  header.dim_y = load_le32(raw.data() + 64);
  header.dtype = load_le16(raw.data() + 68);
  header.compression = load_le16(raw.data() + 70);
  header.uncompressed_size = load_le64(raw.data() + 72);
  header.channel = load_le16(raw.data() + 80);
  header.channel_count = load_le16(raw.data() + 82);
  header.ring = load_le16(raw.data() + 84);

  if (kind == kFrameWrap) {
    if (header.data_size != 0 || offset == 0 ||
        ring.reader_position > UINT64_MAX - (ring_bytes_ - offset)) {
      return make_unexpected(stream_error("invalid wrap frame"));
    }
    ring.reader_position += ring_bytes_ - offset;
    ++ring.next_sequence;
    return record_credit(ring_index, ring.reader_position);
  }
  if (kind != kFrameStart && kind != kFrameImage && kind != kFrameEnd) {
    return make_unexpected(stream_error("unknown frame kind " + std::to_string(kind)));
  }
  if (header.data_size > ring_bytes_ - offset - kFrameHeaderBytes ||
      header.data_size > UINT64_MAX - kFrameHeaderBytes - (kFrameAlign - 1)) {
    return make_unexpected(stream_error("frame runs past the end of the ring"));
  }
  const uint64_t span64 = (kFrameHeaderBytes + header.data_size + kFrameAlign - 1) &
                          ~static_cast<uint64_t>(kFrameAlign - 1);
  if (span64 > ring_bytes_ - offset || ring.reader_position > UINT64_MAX - span64) {
    return make_unexpected(stream_error("frame runs past the end of the ring"));
  }
  const uint64_t frame_end = ring.reader_position + span64;

  if (kind == kFrameStart) {
    if (active_series_) {
      return make_unexpected(stream_error("START of series " + std::to_string(header.series_id) +
                                          " before END of series " +
                                          std::to_string(active_series_id_)));
    }
    if (header.data_size > kMaxStartBytes) {
      return make_unexpected(make_error(ErrorCode::kProtocolError,
                                        "RDMA START metadata exceeds 16 KiB"));
    }
    pending_start_metadata_.assign(frame + kFrameHeaderBytes,
                                   frame + kFrameHeaderBytes + header.data_size);
    dectris_start_metadata_t start_metadata{};
    if (pending_start_metadata_.empty() ||
        fast_extract_start_metadata(pending_start_metadata_.data(), pending_start_metadata_.size(),
                                    &start_metadata) != 0 ||
        start_metadata.series_id != header.series_id) {
      return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid START image geometry"));
    }
    ::dectris::log_header_appendix(start_metadata.series_id, pending_start_metadata_.data(),
                                   pending_start_metadata_.size());
    active_width_ = start_metadata.image_size_x;
    active_height_ = start_metadata.image_size_y;
    active_image_bytes_ = static_cast<size_t>(active_width_) * active_height_ * sizeof(uint16_t);
    active_series_ = true;
    active_series_id_ = header.series_id;
    active_series_images_ = 0;
    // Series k is processed by lane k % lanes, whatever ring the sender wrote it into.
    active_destination_ = static_cast<uint32_t>(series_ordinal_ % lanes_);
    ++series_ordinal_;
    pending_control_ = ControlKind::kStart;
    pending_control_series_id_ = header.series_id;
    pending_control_images_ = 0;
    ring.reader_position = frame_end;
    ++ring.next_sequence;
    return record_credit(ring_index, ring.reader_position);
  }

  if (kind == kFrameEnd) {
    if (!active_series_ || header.series_id != active_series_id_) {
      return make_unexpected(stream_error("END of series " + std::to_string(header.series_id) +
                                          " without its START"));
    }
    pending_control_ = ControlKind::kEnd;
    pending_control_series_id_ = active_series_id_;
    pending_control_images_ = active_series_images_;
    active_series_ = false;
    ring.reader_position = frame_end;
    ++ring.next_sequence;
    HOLOSCAN_TRY(record_credit(ring_index, ring.reader_position));
    // The sender writes the next series into the following ring.
    current_ring_ = (ring_index + 1) % rings_.size();
    return {};
  }

  if (!active_series_ || header.series_id != active_series_id_ || header.image_count == 0 ||
      header.dtype != kDataTypeUint16 || header.compression != kCompressionNone ||
      header.channel != 0 || header.image_id > UINT64_MAX - header.image_count ||
      active_series_images_ > UINT64_MAX - header.image_count) {
    return make_unexpected(stream_error("unexpected IMAGE frame of series " +
                                        std::to_string(header.series_id)));
  }
  if (header.dim_x != active_width_ || header.dim_y != active_height_ ||
      header.image_count > UINT64_MAX / active_image_bytes_ ||
      header.data_size != static_cast<uint64_t>(header.image_count) * active_image_bytes_ ||
      header.uncompressed_size !=
          static_cast<uint64_t>(header.image_count) * active_image_bytes_) {
    return make_unexpected(make_error(
        ErrorCode::kProtocolError, "RDMA image geometry does not match START"));
  }
  active_series_images_ += header.image_count;
  ring.reader_position = frame_end;
  ++ring.next_sequence;
  frame_.emplace(FrameState{header, ring_index, offset, 0});
  return {};
}

void RdmaRingsHostRxOp::stop() {
  clear_state();
  qp_ = nullptr;
}

void RdmaRingsHostRxOp::clear_state() noexcept {
  rings_.clear();
  frame_.reset();
  pending_start_metadata_.clear();
  immediate_scratch_.clear();
  staging_.clear();
  active_series_ = false;
}

}  // namespace holoscan::ops
