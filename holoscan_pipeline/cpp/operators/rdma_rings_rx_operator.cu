#include "rdma_rings_rx_operator.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

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
constexpr auto kGpuHeaderSettle = std::chrono::milliseconds(10);
constexpr auto kGpuHeaderRetry = std::chrono::microseconds(20);

Error stream_error(std::string detail) {
  return make_error(ErrorCode::kProtocolError, "Invalid RDMA stream: " + std::move(detail));
}

}  // namespace

RdmaRingsRxOp::CudaEvent::~CudaEvent() {
  if (handle) cudaEventDestroy(handle);
}

RdmaRingsRxOp::~RdmaRingsRxOp() { stop(); }

void RdmaRingsRxOp::setup(OperatorSpec& spec) {
  spec.output(output, "output").produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kCudaDevice,
                         .dtype = DLDataType{kDLUInt, 16, 1}, .rank = 3U},
      .bounds = tensor_bounds(dectris::kMaxBatchSize * dectris::kMaxDetectorPixels * sizeof(uint16_t)),
      .storage = TensorOutputStorage::kRuntimePool});
}

Contract RdmaRingsRxOp::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::microseconds(100)});
  return result;
}

void RdmaRingsRxOp::start() {
  const auto& config = params();
  const bool counts_valid = config.ring_count >= 1 &&
                            static_cast<uint64_t>(config.ring_count) <= kMaxRings;
  const uint64_t ring_count = counts_valid ? static_cast<uint64_t>(config.ring_count) : 1;
  if (config.qp_handle == 0 || config.ring_address == 0 || config.ring_device_address == 0 ||
      config.credit_address == 0 || !counts_valid || config.ring_bytes < 2 * kFrameAlign ||
      config.ring_bytes % kFrameAlign != 0 || config.ring_address % kFrameAlign != 0 ||
      config.ring_bytes > UINT64_MAX / ring_count ||
      config.ring_address > UINT64_MAX - config.ring_bytes * ring_count ||
      config.ring_device_address > UINT64_MAX - config.ring_bytes * ring_count ||
      config.credit_address > UINT64_MAX - kCreditBytes * ring_count ||
      config.credit_rkey > std::numeric_limits<uint32_t>::max() ||
      (config.ring_memory != "gpu" && config.ring_memory != "host_mapped") ||
      config.batch_size < 1 || config.batch_size > 1000 || config.max_batch < 1 ||
      config.max_batch > 1024 || config.device < 0 || config.device > INT_MAX) {
    throw std::runtime_error("Invalid RDMA receiver configuration");
  }

  qp_ = reinterpret_cast<imx_qp*>(static_cast<uintptr_t>(config.qp_handle));
  publication_window_ = ::dectris::access(config.publication_window);
  ring_address_ = config.ring_address;
  ring_device_address_ = config.ring_device_address;
  ring_bytes_ = static_cast<size_t>(config.ring_bytes);
  gpu_ring_ = config.ring_memory == "gpu";
  credit_address_ = config.credit_address;
  credit_rkey_ = static_cast<uint32_t>(config.credit_rkey);
  device_ = static_cast<int>(config.device);
  batch_size_ = static_cast<size_t>(config.batch_size);
  max_batch_ = static_cast<size_t>(config.max_batch);
  active_width_ = static_cast<uint32_t>(dectris::kDetectorWidth);
  active_height_ = static_cast<uint32_t>(dectris::kDetectorHeight);
  active_image_bytes_ = kImageBytes;

  const cudaError_t set_device_status = cudaSetDevice(device_);
  if (set_device_status != cudaSuccess) {
    throw std::runtime_error("Invalid RDMA receiver configuration: "
                             "cudaSetDevice: " + cuda_detail(set_device_status));
  }
  const uint64_t last_byte = ring_address_ + ring_bytes_ * ring_count - 1;
  if (gpu_ring_) {
    for (const uint64_t address : {ring_address_, last_byte}) {
      cudaPointerAttributes attributes{};
      const cudaError_t status = cudaPointerGetAttributes(
          &attributes, reinterpret_cast<const void*>(static_cast<uintptr_t>(address)));
      if (status != cudaSuccess || attributes.type != cudaMemoryTypeDevice ||
          attributes.device != device_ || ring_device_address_ != ring_address_) {
        throw std::runtime_error("Invalid RDMA receiver configuration: GPU ring address or "
                                 "device does not match");
      }
    }
    const unsigned int enabled = 1;
    const CUresult driver_status = cuPointerSetAttribute(
        &enabled, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, static_cast<CUdeviceptr>(ring_address_));
    if (driver_status != CUDA_SUCCESS) {
      const char* message = nullptr;
      cuGetErrorString(driver_status, &message);
      throw std::runtime_error(std::string("Invalid RDMA receiver configuration: "
                                           "cuPointerSetAttribute(SYNC_MEMOPS): ") +
                               (message ? message : "CUDA driver error"));
    }
  } else {
    // The rings share one mapped allocation, so its device alias is contiguous as well.
    for (const uint64_t address : {ring_address_, last_byte}) {
      void* alias = nullptr;
      const cudaError_t status = cudaHostGetDevicePointer(
          &alias, reinterpret_cast<void*>(static_cast<uintptr_t>(address)), 0);
      if (status != cudaSuccess ||
          reinterpret_cast<uintptr_t>(alias) != ring_device_address_ + (address - ring_address_)) {
        throw std::runtime_error("Invalid RDMA receiver configuration: mapped host ring alias "
                                 "does not match");
      }
    }
  }

  rings_.assign(static_cast<size_t>(ring_count), RingState{});
  current_ring_ = 0;
  pending_credit_count_ = 0;
  next_batch_id_ = 0;
  last_capture_ns_ = -1;
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
  frame_.reset();
}

expected<void, Error> RdmaRingsRxOp::compute(ExecutionContext& context) {
  if (!qp_) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA receiver configuration"));
  }
  cudaError_t cuda_status = cudaSetDevice(device_);
  if (cuda_status != cudaSuccess) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA stream: cudaSetDevice: " +
                                          cuda_detail(cuda_status)));
  }

  if (context.cancellation().stop_requested()) return {};
  HOLOSCAN_TRY(poll_credits());
  HOLOSCAN_TRY(retry_credit_writes());
  if (publication_window_ && !publication_window_->admits(next_batch_id_ + 1)) return {};
  if (pending_control_ != ControlKind::kNone) return emit_control(context, pending_control_);

  auto producer_stream = context.cuda_stream();
  const cudaStream_t stream = producer_stream.get();
  std::optional<TensorOutputLoan> loan;
  std::optional<TensorOutputWriteGuard> writer;
  bool work_enqueued = false;
  bool published = false;
  StreamSyncOnAbandon sync_on_abandon{stream, work_enqueued, published};

  DecodedBatch metadata{};
  metadata.image_ids.reserve(batch_size_);
  size_t image_count = 0;
  uint64_t series_id = 0;
  bool frame_event_waited = false;

  auto open_writer = [&]() -> expected<void, Error> {
    if (writer) return {};
    const std::array<int64_t, 3> shape{static_cast<int64_t>(batch_size_),
                                       static_cast<int64_t>(active_height_),
                                       static_cast<int64_t>(active_width_)};
    auto allocated = output.allocate_tensor(shape);
    if (!allocated) {
      return make_unexpected(make_error(
          ErrorCode::kFailure, "RDMA data tensor allocation failed: " +
                                   sdk_detail(allocated.error())));
    }
    loan.emplace(std::move(*allocated));
    auto result = loan->write(producer_stream);
    if (!result) {
      return make_unexpected(make_error(
          ErrorCode::kFailure, "RDMA data tensor write failed: " + sdk_detail(result.error()) +
                                   " (" + loan_detail(*loan) + ")"));
    }
    writer.emplace(std::move(*result));
    return {};
  };

  auto record_read_event = [&]() -> expected<std::shared_ptr<CudaEvent>, Error> {
    auto event = std::make_shared<CudaEvent>();
    cudaError_t status = cudaEventCreateWithFlags(&event->handle, cudaEventDisableTiming);
    if (status == cudaSuccess) status = cudaEventRecord(event->handle, stream);
    if (status != cudaSuccess) {
      return make_unexpected(make_error(ErrorCode::kFailure,
                                        "Invalid RDMA stream: CUDA read event: " +
                                            cuda_detail(status)));
    }
    return event;
  };

  while (image_count < batch_size_) {
    if (context.cancellation().stop_requested()) return {};
    HOLOSCAN_TRY(poll_credits());
    HOLOSCAN_TRY(retry_credit_writes());

    if (frame_) {
      HOLOSCAN_TRY(open_writer());
      if (frame_->last_read && !frame_event_waited) {
        cuda_status = cudaStreamWaitEvent(stream, frame_->last_read->handle, 0);
        if (cuda_status != cudaSuccess) {
          return make_unexpected(make_error(ErrorCode::kFailure,
                                            "Invalid RDMA stream: cudaStreamWaitEvent: " +
                                                cuda_detail(cuda_status)));
        }
        work_enqueued = true;
        frame_event_waited = true;
      }

      FrameState& frame = *frame_;
      const size_t frame_remaining = frame.header.image_count - frame.images_consumed;
      const size_t batch_remaining = batch_size_ - image_count;
      const size_t take = std::min(frame_remaining, batch_remaining);
      if (take == 0) return make_unexpected(stream_error("empty image frame remainder"));

      const size_t source_offset = frame.offset + kFrameHeaderBytes +
                                   static_cast<size_t>(frame.images_consumed) * active_image_bytes_;
      const size_t destination_offset = image_count * active_image_bytes_;
      const size_t copy_bytes = take * active_image_bytes_;
      const auto* source = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(
          ring_device_address(frame.ring) + source_offset));
      auto* destination = static_cast<uint8_t*>(writer->data()) + destination_offset;
      if (gpu_ring_) {
        cuda_status = cudaMemcpyAsync(destination, source, copy_bytes, cudaMemcpyDeviceToDevice,
                                      stream);
      } else {
        const unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>((copy_bytes + threads - 1) / threads);
        gather_mapped_bytes<<<blocks, threads, 0, stream>>>(source, destination, copy_bytes);
        cuda_status = cudaGetLastError();
      }
      if (cuda_status != cudaSuccess) {
        return make_unexpected(make_error(ErrorCode::kFailure,
                                          "Invalid RDMA stream: image gather: " +
                                              cuda_detail(cuda_status)));
      }
      work_enqueued = true;
      for (size_t i = 0; i < take; ++i) {
        metadata.image_ids.push_back(
            frame.header.image_id + frame.images_consumed + static_cast<uint64_t>(i));
      }
      if (image_count == 0) series_id = frame.header.series_id;
      image_count += take;
      frame.images_consumed += static_cast<uint32_t>(take);

      if (frame.images_consumed == frame.header.image_count) {
        auto event = record_read_event();
        if (!event) return make_unexpected(std::move(event).error());
        HOLOSCAN_TRY(record_credit(frame.ring, rings_[frame.ring].reader_position, *event));
        frame_.reset();
        frame_event_waited = false;
      }

      if (image_count == batch_size_) {
        if (frame_) {
          auto event = record_read_event();
          if (!event) return make_unexpected(std::move(event).error());
          frame_->last_read = std::move(*event);
        }
        break;
      }
      continue;
    }

    if (pending_control_ != ControlKind::kNone) break;
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
    if (pending_control_ != ControlKind::kNone) return emit_control(context, pending_control_);
    return {};
  }
  if (!loan || !writer || metadata.image_ids.size() != image_count) {
    return make_unexpected(stream_error("incomplete output batch"));
  }

  metadata.batch_id = ++next_batch_id_;
  metadata.series_id = series_id;
  metadata.end_of_series = false;
  metadata.start_of_series = false;
  metadata.pipeline_ready = false;
  metadata.series_images = 0;
  auto committed = std::move(*writer).commit();
  writer.reset();
  if (!committed) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA stream: tensor write commit failed: " +
                                          sdk_detail(committed.error())));
  }
  auto capture = context.activation_time();
  capture.timestamp_ns = std::max(capture.timestamp_ns, last_capture_ns_ + 1);
  last_capture_ns_ = capture.timestamp_ns;
  auto emitted = output.emit_tensor(std::move(*loan), metadata,
                                    EmitOptions{.capture_time = capture});
  if (!emitted) return make_unexpected(std::move(emitted).error());
  loan.reset();
  published = true;
  return {};
}

expected<void, Error> RdmaRingsRxOp::poll_completions() {
  if (recv_deficit_ > 0) HOLOSCAN_TRY(repost_recvs(0));

  const size_t room = max_batch_ > pending_credit_count_ ? max_batch_ - pending_credit_count_ : 0;
  if (room == 0) return {};
  imx_poll_result result{};
  const int rc = imx_qp_poll(qp_, immediate_scratch_.data(), static_cast<int>(room), &result);
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
  if (result.n_imm < 0 || static_cast<size_t>(result.n_imm) > room) {
    return make_unexpected(stream_error("receive completion count out of range"));
  }

  if (result.n_imm > 0) {
    HOLOSCAN_TRY(repost_recvs(result.n_imm));
    if (gpu_ring_) HOLOSCAN_TRY(flush_gpudirect_writes());
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
expected<void, Error> RdmaRingsRxOp::check_series_continues() const {
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

expected<void, Error> RdmaRingsRxOp::poll_credits() {
  for (RingState& ring : rings_) {
    while (!ring.credits.empty()) {
      const auto& credit = ring.credits.front();
      if (credit.event) {
        const cudaError_t status = cudaEventQuery(credit.event->handle);
        if (status == cudaErrorNotReady) break;
        if (status != cudaSuccess) {
          return make_unexpected(make_error(ErrorCode::kFailure,
                                            "Invalid RDMA stream: CUDA credit event: " +
                                                cuda_detail(status)));
        }
      }
      ring.safe_credit_position = credit.position;
      ring.credits.pop_front();
      --pending_credit_count_;
    }
  }
  return {};
}

expected<void, Error> RdmaRingsRxOp::retry_credit_writes() {
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

expected<void, Error> RdmaRingsRxOp::repost_recvs(int consumed) {
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

expected<void, Error> RdmaRingsRxOp::read_next_frame(size_t ring_index, uint32_t sequence_bits) {
  RingState& ring = rings_[ring_index];
  if (sequence_bits != (ring.next_sequence & kSequenceMask)) {
    return make_unexpected(stream_error(
        "ring " + std::to_string(ring_index) + " announced sequence " +
        std::to_string(sequence_bits) + ", expected " +
        std::to_string(ring.next_sequence & kSequenceMask)));
  }
  const size_t offset = static_cast<size_t>(ring.reader_position % ring_bytes_);
  const uint64_t frame_address = ring_host_address(ring_index) + offset;
  std::array<uint8_t, kFrameHeaderBytes> raw{};
  const auto deadline = std::chrono::steady_clock::now() + kGpuHeaderSettle;
  for (;;) {
    if (gpu_ring_) {
      const cudaError_t status = cudaMemcpy(
          raw.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(frame_address)),
          raw.size(), cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) {
        return make_unexpected(make_error(ErrorCode::kFailure,
                                          "Invalid RDMA stream: GPU header read: " +
                                              cuda_detail(status)));
      }
    } else {
      std::memcpy(raw.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(frame_address)),
                  raw.size());
    }
    const bool valid = load_le32(raw.data()) == kFrameMagic &&
                       load_le16(raw.data() + 4) == kProtocolVersion &&
                       load_le64(raw.data() + 8) == ring.next_sequence &&
                       load_le64(raw.data() + 16) == ring.reader_position &&
                       load_le16(raw.data() + 84) == ring_index;
    if (valid) break;
    if (!gpu_ring_ || std::chrono::steady_clock::now() >= deadline) {
      return make_unexpected(stream_error(
          "frame header in ring " + std::to_string(ring_index) + " does not match sequence " +
          std::to_string(ring.next_sequence) + " at position " +
          std::to_string(ring.reader_position)));
    }
    std::this_thread::sleep_for(kGpuHeaderRetry);
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
    pending_start_metadata_.resize(static_cast<size_t>(header.data_size));
    if (!pending_start_metadata_.empty()) {
      const auto* source = reinterpret_cast<const void*>(
          static_cast<uintptr_t>(frame_address + kFrameHeaderBytes));
      if (gpu_ring_) {
        const cudaError_t status = cudaMemcpy(pending_start_metadata_.data(), source,
                                              pending_start_metadata_.size(),
                                              cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
          return make_unexpected(make_error(ErrorCode::kFailure,
                                            "Invalid RDMA stream: START read: " +
                                                cuda_detail(status)));
        }
      } else {
        std::memcpy(pending_start_metadata_.data(), source, pending_start_metadata_.size());
      }
    }
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
  frame_.emplace(FrameState{header, ring_index, offset, 0, {}});
  return {};
}

expected<void, Error> RdmaRingsRxOp::record_credit(size_t ring_index, uint64_t position,
                                                  const std::shared_ptr<CudaEvent>& event) {
  RingState& ring = rings_[ring_index];
  if (!ring.credits.empty() && ring.credits.back().position >= position) {
    return make_unexpected(stream_error("ring credit position did not advance"));
  }
  ring.credits.push_back(CreditState{position, event});
  ++pending_credit_count_;
  return {};
}

expected<void, Error> RdmaRingsRxOp::flush_gpudirect_writes() {
  const cudaError_t status = cudaDeviceFlushGPUDirectRDMAWrites(
      cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
      cudaFlushGPUDirectRDMAWritesToOwner);
  if (status != cudaSuccess) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA stream: GPUDirect flush: " +
                                          cuda_detail(status)));
  }
  return {};
}

expected<void, Error> RdmaRingsRxOp::emit_control(ExecutionContext& context, ControlKind kind) {
  const uint32_t width = kind == ControlKind::kReady
                             ? static_cast<uint32_t>(dectris::kDetectorWidth) : active_width_;
  const uint32_t height = kind == ControlKind::kReady
                              ? static_cast<uint32_t>(dectris::kDetectorHeight) : active_height_;
  const size_t image_bytes = static_cast<size_t>(width) * height * sizeof(uint16_t);
  const std::array<int64_t, 3> shape{1, static_cast<int64_t>(height), static_cast<int64_t>(width)};
  auto allocated = output.allocate_tensor(shape);
  if (!allocated) {
    return make_unexpected(make_error(
        ErrorCode::kFailure, "RDMA control tensor allocation failed: " +
                                 sdk_detail(allocated.error())));
  }
  auto loan = std::move(*allocated);
  auto producer_stream = context.cuda_stream();
  auto writer_result = loan.write(producer_stream);
  if (!writer_result) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "RDMA control tensor write failed: " +
                                          sdk_detail(writer_result.error()) + " (" +
                                          loan_detail(loan) + ")"));
  }
  auto writer = std::move(*writer_result);
  bool work_enqueued = false;
  bool published = false;
  StreamSyncOnAbandon sync_on_abandon{producer_stream.get(), work_enqueued, published};
  auto* destination = static_cast<uint8_t*>(writer.data());
  cudaError_t status = cudaSuccess;
  if (kind == ControlKind::kReady) {
    // READY warm-up touches the ring through the same path as image gathers.
    const auto* source = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ring_device_address_));
    if (gpu_ring_) {
      status = cudaMemcpyAsync(destination, source, 1, cudaMemcpyDeviceToDevice,
                               producer_stream.get());
    } else {
      gather_mapped_bytes<<<1, 1, 0, producer_stream.get()>>>(source, destination, 1);
      status = cudaGetLastError();
    }
    if (status == cudaSuccess) work_enqueued = true;
  }
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(destination, 0, image_bytes, producer_stream.get());
    if (status == cudaSuccess) work_enqueued = true;
  }
  if (status != cudaSuccess) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA stream: control tensor write: " +
                                          cuda_detail(status)));
  }
  auto committed = std::move(writer).commit();
  if (!committed) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "Invalid RDMA stream: tensor write commit failed: " +
                                          sdk_detail(committed.error())));
  }

  DecodedBatch metadata{};
  metadata.batch_id = ++next_batch_id_;
  if (kind == ControlKind::kReady) {
    metadata.pipeline_ready = true;
  } else if (kind == ControlKind::kStart) {
    metadata.start_of_series = true;
    metadata.series_id = pending_control_series_id_;
    metadata.start_metadata = pending_start_metadata_;
  } else if (kind == ControlKind::kEnd) {
    metadata.end_of_series = true;
    metadata.series_id = pending_control_series_id_;
    metadata.series_images = pending_control_images_;
  }

  auto capture = context.activation_time();
  capture.timestamp_ns = std::max(capture.timestamp_ns, last_capture_ns_ + 1);
  last_capture_ns_ = capture.timestamp_ns;
  auto emitted = output.emit_tensor(std::move(loan), metadata,
                                    EmitOptions{.capture_time = capture});
  if (!emitted) return make_unexpected(std::move(emitted).error());
  published = true;
  if (kind == ControlKind::kReady) ready_emitted_ = true;
  if (kind == ControlKind::kStart) pending_start_metadata_.clear();
  pending_control_ = ControlKind::kNone;
  pending_control_series_id_ = 0;
  pending_control_images_ = 0;
  return {};
}

void RdmaRingsRxOp::stop() {
  if (device_ >= 0) {
    cudaSetDevice(device_);
    cudaDeviceSynchronize();
  }
  clear_state();
  qp_ = nullptr;
}

void RdmaRingsRxOp::clear_state() noexcept {
  rings_.clear();
  pending_credit_count_ = 0;
  frame_.reset();
  pending_start_metadata_.clear();
  immediate_scratch_.clear();
  active_series_ = false;
}

}  // namespace holoscan::ops
