#include "rdma_rx_operator.hpp"

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

namespace holoscan::ops {
namespace {

constexpr size_t kFrameAlign = 128;
constexpr size_t kFrameHeaderBytes = 128;
constexpr uint32_t kFrameMagic = 0x46584D49U;  // "IMXF" in little-endian wire order.
constexpr uint16_t kProtocolVersion = 2;
constexpr uint16_t kFrameStart = 1;
constexpr uint16_t kFrameImage = 2;
constexpr uint16_t kFrameEnd = 3;
constexpr uint16_t kFrameWrap = 0xFFFF;
constexpr uint16_t kDataTypeUint16 = 2;
constexpr uint16_t kCompressionNone = 0;
constexpr size_t kImageBytes = dectris::kDetectorPixels * sizeof(uint16_t);
constexpr size_t kMaxStartBytes = 16 * 1024;
constexpr int kMaxReceiveDeficit = 256;
constexpr auto kGpuHeaderSettle = std::chrono::milliseconds(10);
constexpr auto kGpuHeaderRetry = std::chrono::microseconds(20);

Error make_error(ErrorCode code, std::string message) {
  return Error{code, std::move(message)};
}

uint16_t load_le16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t load_le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t load_le64(const uint8_t* p) {
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= static_cast<uint64_t>(p[i]) << (8 * i);
  return value;
}

__global__ void gather_mapped_bytes(const uint8_t* source, uint8_t* destination, size_t bytes) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < bytes) destination[index] = source[index];
}

std::string cuda_detail(cudaError_t status) {
  return std::string(cudaGetErrorString(status));
}

std::string sdk_detail(const Error& error) {
  std::string detail(error_code_name(error.code));
  if (!error.message.empty()) detail += ": " + error.message;
  return detail;
}

std::string loan_detail(const TensorOutputLoan& loan) {
  return "device_id=" + std::to_string(loan.device_id().value) +
         ", memory_kind=" + std::to_string(static_cast<unsigned>(loan.memory_kind())) +
         ", byte_size=" + std::to_string(loan.byte_size());
}

struct StreamSyncOnAbandon {
  cudaStream_t stream{};
  bool& work_enqueued;
  bool& published;

  ~StreamSyncOnAbandon() {
    if (work_enqueued && !published) cudaStreamSynchronize(stream);
  }
};

}  // namespace

RdmaRxOp::CudaEvent::~CudaEvent() {
  if (handle) cudaEventDestroy(handle);
}

RdmaRxOp::~RdmaRxOp() { stop(); }

void RdmaRxOp::setup(OperatorSpec& spec) {
  spec.output(output, "output").produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kCudaDevice,
                         .dtype = DLDataType{kDLUInt, 16, 1}, .rank = 3U},
      .bounds = tensor_bounds(dectris::kMaxBatchSize * kImageBytes),
      .storage = TensorOutputStorage::kRuntimePool});
}

Contract RdmaRxOp::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::microseconds(100)});
  return result;
}

void RdmaRxOp::start() {
  const auto& config = params();
  if (config.qp_handle == 0 || config.ring_address == 0 || config.ring_device_address == 0 ||
      config.credit_address == 0 || config.ring_bytes < 2 * kFrameAlign ||
      config.ring_bytes % kFrameAlign != 0 || config.ring_address % kFrameAlign != 0 ||
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

  const cudaError_t set_device_status = cudaSetDevice(device_);
  if (set_device_status != cudaSuccess) {
    throw std::runtime_error("Invalid RDMA receiver configuration: "
                             "cudaSetDevice: " + cuda_detail(set_device_status));
  }
  if (gpu_ring_) {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(
        &attributes, reinterpret_cast<const void*>(static_cast<uintptr_t>(ring_address_)));
    if (status != cudaSuccess || attributes.type != cudaMemoryTypeDevice ||
        attributes.device != device_ || ring_device_address_ != ring_address_) {
      throw std::runtime_error("Invalid RDMA receiver configuration: GPU ring address or device "
                               "does not match");
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
    void* alias = nullptr;
    const cudaError_t status = cudaHostGetDevicePointer(
        &alias, reinterpret_cast<void*>(static_cast<uintptr_t>(ring_address_)), 0);
    if (status != cudaSuccess || reinterpret_cast<uintptr_t>(alias) != ring_device_address_) {
      throw std::runtime_error("Invalid RDMA receiver configuration: mapped host ring alias "
                               "does not match");
    }
  }

  next_sequence_ = 0;
  reader_position_ = 0;
  safe_credit_position_ = 0;
  sent_credit_position_ = 0;
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
  pending_immediates_.clear();
  pending_credits_.clear();
  frame_.reset();
}

expected<void, Error> RdmaRxOp::compute(ExecutionContext& context) {
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
  HOLOSCAN_TRY(retry_credit_write());
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
                                       static_cast<int64_t>(dectris::kDetectorHeight),
                                       static_cast<int64_t>(dectris::kDetectorWidth)};
    auto allocated = output.allocate_tensor(shape);
    if (!allocated) {
      return make_unexpected(make_error(
          ErrorCode::kFailure, "RDMA tensor allocation failed: allocate_tensor data shape [" +
                                   std::to_string(batch_size_) + ",96,96]: " +
                                   sdk_detail(allocated.error())));
    }
    loan.emplace(std::move(*allocated));
    auto result = loan->write(producer_stream);
    if (!result) {
      return make_unexpected(make_error(
          ErrorCode::kFailure, "RDMA tensor allocation failed: loan.write data shape [" +
                                   std::to_string(batch_size_) + ",96,96]: " +
                                   sdk_detail(result.error()) + " (" + loan_detail(*loan) + ")"));
    }
    writer.emplace(std::move(*result));
    return {};
  };

  while (image_count < batch_size_) {
    if (context.cancellation().stop_requested()) return {};
    HOLOSCAN_TRY(poll_credits());
    HOLOSCAN_TRY(retry_credit_write());

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
      if (take == 0) {
        return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
      }

      const size_t source_offset = frame.offset + kFrameHeaderBytes +
                                   static_cast<size_t>(frame.images_consumed) * kImageBytes;
      const size_t destination_offset = image_count * kImageBytes;
      const size_t copy_bytes = take * kImageBytes;
      const auto* source = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(
          ring_device_address_ + source_offset));
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
        auto event = std::make_shared<CudaEvent>();
        cuda_status = cudaEventCreateWithFlags(&event->handle, cudaEventDisableTiming);
        if (cuda_status == cudaSuccess) cuda_status = cudaEventRecord(event->handle, stream);
        if (cuda_status != cudaSuccess) {
          return make_unexpected(make_error(ErrorCode::kFailure,
                                            "Invalid RDMA stream: CUDA read event: " +
                                                cuda_detail(cuda_status)));
        }
        HOLOSCAN_TRY(record_credit(reader_position_, event));
        frame_.reset();
        frame_event_waited = false;
      }

      if (image_count == batch_size_) {
        if (frame_) {
          auto event = std::make_shared<CudaEvent>();
          cuda_status = cudaEventCreateWithFlags(&event->handle, cudaEventDisableTiming);
          if (cuda_status == cudaSuccess) cuda_status = cudaEventRecord(event->handle, stream);
          if (cuda_status != cudaSuccess) {
            return make_unexpected(make_error(ErrorCode::kFailure,
                                              "Invalid RDMA stream: CUDA read event: " +
                                                  cuda_detail(cuda_status)));
          }
          frame_->last_read = std::move(event);
        }
        break;
      }
      continue;
    }

    if (pending_control_ != ControlKind::kNone) break;
    if (!pending_immediates_.empty()) {
      const uint32_t immediate = pending_immediates_.front();
      pending_immediates_.pop_front();
      HOLOSCAN_TRY(read_next_frame(immediate));
      continue;
    }

    HOLOSCAN_TRY(poll_completions());
    if (pending_control_ != ControlKind::kNone) break;
    if (!pending_immediates_.empty()) continue;
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
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
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

expected<void, Error> RdmaRxOp::poll_completions() {
  if (recv_deficit_ > 0) HOLOSCAN_TRY(repost_recvs(0));

  const size_t room = max_batch_ > pending_credits_.size()
                          ? max_batch_ - pending_credits_.size()
                          : 0;
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
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
  }

  if (result.n_imm > 0) {
    HOLOSCAN_TRY(repost_recvs(result.n_imm));
    if (gpu_ring_) HOLOSCAN_TRY(flush_gpudirect_writes());
    for (int i = 0; i < result.n_imm; ++i) pending_immediates_.push_back(immediate_scratch_[i]);
  }
  if (!ready_emitted_ && pending_control_ == ControlKind::kNone)
    pending_control_ = ControlKind::kReady;
  return {};
}

expected<void, Error> RdmaRxOp::poll_credits() {
  while (!pending_credits_.empty()) {
    const auto& credit = pending_credits_.front();
    if (credit.event) {
      const cudaError_t status = cudaEventQuery(credit.event->handle);
      if (status == cudaErrorNotReady) break;
      if (status != cudaSuccess) {
        return make_unexpected(make_error(ErrorCode::kFailure,
                                          "Invalid RDMA stream: CUDA credit event: " +
                                              cuda_detail(status)));
      }
    }
    safe_credit_position_ = credit.position;
    pending_credits_.pop_front();
  }
  return {};
}

expected<void, Error> RdmaRxOp::retry_credit_write() {
  if (safe_credit_position_ <= sent_credit_position_) return {};
  const uint64_t value = htole64(safe_credit_position_);
  const int rc = imx_qp_write_inline(qp_, &value, sizeof(value), credit_address_, credit_rkey_);
  if (rc < 0) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      std::string("RDMA connection failed: ") + imx_last_error()));
  }
  if (rc == 1) sent_credit_position_ = safe_credit_position_;
  return {};
}

expected<void, Error> RdmaRxOp::repost_recvs(int consumed) {
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

expected<void, Error> RdmaRxOp::read_next_frame(uint32_t immediate) {
  if (immediate != static_cast<uint32_t>(next_sequence_)) {
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
  }
  const size_t offset = static_cast<size_t>(reader_position_ % ring_bytes_);
  std::array<uint8_t, kFrameHeaderBytes> raw{};
  const auto deadline = std::chrono::steady_clock::now() + kGpuHeaderSettle;
  for (;;) {
    if (gpu_ring_) {
      const cudaError_t status = cudaMemcpy(raw.data(),
                                            reinterpret_cast<const void*>(static_cast<uintptr_t>(
                                                ring_address_ + offset)),
                                            raw.size(), cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) {
        return make_unexpected(make_error(ErrorCode::kFailure,
                                          "Invalid RDMA stream: GPU header read: " +
                                              cuda_detail(status)));
      }
    } else {
      std::memcpy(raw.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(
                                  ring_address_ + offset)),
                  raw.size());
    }
    const bool valid = load_le32(raw.data()) == kFrameMagic &&
                       load_le16(raw.data() + 4) == kProtocolVersion &&
                       load_le64(raw.data() + 8) == next_sequence_ &&
                       load_le64(raw.data() + 16) == reader_position_;
    if (valid) break;
    if (!gpu_ring_ || std::chrono::steady_clock::now() >= deadline) {
      return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
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

  if (kind == kFrameWrap) {
    if (header.data_size != 0 || offset == 0 || reader_position_ > UINT64_MAX - (ring_bytes_ - offset)) {
      return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
    }
    reader_position_ += ring_bytes_ - offset;
    ++next_sequence_;
    return record_credit(reader_position_);
  }
  if (kind != kFrameStart && kind != kFrameImage && kind != kFrameEnd) {
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
  }
  if (header.data_size > ring_bytes_ - offset - kFrameHeaderBytes ||
      header.data_size > UINT64_MAX - kFrameHeaderBytes - (kFrameAlign - 1)) {
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
  }
  const uint64_t span64 = (kFrameHeaderBytes + header.data_size + kFrameAlign - 1) &
                          ~static_cast<uint64_t>(kFrameAlign - 1);
  if (span64 > ring_bytes_ - offset || reader_position_ > UINT64_MAX - span64) {
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
  }
  const uint64_t frame_end = reader_position_ + span64;

  if (kind == kFrameStart) {
    if (active_series_) {
      return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
    }
    if (header.data_size > kMaxStartBytes) {
      return make_unexpected(make_error(ErrorCode::kProtocolError,
                                        "RDMA START metadata exceeds 16 KiB"));
    }
    pending_start_metadata_.resize(static_cast<size_t>(header.data_size));
    if (!pending_start_metadata_.empty()) {
      if (gpu_ring_) {
        const cudaError_t status = cudaMemcpy(
            pending_start_metadata_.data(),
            reinterpret_cast<const void*>(static_cast<uintptr_t>(
                ring_address_ + offset + kFrameHeaderBytes)),
            pending_start_metadata_.size(), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
          return make_unexpected(make_error(ErrorCode::kFailure,
                                            "Invalid RDMA stream: START read: " +
                                                cuda_detail(status)));
        }
      } else {
        std::memcpy(pending_start_metadata_.data(),
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(
                        ring_address_ + offset + kFrameHeaderBytes)),
                    pending_start_metadata_.size());
      }
    }
    uint64_t metadata_series_id = 0;
    if (pending_start_metadata_.empty() ||
        fast_extract_series_id(pending_start_metadata_.data(), pending_start_metadata_.size(),
                               &metadata_series_id) != 0 ||
        metadata_series_id != header.series_id) {
      return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
    }
    active_series_ = true;
    active_series_id_ = header.series_id;
    active_series_images_ = 0;
    pending_control_ = ControlKind::kStart;
    pending_control_series_id_ = header.series_id;
    pending_control_images_ = 0;
    reader_position_ = frame_end;
    ++next_sequence_;
    return record_credit(reader_position_);
  }

  if (kind == kFrameEnd) {
    if (!active_series_ || header.series_id != active_series_id_) {
      return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
    }
    pending_control_ = ControlKind::kEnd;
    pending_control_series_id_ = active_series_id_;
    pending_control_images_ = active_series_images_;
    active_series_ = false;
    reader_position_ = frame_end;
    ++next_sequence_;
    return record_credit(reader_position_);
  }

  if (!active_series_ || header.series_id != active_series_id_ || header.image_count == 0 ||
      header.dtype != kDataTypeUint16 || header.compression != kCompressionNone ||
      header.dim_x != dectris::kDetectorWidth || header.dim_y != dectris::kDetectorHeight ||
      header.channel != 0 || header.image_count > UINT64_MAX / kImageBytes ||
      header.data_size != static_cast<uint64_t>(header.image_count) * kImageBytes ||
      header.image_id > UINT64_MAX - header.image_count ||
      active_series_images_ > UINT64_MAX - header.image_count) {
    return make_unexpected(make_error(
        ErrorCode::kProtocolError,
        "RDMA image is outside the supported 96x96 uint16 scope"));
  }
  active_series_images_ += header.image_count;
  reader_position_ = frame_end;
  ++next_sequence_;
  frame_.emplace(FrameState{header, offset, static_cast<size_t>(span64), 0, {}});
  return {};
}

expected<void, Error> RdmaRxOp::record_credit(uint64_t position,
                                             const std::shared_ptr<CudaEvent>& event) {
  if (!pending_credits_.empty() && pending_credits_.back().position >= position) {
    return make_unexpected(make_error(ErrorCode::kProtocolError, "Invalid RDMA stream"));
  }
  pending_credits_.push_back(CreditState{position, event});
  return {};
}

expected<void, Error> RdmaRxOp::flush_gpudirect_writes() {
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

expected<void, Error> RdmaRxOp::emit_control(ExecutionContext& context, ControlKind kind) {
  const std::array<int64_t, 3> shape{1, static_cast<int64_t>(dectris::kDetectorHeight),
                                     static_cast<int64_t>(dectris::kDetectorWidth)};
  auto allocated = output.allocate_tensor(shape);
  if (!allocated) {
    return make_unexpected(make_error(
        ErrorCode::kFailure, "RDMA tensor allocation failed: allocate_tensor control shape "
                             "[1,96,96]: " + sdk_detail(allocated.error())));
  }
  auto loan = std::move(*allocated);
  auto producer_stream = context.cuda_stream();
  auto writer_result = loan.write(producer_stream);
  if (!writer_result) {
    return make_unexpected(make_error(ErrorCode::kFailure,
                                      "RDMA tensor allocation failed: loan.write control shape "
                                      "[1,96,96]: " +
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
    status = cudaMemsetAsync(destination, 0, kImageBytes, producer_stream.get());
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

void RdmaRxOp::stop() {
  if (device_ >= 0) {
    cudaSetDevice(device_);
    cudaDeviceSynchronize();
  }
  clear_state();
  qp_ = nullptr;
}

void RdmaRxOp::clear_state() noexcept {
  pending_immediates_.clear();
  pending_credits_.clear();
  frame_.reset();
  pending_start_metadata_.clear();
  immediate_scratch_.clear();
  active_series_ = false;
}

}  // namespace holoscan::ops
