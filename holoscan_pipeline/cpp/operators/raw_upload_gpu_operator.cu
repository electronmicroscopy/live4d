#include "raw_upload_gpu_operator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <thread>

#include "fast_cbor_parse.h"

namespace holoscan::ops {
namespace {

Error cuda_error(const char* what, cudaError_t status) {
  return Error{ErrorCode::kFailure,
               std::string(what) + ": " + std::string(cudaGetErrorString(status))};
}

}  // namespace

RawUploadGpuOp::~RawUploadGpuOp() { stop(); }

void RawUploadGpuOp::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(4U);
  spec.output(output, "output").produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kCudaDevice,
                         .dtype = DLDataType{kDLUInt, 16, 1}, .rank = 3U},
      .bounds = tensor_bounds(BATCH_SIZE * MAX_DETECTOR_SIZE * sizeof(uint16_t)),
      .storage = TensorOutputStorage::kRuntimePool});
}

Contract RawUploadGpuOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void RawUploadGpuOp::start() {
  device_ = static_cast<int>(params().device);
  const cudaError_t status = cudaSetDevice(device_);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("RawUploadGpuOp: cudaSetDevice: ") +
                             cudaGetErrorString(status));
  }
  publication_window_ = ::dectris::access(params().publication_window);
  input_window_ = ::dectris::access(params().input_window);
  last_capture_ns_ = -1;
  batch_id_ = 0;
  active_width_ = DETECTOR_WIDTH;
  active_height_ = DETECTOR_HEIGHT;
  active_series_id_ = 0;
  active_series_ = false;
}

expected<void, Error> RawUploadGpuOp::compute(ExecutionContext& context) {
  if (context.cancellation().stop_requested()) return {};
  cudaError_t status = cudaSetDevice(device_);
  if (status != cudaSuccess) return make_unexpected(cuda_error("cudaSetDevice", status));
  HOLOSCAN_TRY(auto selected, input.selected_sample());
  const auto& batch = selected.data();
  if (!batch.pipeline_ready && batch.destination != params().destination) {
    if (input_window_) input_window_->release(batch.batch_id);
    return {};  // Host tensor fanout only; this lane has nothing to do with the batch.
  }
  while (publication_window_ && !publication_window_->admits(batch_id_ + 1)) {
    if (context.cancellation().stop_requested()) return {};
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (context.cancellation().stop_requested()) return {};

  const size_t count = batch.image_ids.size();
  if (count > BATCH_SIZE || batch.sizes.size() != count || batch.offsets.size() != count)
    return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid raw batch metadata"});
  if (batch.start_of_series) {
    dectris_start_metadata_t start_metadata{};
    if (fast_extract_start_metadata(batch.start_metadata.data(), batch.start_metadata.size(),
                                    &start_metadata) != 0 ||
        start_metadata.series_id != batch.series_id)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid START image geometry"});
    active_width_ = start_metadata.image_size_x;
    active_height_ = start_metadata.image_size_y;
    active_series_id_ = batch.series_id;
    active_series_ = true;
  } else if (!batch.pipeline_ready && (!active_series_ || batch.series_id != active_series_id_)) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "Image has no matching START"});
  }

  const size_t height = batch.pipeline_ready ? DETECTOR_HEIGHT : active_height_;
  const size_t width = batch.pipeline_ready ? DETECTOR_WIDTH : active_width_;
  const size_t image_bytes = height * width * sizeof(uint16_t);
  const std::array<int64_t, 3> shape{static_cast<int64_t>(std::max(size_t{1}, count)),
                                     static_cast<int64_t>(height), static_cast<int64_t>(width)};
  HOLOSCAN_TRY(auto loan, output.allocate_tensor(shape));
  const cudaStream_t stream = context.cuda_stream().get();
  HOLOSCAN_TRY(auto writer, loan.write(context.cuda_stream()));

  if (batch.pipeline_ready) {
    status = cudaMemsetAsync(writer.data(), 0, image_bytes, stream);
    if (status != cudaSuccess) return make_unexpected(cuda_error("READY tensor", status));
  } else if (count > 0) {
    // The host side must be one contiguous run of count images in order.
    const size_t total = count * image_bytes;
    for (size_t i = 0; i < count; ++i) {
      if (batch.sizes[i] != image_bytes || batch.offsets[i] != i * image_bytes)
        return make_unexpected(Error{ErrorCode::kProtocolError, "Raw image span does not match START"});
    }
    if (!batch.payload || static_cast<size_t>(batch.payload->nbytes()) < total)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Raw payload is too small"});
    status = cudaMemcpyAsync(writer.data(), batch.payload->data(), total, cudaMemcpyHostToDevice,
                             stream);
    if (status != cudaSuccess) return make_unexpected(cuda_error("host-to-device copy", status));
    // The pinned host staging is runtime-owned; finish reading it before the input is released.
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) return make_unexpected(cuda_error("upload synchronize", status));
  }
  HOLOSCAN_TRY(std::move(writer).commit());

  DecodedBatch decoded;
  decoded.batch_id = ++batch_id_;
  decoded.image_ids = batch.image_ids;
  decoded.end_of_series = batch.end_of_series;
  decoded.start_of_series = batch.start_of_series;
  decoded.series_id = batch.series_id;
  decoded.start_metadata = batch.start_metadata;
  decoded.series_images = batch.series_images;
  decoded.pipeline_ready = batch.pipeline_ready;
  auto capture = context.activation_time();
  // Several OnEach activations may share a reference time; the fanout join needs one key each.
  capture.timestamp_ns = std::max(capture.timestamp_ns, last_capture_ns_ + 1);
  last_capture_ns_ = capture.timestamp_ns;
  auto published = output.emit_tensor(std::move(loan), decoded,
                                      EmitOptions{.capture_time = capture});
  if (!published) return make_unexpected(std::move(published).error());
  if (input_window_) input_window_->release(batch.batch_id);
  if (batch.end_of_series) active_series_ = false;
  return {};
}

void RawUploadGpuOp::stop() {
  device_ = -1;
}

}  // namespace holoscan::ops
