#include "h5_writer_operator.hpp"

#include <nvcomp/zstd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <utility>

#include <unistd.h>

#include <nvtx3/nvtx3.hpp>

#include "bitshuffle_kernel_gpu.cuh"
#include "cbor_value.hpp"
#include "constants.h"
#include "fast_cbor_parse.h"
#include "nxmx_writer.hpp"

#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    cudaError_t err = call;                                                            \
    if (err != cudaSuccess)                                                            \
      throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +            \
                               std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
  } while (0)

#define NVCOMP_CHECK(call)                                                             \
  do {                                                                                 \
    nvcompStatus_t status = call;                                                      \
    if (status != nvcompSuccess)                                                       \
      throw std::runtime_error(std::string("nvcomp error at " __FILE__ ":") +          \
                               std::to_string(__LINE__) + ": " + std::to_string(status)); \
  } while (0)

namespace {

// Bitshuffle HDF5 framing: 8-byte big-endian image bytes, 4-byte big-endian block bytes,
// then per block a 4-byte big-endian compressed size and the zstd frame.
constexpr size_t kBlockBytes = 8192;
constexpr size_t kHeaderBytes = 12;
constexpr size_t kMaxChunks =
    (holoscan::ops::MAX_DETECTOR_SIZE * sizeof(uint16_t) + kBlockBytes - 1) / kBlockBytes;

__global__ void frame_images_kernel(const uint8_t* __restrict__ compressed,
                                    const size_t* __restrict__ compressed_sizes,
                                    const nvcompStatus_t* __restrict__ statuses, size_t chunks,
                                    size_t chunk_stride, uint32_t image_bytes,
                                    uint8_t* __restrict__ out, size_t out_stride,
                                    uint32_t* __restrict__ sizes) {
  __shared__ size_t offsets[kMaxChunks + 1];
  __shared__ bool ok;
  const size_t image = blockIdx.x;
  const size_t first = image * chunks;
  if (threadIdx.x == 0) {
    ok = true;
    size_t offset = kHeaderBytes;
    for (size_t c = 0; c < chunks; ++c) {
      offsets[c] = offset;
      ok = ok && statuses[first + c] == nvcompSuccess;
      offset += 4 + compressed_sizes[first + c];
    }
    offsets[chunks] = offset;
    ok = ok && offset <= out_stride;
  }
  __syncthreads();
  if (!ok) {
    if (threadIdx.x == 0) sizes[image] = 0;
    return;
  }
  uint8_t* dst = out + image * out_stride;
  if (threadIdx.x < 8) dst[threadIdx.x] = (uint64_t{image_bytes} >> (56 - 8 * threadIdx.x)) & 255;
  else if (threadIdx.x < 12) dst[threadIdx.x] = (kBlockBytes >> (24 - 8 * (threadIdx.x - 8))) & 255;
  for (size_t c = 0; c < chunks; ++c) {
    const size_t n = compressed_sizes[first + c];
    uint8_t* block = dst + offsets[c];
    if (threadIdx.x < 4) block[threadIdx.x] = (n >> (24 - 8 * threadIdx.x)) & 255;
    const uint8_t* src = compressed + (first + c) * chunk_stride;
    for (size_t b = threadIdx.x; b < n; b += blockDim.x) block[4 + b] = src[b];
  }
  if (threadIdx.x == 0) sizes[image] = static_cast<uint32_t>(offsets[chunks]);
}

size_t align_up(size_t value, size_t alignment) {
  return alignment > 1 ? (value + alignment - 1) / alignment * alignment : value;
}

std::string utc_now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()).count() % 1000;
  std::tm utc{};
  gmtime_r(&seconds, &utc);
  char text[64];
  std::snprintf(text, sizeof(text), "%04d-%02d-%02dT%02d:%02d:%02d.%03lld+00:00",
                utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
                utc.tm_sec, static_cast<long long>(millis));
  return text;
}

// File name stem for a series: $id is replaced by the series ID. The result must stay below
// output_dir, so absolute paths and ".." components are rejected.
std::string series_name(std::string pattern, uint64_t series_id) {
  const std::string id = std::to_string(series_id);
  for (size_t at = pattern.find("$id"); at != std::string::npos; at = pattern.find("$id", at + id.size()))
    pattern.replace(at, 3, id);
  const std::filesystem::path path(pattern);
  bool escapes = path.empty() || path.has_root_path() || !path.has_filename() ||
                 path.filename() == ".";
  for (const auto& part : path) escapes = escapes || part == "..";
  if (escapes)
    throw std::runtime_error("name_pattern must be a relative file name stem below output_dir: " +
                             pattern);
  return pattern;
}

std::string json_string(const std::string& value) {
  std::string out = "\"";
  for (char c : value) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

}  // namespace

namespace holoscan::ops {

H5WriterOp::~H5WriterOp() {
  try {
    stop();
  } catch (...) {
  }
}

void H5WriterOp::setup(OperatorSpec& spec) { spec.input(input, "input").queue_depth(4U); }

Contract H5WriterOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void H5WriterOp::start() {
  if (params().images_per_file == 0) throw std::invalid_argument("images_per_file must be positive");
  window_ = ::dectris::access(params().publication_window);
  completion_ = ::dectris::access_latch(params().completion_latch);
  active_series_ = false;
  ends_pushed_ = ends_done_ = 0;
  error_.clear();
  CUDA_CHECK(cudaSetDevice(static_cast<int>(params().device)));
  ::dectris::nxmx::register_filters();
  allocate();

  // Warm up both geometries before readiness on a private stream; this also proves that
  // the nvCOMP zstd path produces valid frames on this device.
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  try {
    CUDA_CHECK(cudaMemsetAsync(d_shuffled_, 0, batch_capacity_ * MAX_DETECTOR_SIZE * sizeof(uint16_t),
                               stream));
    for (auto& geometry : geometries_) {
      compress(reinterpret_cast<const uint16_t*>(d_shuffled_), batch_capacity_, geometry,
               slots_[0], stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (size_t i = 0; i < batch_capacity_; ++i)
        if (slots_[0].sizes[i] == 0) throw std::runtime_error("GPU zstd warm-up failed");
    }
  } catch (...) {
    cudaStreamDestroy(stream);
    release_resources();
    throw;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  writer_ = std::thread([this] { writer_loop(); });
}

void H5WriterOp::allocate() {
  batch_capacity_ = static_cast<size_t>(params().batch_size);
  const auto opts = nvcompBatchedZstdCompressDefaultOpts;
  nvcompAlignmentRequirements_t alignment{};
  NVCOMP_CHECK(nvcompBatchedZstdCompressGetRequiredAlignments(opts, &alignment));
  size_t max_out = 0;
  NVCOMP_CHECK(nvcompBatchedZstdCompressGetMaxOutputChunkSize(kBlockBytes, opts, &max_out));
  chunk_stride_ = align_up(max_out, alignment.output);
  slot_stride_ = kHeaderBytes + kMaxChunks * (4 + max_out);
  const size_t max_chunks = batch_capacity_ * kMaxChunks;
  const size_t max_bytes = batch_capacity_ * MAX_DETECTOR_SIZE * sizeof(uint16_t);
  NVCOMP_CHECK(nvcompBatchedZstdCompressGetTempSizeAsync(max_chunks, kBlockBytes, opts,
                                                         &temp_bytes_, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_shuffled_, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_compressed_, max_chunks * chunk_stride_));
  CUDA_CHECK(cudaMalloc(&d_compressed_sizes_, max_chunks * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_statuses_, max_chunks * sizeof(nvcompStatus_t)));
  if (temp_bytes_) CUDA_CHECK(cudaMalloc(&d_temp_, temp_bytes_));

  for (const auto [height, width] : {std::pair{DETECTOR_HEIGHT, DETECTOR_WIDTH},
                                     std::pair{MAX_DETECTOR_HEIGHT, MAX_DETECTOR_WIDTH}}) {
    Geometry geometry{height, width};
    const size_t image_bytes = size_t{height} * width * sizeof(uint16_t);
    geometry.chunks = (image_bytes + kBlockBytes - 1) / kBlockBytes;
    const size_t n = batch_capacity_ * geometry.chunks;
    std::vector<const void*> in_ptrs(n);
    std::vector<size_t> in_sizes(n);
    std::vector<void*> out_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
      const size_t image = i / geometry.chunks, chunk = i % geometry.chunks;
      in_ptrs[i] = d_shuffled_ + image * image_bytes + chunk * kBlockBytes;
      in_sizes[i] = std::min(kBlockBytes, image_bytes - chunk * kBlockBytes);
      out_ptrs[i] = d_compressed_ + i * chunk_stride_;
      if (reinterpret_cast<uintptr_t>(in_ptrs[i]) % alignment.input != 0)
        throw std::runtime_error("bitshuffle blocks violate nvCOMP zstd input alignment");
    }
    CUDA_CHECK(cudaMalloc(&geometry.in_ptrs, n * sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&geometry.in_sizes, n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&geometry.out_ptrs, n * sizeof(void*)));
    geometries_.push_back(geometry);
    CUDA_CHECK(cudaMemcpy(geometry.in_ptrs, in_ptrs.data(), n * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(geometry.in_sizes, in_sizes.data(), n * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(geometry.out_ptrs, out_ptrs.data(), n * sizeof(void*), cudaMemcpyHostToDevice));
  }

  slots_.resize(static_cast<size_t>(params().staging_batches));
  for (size_t i = 0; i < slots_.size(); ++i) {
    Slot& slot = slots_[i];
    CUDA_CHECK(cudaHostAlloc(&slot.host, batch_capacity_ * slot_stride_, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&slot.device), slot.host, 0));
    CUDA_CHECK(cudaHostAlloc(&slot.sizes, batch_capacity_ * sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&slot.device_sizes), slot.sizes, 0));
    CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming | cudaEventBlockingSync));
    free_slots_.push_back(i);
  }
  HOLOSCAN_LOG_INFO("H5WriterOp: {} staging batches of {} images, {} MiB pinned, {} MiB device",
                    slots_.size(), batch_capacity_,
                    slots_.size() * batch_capacity_ * slot_stride_ >> 20,
                    (max_bytes + max_chunks * chunk_stride_ + temp_bytes_) >> 20);
}

void H5WriterOp::release_resources() {
  for (auto& geometry : geometries_) {
    cudaFree(geometry.in_ptrs);
    cudaFree(geometry.in_sizes);
    cudaFree(geometry.out_ptrs);
  }
  geometries_.clear();
  for (auto& slot : slots_) {
    if (slot.ready) {
      cudaEventSynchronize(slot.ready);
      cudaEventDestroy(slot.ready);
    }
    if (slot.host) cudaFreeHost(slot.host);
    if (slot.sizes) cudaFreeHost(slot.sizes);
  }
  slots_.clear();
  free_slots_.clear();
  last_ready_ = nullptr;
  for (void* p : {static_cast<void*>(d_shuffled_), static_cast<void*>(d_compressed_),
                  static_cast<void*>(d_compressed_sizes_), static_cast<void*>(d_statuses_), d_temp_})
    if (p) cudaFree(p);
  d_shuffled_ = d_compressed_ = nullptr;
  d_compressed_sizes_ = nullptr;
  d_statuses_ = nullptr;
  d_temp_ = nullptr;
}

H5WriterOp::Geometry& H5WriterOp::geometry_for(uint32_t height, uint32_t width) {
  for (auto& geometry : geometries_)
    if (geometry.height == height && geometry.width == width) return geometry;
  throw std::invalid_argument("Unsupported image geometry for HDF5 writing");
}

void H5WriterOp::compress(const uint16_t* frames, size_t count, const Geometry& geometry,
                          Slot& slot, cudaStream_t stream) {
  nvtx3::scoped_range range("H5 bitshuffle/zstd compression");
  const size_t pixels = size_t{geometry.height} * geometry.width;
  // Device scratch is shared between activations, which may use different streams.
  if (last_ready_) CUDA_CHECK(cudaStreamWaitEvent(stream, last_ready_, 0));
  CUDA_CHECK(bitshuffle::shuffle_batch(reinterpret_cast<const uint8_t*>(frames), d_shuffled_,
                                       pixels, sizeof(uint16_t), count, stream));
  NVCOMP_CHECK(nvcompBatchedZstdCompressAsync(
      geometry.in_ptrs, geometry.in_sizes, kBlockBytes, count * geometry.chunks, d_temp_,
      temp_bytes_, geometry.out_ptrs, d_compressed_sizes_, nvcompBatchedZstdCompressDefaultOpts,
      d_statuses_, stream));
  frame_images_kernel<<<count, 256, 0, stream>>>(
      d_compressed_, d_compressed_sizes_, d_statuses_, geometry.chunks, chunk_stride_,
      static_cast<uint32_t>(pixels * sizeof(uint16_t)), slot.device, slot_stride_,
      slot.device_sizes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(slot.ready, stream));
  last_ready_ = slot.ready;
}

void H5WriterOp::push(Job job) {
  {
    std::lock_guard lock(mutex_);
    jobs_.push_back(std::move(job));
  }
  // The callback also waits on this variable for END completion.
  jobs_changed_.notify_all();
}

std::string H5WriterOp::error() const {
  std::lock_guard lock(mutex_);
  return error_;
}

expected<void, Error> H5WriterOp::compute(ExecutionContext& context) {
  try {
    if (context.cancellation().stop_requested()) return {};
    if (auto message = error(); !message.empty())
      return make_unexpected(Error{ErrorCode::kFailure, "HDF5 writer failed: " + message});
    HOLOSCAN_TRY(auto selected, input.selected_sample());
    const auto& batch = selected.data();
    if (batch.pipeline_ready) {
      if (window_) window_->release(batch.batch_id, params().window_consumer);
      return {};
    }
    if (batch.start_of_series) {
      dectris_start_metadata_t start{};
      if (fast_extract_start_metadata(batch.start_metadata.data(), batch.start_metadata.size(),
                                      &start) != 0 ||
          start.series_id != batch.series_id)
        return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid START image geometry"});
      // header_appendix enable_h5_writer: false skips this series, including GPU compression.
      bool write = true;
      const auto metadata = ::dectris::cbor::decode(batch.start_metadata.data(),
                                                    batch.start_metadata.size());
      if (const auto* appendix = metadata.get("header_appendix");
          appendix && appendix->untagged().kind == ::dectris::cbor::Value::Kind::kMap) {
        if (const auto* flag = appendix->get("enable_h5_writer")) {
          const auto value = flag->as_bool();
          if (!value)
            return make_unexpected(Error{ErrorCode::kProtocolError,
                                         "header_appendix enable_h5_writer must be a boolean"});
          write = *value;
        }
      }
      active_series_ = true;
      active_writing_ = write;
      active_series_id_ = batch.series_id;
      active_height_ = start.image_size_y;
      active_width_ = start.image_size_x;
      push(Job{.kind = JobKind::kStart, .series_id = batch.series_id, .height = active_height_,
               .width = active_width_, .write = write, .start_metadata = batch.start_metadata});
    } else if (!active_series_ || batch.series_id != active_series_id_) {
      return make_unexpected(Error{ErrorCode::kProtocolError, "Image batch has no matching START"});
    }

    const size_t count = batch.image_ids.size();
    if (count > 0) {
      const auto& frames = batch.frames;
      const auto shape = frames ? frames->shape() : std::vector<int64_t>{};
      if (!frames || shape.size() != 3 || shape[1] != active_height_ || shape[2] != active_width_ ||
          shape[0] < static_cast<int64_t>(count))
        return make_unexpected(Error{ErrorCode::kProtocolError, "Image geometry does not match START"});
      if (frames->device().device_type != kDLCUDA)
        return make_unexpected(Error{ErrorCode::kInvalidArgument, "HDF5 writer requires CUDA tensors"});
      const auto* pixels = frames->data_as<uint16_t>();
      if (!pixels)
        return make_unexpected(Error{ErrorCode::kInvalidArgument, "HDF5 writer requires uint16 images"});
      if (count > batch_capacity_)
        return make_unexpected(Error{ErrorCode::kInvalidArgument, "Batch exceeds writer batch_size"});
    }
    if (count > 0 && active_writing_) {
      const auto* pixels = batch.frames->data_as<uint16_t>();
      std::optional<size_t> slot_index;
      {
        std::unique_lock lock(mutex_);
        while (free_slots_.empty() && error_.empty()) {
          slots_changed_.wait_for(lock, std::chrono::milliseconds(10));
          if (context.cancellation().stop_requested()) return {};
        }
        if (!error_.empty())
          return make_unexpected(Error{ErrorCode::kFailure, "HDF5 writer failed: " + error_});
        slot_index = free_slots_.front();
        free_slots_.pop_front();
      }
      Slot& slot = slots_[*slot_index];
      slot.ids.assign(batch.image_ids.begin(), batch.image_ids.end());
      compress(pixels, count, geometry_for(active_height_, active_width_), slot,
               context.cuda_stream().get());
      push(Job{.kind = JobKind::kBatch, .series_id = batch.series_id, .height = active_height_,
               .width = active_width_, .slot = *slot_index});
    }

    if (batch.end_of_series) {
      push(Job{.kind = JobKind::kEnd, .series_id = batch.series_id,
               .series_images = batch.series_images});
      const uint64_t end = ++ends_pushed_;
      active_series_ = false;
      // Hold END until the series' files are closed so write failures fail this callback.
      std::unique_lock lock(mutex_);
      while (ends_done_ < end && error_.empty()) {
        jobs_changed_.wait_for(lock, std::chrono::milliseconds(10));
        if (context.cancellation().stop_requested()) return {};
      }
      if (!error_.empty())
        return make_unexpected(Error{ErrorCode::kFailure, "HDF5 writer failed: " + error_});
    }
    if (window_) window_->release(batch.batch_id, params().window_consumer);
    if (batch.end_of_series && completion_ && completion_->arrive()) context.request_completion();
    return {};
  } catch (const std::exception& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }
}

void H5WriterOp::writer_loop() {
  namespace nxmx = ::dectris::nxmx;
  struct Series {
    uint64_t id{};
    nxmx::SeriesLayout layout;
    nxmx::DataFile file;
    uint64_t written{}, dropped{}, files{}, bytes{};
    bool enabled{true};
    std::string name_source;
    std::chrono::steady_clock::time_point first_batch{}, last_write{};
  };
  std::optional<Series> series;
  cudaSetDevice(static_cast<int>(params().device));

  const auto finish = [&](uint64_t series_images, bool ended) {
    series->file.close();
    const double seconds = std::chrono::duration<double>(series->last_write - series->first_batch).count();
    const uint64_t raw = series->written * series->layout.height * series->layout.width * sizeof(uint16_t);
    std::string line = "[h5-series] {";
    line += "\"compressed_bytes\": " + std::to_string(series->bytes);
    line += ", \"compression_ratio\": " + (series->bytes ? std::to_string(double(raw) / series->bytes) : "null");
    line += ", \"data_files\": " + std::to_string(series->files);
    line += ", \"dropped_images\": " + std::to_string(series->dropped);
    line += ", \"ended\": " + std::string(ended ? "true" : "false");
    line += ", \"image_shape\": [" + std::to_string(series->layout.height) + ", " +
            std::to_string(series->layout.width) + "]";
    line += ", \"images_per_file\": " + std::to_string(series->layout.images_per_file);
    line += ", \"enabled\": " + std::string(series->enabled ? "true" : "false");
    line += ", \"master_file\": " + (series->enabled && series->layout.number_of_images
                                         ? json_string(series->layout.master_file().string())
                                         : std::string("null"));
    line += ", \"name_source\": " +
            (series->enabled ? json_string(series->name_source) : std::string("null"));
    line += ", \"number_of_images\": " + std::to_string(series->layout.number_of_images);
    line += ", \"series_id\": " + std::to_string(series->id);
    line += ", \"series_images\": " + std::to_string(series_images);
    line += ", \"timestamp_utc\": " + json_string(utc_now());
    line += ", \"write_interval_s\": " + std::to_string(seconds);
    line += ", \"written_fps\": " + (seconds > 0 ? std::to_string(series->written / seconds) : "null");
    line += ", \"written_images\": " + std::to_string(series->written) + "}";
    // One write per record keeps lines whole beside Python output and bypasses stdio buffering.
    line += '\n';
    for (size_t done = 0; done < line.size();) {
      const ssize_t n = ::write(STDOUT_FILENO, line.data() + done, line.size() - done);
      if (n <= 0) break;
      done += static_cast<size_t>(n);
    }
    series.reset();
  };

  for (;;) {
    Job job;
    {
      std::unique_lock lock(mutex_);
      jobs_changed_.wait(lock, [this] { return !jobs_.empty(); });
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    if (job.kind == JobKind::kStop) break;
    try {
      bool failed;
      {
        std::lock_guard lock(mutex_);
        failed = !error_.empty();
      }
      if (job.kind == JobKind::kStart && !failed) {
        if (series) finish(series->written, false);
        const auto start = ::dectris::cbor::decode(job.start_metadata.data(), job.start_metadata.size());
        const auto* announced = start.get("number_of_images");
        series.emplace();
        series->id = job.series_id;
        series->enabled = job.write;
        auto& layout = series->layout;
        layout.images_per_file = params().images_per_file;
        layout.height = job.height;
        layout.width = job.width;
        if (!job.write) {
          if (announced && announced->as_int() && *announced->as_int() >= 0)
            layout.number_of_images = static_cast<uint64_t>(*announced->as_int());
        } else {
          if (!announced || !announced->as_int() || *announced->as_int() < 0)
            throw std::runtime_error("START has no number_of_images for the NXmx layout");
          // A non-empty header_appendix name_pattern overrides the configured one for this series.
          std::string pattern = params().name_pattern;
          bool from_appendix = false;
          if (const auto* appendix = start.get("header_appendix");
              appendix && appendix->untagged().kind == ::dectris::cbor::Value::Kind::kMap) {
            if (const auto* announced_pattern = appendix->get("name_pattern")) {
              const auto text = announced_pattern->as_text();
              if (!text) throw std::runtime_error("header_appendix name_pattern must be a string");
              if (!text->empty()) {
                pattern = *text;
                from_appendix = true;
              }
            }
          }
          series->name_source = from_appendix ? "header_appendix" : "config";
          layout.prefix = std::filesystem::path(params().output_dir) / series_name(pattern, job.series_id);
          layout.number_of_images = static_cast<uint64_t>(*announced->as_int());
          if (const auto* channels = start.get("channels");
              channels && channels->untagged().kind == ::dectris::cbor::Value::Kind::kArray &&
              !channels->untagged().items.empty() && channels->untagged().items[0].as_text())
            layout.channel = *channels->untagged().items[0].as_text();
          // A series announcing no images gets no files; images that arrive anyway are dropped.
          if (layout.number_of_images > 0) nxmx::write_master_file(layout, start);
        }
      } else if (job.kind == JobKind::kBatch) {
        Slot& slot = slots_[job.slot];
        CUDA_CHECK(cudaEventSynchronize(slot.ready));
        if (!failed && series) {
          nvtx3::scoped_range range("H5 chunk writes");
          if (series->written == 0 && series->dropped == 0)
            series->first_batch = std::chrono::steady_clock::now();
          auto& layout = series->layout;
          for (size_t i = 0; i < slot.ids.size(); ++i) {
            const uint64_t id = slot.ids[i];
            if (id >= layout.number_of_images) {
              ++series->dropped;
              continue;
            }
            const uint32_t bytes = slot.sizes[i];
            if (bytes == 0) throw std::runtime_error("GPU zstd compression failed for an image");
            const uint64_t file_number = id / layout.images_per_file + 1;
            if (!series->file.is_open() || series->file.file_number() != file_number) {
              series->file.close();
              series->file.open(layout, file_number);
              ++series->files;
            }
            series->file.write_chunk(id, slot.host + i * slot_stride_, bytes);
            ++series->written;
            series->bytes += bytes;
            if (series->file.written() == series->file.images()) series->file.close();
          }
          series->last_write = std::chrono::steady_clock::now();
        }
        {
          std::lock_guard lock(mutex_);
          free_slots_.push_back(job.slot);
        }
        slots_changed_.notify_one();
      } else if (job.kind == JobKind::kEnd) {
        if (series && !failed) {
          if (series->dropped)
            HOLOSCAN_LOG_WARN("series {}: {} images beyond number_of_images {} were not written",
                              series->id, series->dropped, series->layout.number_of_images);
          finish(job.series_images, true);
        }
        std::lock_guard lock(mutex_);
        ++ends_done_;
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("H5WriterOp: {}", e.what());
      {
        std::lock_guard lock(mutex_);
        if (error_.empty()) error_ = e.what();
        if (job.kind == JobKind::kBatch) free_slots_.push_back(job.slot);
        if (job.kind == JobKind::kEnd) ++ends_done_;
      }
      slots_changed_.notify_all();
    }
    jobs_changed_.notify_all();
  }
  if (series) {
    try {
      finish(series->written, false);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("H5WriterOp: closing an unfinished series failed: {}", e.what());
    }
  }
}

void H5WriterOp::stop() {
  if (writer_.joinable()) {
    // Queued batches are written before the thread exits; their GPU work completes first.
    push(Job{.kind = JobKind::kStop});
    writer_.join();
  }
  release_resources();
}

}  // namespace holoscan::ops
