/**
 * GPU-accelerated BSLZ4 Decompression Operator Implementation
 */


#include "decompress_gpu_operator.hpp"

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include "constants.h"
#include "bitshuffle_kernel_gpu.cuh"
#include "fast_cbor_parse.h"

// Note: This operator uses the fast CBOR parser (10-50x faster than full parser).
// The original full CBOR parser validation path has been removed.

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      HOLOSCAN_LOG_ERROR("CUDA error at {}:{}: {}", __FILE__, __LINE__,      \
                         cudaGetErrorString(err));                            \
      throw std::runtime_error(std::string("CUDA error: ") +                 \
                               cudaGetErrorString(err));                      \
    }                                                                         \
  } while (0)

// nvcomp error checking macro
#define NVCOMP_CHECK(call)                                                    \
  do {                                                                        \
    nvcompStatus_t status = call;                                             \
    if (status != nvcompSuccess) {                                            \
      HOLOSCAN_LOG_ERROR("nvcomp error at {}:{}: {}", __FILE__, __LINE__,    \
                        (int)status);                                        \
      throw std::runtime_error("nvcomp error: " + std::to_string((int)status)); \
    }                                                                         \
  } while (0)

// ============================================================================
// BSLZ4 Preprocessing Helpers (for nvcomp)
// ============================================================================
// These functions prepare Dectris BSLZ4 format data for nvcomp's batched LZ4
// decompressor. They parse the BSLZ4 block structure and build the required
// device-side arrays for nvcomp. Actual LZ4 decompression is done by nvcomp.
// ============================================================================

namespace {  // Anonymous namespace - internal helpers

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  ~DeviceAllocation() { reset(); }

  void allocate(size_t bytes) {
    reset();
    if (bytes > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    }
  }

  void* get() const { return ptr_; }

  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }

  void reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
  }

 private:
  void* ptr_ = nullptr;
};

/**
 * @brief GPU kernel to parse BSLZ4 headers and extract block metadata
 *
 * Dectris BSLZ4 format structure per image (after 12-byte header):
 * - Sequence of blocks: [4-byte size (big-endian)][compressed LZ4 data]
 *
 * This kernel parses all blocks for each image in parallel.
 */
__global__ void parse_bslz4_headers_kernel(
    const uint8_t** __restrict__ d_compressed_ptrs,
    const size_t* __restrict__ compressed_sizes,
    size_t* __restrict__ block_offsets,     // Output: [batch_size * max_blocks]
    size_t* __restrict__ block_sizes,        // Output: [batch_size * max_blocks]
    int* __restrict__ num_blocks_per_image,  // Output: [batch_size]
    size_t batch_size,
    size_t max_blocks_per_image
) {
    int image_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (image_idx >= batch_size) return;

    const uint8_t* data = d_compressed_ptrs[image_idx];
    size_t data_size = compressed_sizes[image_idx];

    size_t offset = 0;
    int block_count = 0;

    // Parse all blocks for this image
    while (offset + 4 <= data_size && block_count < max_blocks_per_image) {
        // Read 4-byte block size header (big-endian)
        uint32_t block_size = (data[offset] << 24) | (data[offset+1] << 16) |
                             (data[offset+2] << 8) | data[offset+3];

        offset += 4;  // Skip header

        if (block_size == 0 || offset + block_size > data_size) {
            break;  // Invalid block
        }

        // Store block metadata
        size_t block_idx = image_idx * max_blocks_per_image + block_count;
        block_offsets[block_idx] = offset;
        block_sizes[block_idx] = block_size;

        offset += block_size;
        block_count++;
    }

    num_blocks_per_image[image_idx] = block_count;
}

/**
 * @brief Launch BSLZ4 header parsing on GPU
 */
cudaError_t parse_bslz4_headers_gpu(
    const uint8_t** d_compressed_ptrs,
    const size_t* d_compressed_sizes,
    size_t* d_block_offsets,
    size_t* d_block_sizes,
    int* d_num_blocks,
    size_t batch_size,
    size_t max_blocks_per_image,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    parse_bslz4_headers_kernel<<<blocks, threads, 0, stream>>>(
        d_compressed_ptrs, d_compressed_sizes,
        d_block_offsets, d_block_sizes, d_num_blocks,
        batch_size, max_blocks_per_image
    );

    return cudaGetLastError();
}

/**
 * @brief GPU kernel to build nvcomp parameter arrays from block metadata
 *
 * Converts our parsed block info into nvcomp's required device arrays.
 * Each thread handles one LZ4 block across all images.
 */
__global__ void build_nvcomp_arrays_kernel(
    const uint8_t** __restrict__ d_compressed_base_ptrs,
    const size_t* __restrict__ block_offsets,
    const size_t* __restrict__ block_sizes,
    const int* __restrict__ num_blocks_per_image,
    uint8_t* __restrict__ d_bitshuffled_base,  // Single base pointer (optimized)
    size_t image_stride,                        // Stride between images
    size_t batch_size,
    size_t max_blocks_per_image,
    size_t block_output_size,
    // nvcomp output arrays (device memory):
    const void** __restrict__ nvcomp_in_ptrs,
    size_t* __restrict__ nvcomp_in_sizes,
    void** __restrict__ nvcomp_out_ptrs,
    size_t* __restrict__ nvcomp_out_sizes
) {
    // Each thread handles one block across all images
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate total number of blocks
    int total_blocks = 0;
    for (size_t img = 0; img < batch_size; ++img) {
        total_blocks += num_blocks_per_image[img];
    }

    if (global_idx >= total_blocks) return;

    // Find which image and block this global_idx corresponds to
    int current_idx = 0;
    for (size_t img = 0; img < batch_size; ++img) {
        int img_blocks = num_blocks_per_image[img];
        if (global_idx < current_idx + img_blocks) {
            // This is in image 'img', block (global_idx - current_idx)
            int block_within_img = global_idx - current_idx;
            size_t meta_idx = img * max_blocks_per_image + block_within_img;

            // Build nvcomp arrays (calculate output pointer directly)
            nvcomp_in_ptrs[global_idx] = d_compressed_base_ptrs[img] + block_offsets[meta_idx];
            nvcomp_in_sizes[global_idx] = block_sizes[meta_idx];
            nvcomp_out_ptrs[global_idx] = d_bitshuffled_base + (img * image_stride) + (block_within_img * block_output_size);
            nvcomp_out_sizes[global_idx] = block_output_size;
            return;
        }
        current_idx += img_blocks;
    }
}

/**
 * @brief Launch nvcomp array building on GPU
 */
cudaError_t build_nvcomp_arrays_gpu(
    const uint8_t** d_compressed_base_ptrs,
    const size_t* d_block_offsets,
    const size_t* d_block_sizes,
    const int* d_num_blocks_per_image,
    uint8_t* d_bitshuffled_base,
    size_t image_stride,
    size_t batch_size,
    size_t max_blocks_per_image,
    size_t block_output_size,
    size_t total_blocks,
    const void** d_nvcomp_in_ptrs,
    size_t* d_nvcomp_in_sizes,
    void** d_nvcomp_out_ptrs,
    size_t* d_nvcomp_out_sizes,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    build_nvcomp_arrays_kernel<<<blocks, threads, 0, stream>>>(
        d_compressed_base_ptrs, d_block_offsets, d_block_sizes, d_num_blocks_per_image,
        d_bitshuffled_base, image_stride, batch_size, max_blocks_per_image, block_output_size,
        d_nvcomp_in_ptrs, d_nvcomp_in_sizes, d_nvcomp_out_ptrs, d_nvcomp_out_sizes
    );

    return cudaGetLastError();
}

}  // anonymous namespace

// ============================================================================

namespace holoscan::ops {

// ============================================================================
// Constructor / Destructor
// ============================================================================

DecompressGpuOp::~DecompressGpuOp() {
  free_gpu_resources();
}

// ============================================================================
// Operator Lifecycle Methods
// ============================================================================

void DecompressGpuOp::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(4U);
  spec.output(output, "output").produces_tensor(TensorOutputSpec{
      .representation = {.memory_kind = MemoryKind::kCudaDevice,
                         .dtype = DLDataType{kDLUInt, 16, 1}, .rank = 3U},
      .bounds = tensor_bounds(BATCH_SIZE * MAX_DETECTOR_SIZE * sizeof(uint16_t)),
      .storage = TensorOutputStorage::kRuntimePool});
}

Contract DecompressGpuOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void DecompressGpuOp::start() {
  publication_window_ = ::dectris::access(params().publication_window);
  input_window_ = ::dectris::access(params().input_window);
  last_capture_ns_ = -1;
  active_width_ = DETECTOR_WIDTH;
  active_height_ = DETECTOR_HEIGHT;
  active_series_id_ = 0;
  active_series_ = false;
  const size_t uncompressed_size = MAX_DETECTOR_SIZE * sizeof(uint16_t);
  const size_t max_blocks = (uncompressed_size + 8191) / 8192;
  max_compressed_size_per_image_ = 12 + uncompressed_size + uncompressed_size / 255 +
                                   max_blocks * (16 + 4);
  max_compressed_batch_size_ = max_compressed_size_per_image_ * BATCH_SIZE;
  allocate_gpu_resources();
}

expected<void, Error> DecompressGpuOp::compute(ExecutionContext& context) {
  try {
    if (context.cancellation().stop_requested()) return {};
    HOLOSCAN_TRY(auto selected, input.selected_sample());
    const auto& batch = selected.data();
    while (publication_window_ && !publication_window_->admits(batch.batch_id)) {
      if (context.cancellation().stop_requested()) return {};
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (context.cancellation().stop_requested()) return {};
    const size_t count = batch.image_ids.size();
    if (count > BATCH_SIZE || batch.sizes.size() != count || batch.offsets.size() != count)
      return make_unexpected(Error{ErrorCode::kProtocolError, "Invalid compressed batch metadata"});
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
    } else if (!batch.pipeline_ready &&
               (!active_series_ || batch.series_id != active_series_id_)) {
      return make_unexpected(Error{ErrorCode::kProtocolError, "Image has no matching START"});
    }
    const size_t image_pixels = batch.pipeline_ready
                                    ? static_cast<size_t>(DETECTOR_HEIGHT) * DETECTOR_WIDTH
                                    : static_cast<size_t>(active_height_) * active_width_;
    const std::array<int64_t, 3> shape{static_cast<int64_t>(std::max(size_t{1}, count)),
                                     static_cast<int64_t>(batch.pipeline_ready
                                                              ? DETECTOR_HEIGHT : active_height_),
                                     static_cast<int64_t>(batch.pipeline_ready
                                                              ? DETECTOR_WIDTH : active_width_)};
    HOLOSCAN_TRY(auto loan, output.allocate_tensor(shape));
    stream_ = context.cuda_stream().get();  // Borrowed; Runtime owns this stream.
    HOLOSCAN_TRY(auto writer, loan.write(context.cuda_stream()));
    if (batch.pipeline_ready) {
      auto warm_shape = [&](size_t height, size_t width) {
        std::vector<uint8_t> image(12, 0);
        size_t remaining_image = height * width * sizeof(uint16_t);
        while (remaining_image > 0) {
          const size_t bytes = std::min(size_t{8192}, remaining_image);
          std::vector<uint8_t> block{0xf0};
          size_t extension = bytes - 15;
          while (extension >= 255) { block.push_back(255); extension -= 255; }
          block.push_back(static_cast<uint8_t>(extension));
          block.resize(block.size() + bytes, 0);
          const auto n = static_cast<uint32_t>(block.size());
          for (int shift : {24, 16, 8, 0}) image.push_back((n >> shift) & 255);
          image.insert(image.end(), block.begin(), block.end());
          remaining_image -= bytes;
        }
        std::vector<uint8_t> data;
        data.reserve(image.size() * BATCH_SIZE);
        for (size_t i = 0; i < BATCH_SIZE; ++i)
          data.insert(data.end(), image.begin(), image.end());
        std::vector<const uint8_t*> pointers(BATCH_SIZE);
        std::vector<size_t> sizes(BATCH_SIZE, image.size());
        for (size_t i = 0; i < BATCH_SIZE; ++i)
          pointers[i] = data.data() + i * image.size();
        DeviceAllocation scratch;
        scratch.allocate(BATCH_SIZE * height * width * sizeof(uint16_t));
        return decompress_batch_gpu(pointers, sizes, BATCH_SIZE, height * width,
                                    scratch.as<uint16_t>());
      };
      if (!warm_shape(DETECTOR_HEIGHT, DETECTOR_WIDTH) ||
          !warm_shape(MAX_DETECTOR_HEIGHT, MAX_DETECTOR_WIDTH))
        return make_unexpected(Error{ErrorCode::kFailure, "GPU decompression warm-up failed"});
    }
    if (count > 0) {
      const auto* data = static_cast<const uint8_t*>(batch.payload->data());
      std::vector<const uint8_t*> pointers(count);
      std::vector<size_t> sizes(count);
      for (size_t i = 0; i < count; ++i) {
        if (batch.offsets[i] > static_cast<size_t>(batch.payload->nbytes()) ||
            batch.sizes[i] > static_cast<size_t>(batch.payload->nbytes()) - batch.offsets[i])
          return make_unexpected(Error{ErrorCode::kProtocolError, "Compressed span out of bounds"});
        pointers[i] = data + batch.offsets[i];
        sizes[i] = batch.sizes[i];
      }
      if (!decompress_batch_gpu(pointers, sizes, count, image_pixels, writer.data_as<uint16_t>()))
        return make_unexpected(Error{ErrorCode::kFailure, "GPU BSLZ4 decompression failed"});
    }
    HOLOSCAN_TRY(std::move(writer).commit());
    DecodedBatch decoded;
    decoded.batch_id = batch.batch_id;
    decoded.image_ids = batch.image_ids;
    decoded.end_of_series = batch.end_of_series;
    decoded.start_of_series = batch.start_of_series;
    decoded.series_id = batch.series_id;
    decoded.start_metadata = batch.start_metadata;
    decoded.series_images = batch.series_images;
    decoded.pipeline_ready = batch.pipeline_ready;
    auto capture = context.activation_time();
    // Several OnEach activations may share a reference time. The fanout join needs one key
    // per message, including the data batch and END that can become runnable together.
    capture.timestamp_ns = std::max(capture.timestamp_ns, last_capture_ns_ + 1);
    last_capture_ns_ = capture.timestamp_ns;
    auto published = output.emit_tensor(std::move(loan), decoded,
                                        EmitOptions{.capture_time = capture});
    if (!published) return make_unexpected(std::move(published).error());
    if (input_window_) input_window_->release(batch.batch_id);
    if (batch.end_of_series) active_series_ = false;
    return {};
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure, error.what()});
  }
}

void DecompressGpuOp::stop() {
  HOLOSCAN_LOG_INFO("DecompressGpuOp::stop() - Cleaning up GPU resources");
  free_gpu_resources();
}

// ============================================================================
// Helper Methods
// ============================================================================

void DecompressGpuOp::allocate_gpu_resources() {
  HOLOSCAN_LOG_INFO("  Allocating GPU buffers...");


  // Note: Pinned host buffer for compressed data is now provided by zmq_rx_operator
  // No need to allocate h_pinned_compressed_ here

  // Allocate pinned host memory for async operations
  CUDA_CHECK(cudaMallocHost(&h_pinned_num_blocks_, BATCH_SIZE * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h_pinned_statuses_, BATCH_SIZE * MAX_BLOCKS_PER_IMAGE * sizeof(nvcompStatus_t)));
  HOLOSCAN_LOG_INFO("    Pinned host buffers (async): {} KB",
                    (BATCH_SIZE * sizeof(int) + BATCH_SIZE * MAX_BLOCKS_PER_IMAGE * sizeof(nvcompStatus_t)) / 1024);

  // Allocate device memory for compressed data
  CUDA_CHECK(cudaMalloc(&d_compressed_data_, max_compressed_batch_size_));
  HOLOSCAN_LOG_INFO("    d_compressed_data_: {} MB", max_compressed_batch_size_ / (1024 * 1024));

  // Allocate device memory for intermediate (bitshuffled) data
  size_t intermediate_size = BATCH_SIZE * MAX_DETECTOR_HEIGHT * MAX_DETECTOR_WIDTH * sizeof(uint16_t);
  CUDA_CHECK(cudaMalloc(&d_intermediate_data_, intermediate_size));
  HOLOSCAN_LOG_INFO("    d_intermediate_data_: {} MB", intermediate_size / (1024 * 1024));

  // Allocate reusable input parameter arrays on device.
  CUDA_CHECK(cudaMalloc((void**)&d_compressed_ptrs_, BATCH_SIZE * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&d_compressed_sizes_, BATCH_SIZE * sizeof(size_t)));
  HOLOSCAN_LOG_INFO("    compressed parameter arrays: {} KB",
                    (BATCH_SIZE * (sizeof(void*) + sizeof(size_t))) / 1024);

  HOLOSCAN_LOG_INFO("  GPU buffer allocation complete");
}

void DecompressGpuOp::free_gpu_resources() {
  stream_ = nullptr;  // Borrowed from ExecutionContext; never destroy it here.

  // Note: h_pinned_compressed_ is now owned by zmq_rx_operator, don't free it here

  if (h_pinned_num_blocks_) {
    cudaFreeHost(h_pinned_num_blocks_);
    h_pinned_num_blocks_ = nullptr;
  }

  if (h_pinned_statuses_) {
    cudaFreeHost(h_pinned_statuses_);
    h_pinned_statuses_ = nullptr;
  }

  if (d_compressed_data_) { cudaFree(d_compressed_data_); d_compressed_data_ = nullptr; }
  if (d_intermediate_data_) { cudaFree(d_intermediate_data_); d_intermediate_data_ = nullptr; }

  if (d_compressed_ptrs_) { cudaFree(d_compressed_ptrs_); d_compressed_ptrs_ = nullptr; }
  if (d_compressed_sizes_) { cudaFree(d_compressed_sizes_); d_compressed_sizes_ = nullptr; }
}

bool DecompressGpuOp::decompress_batch_gpu(const std::vector<const uint8_t*>& compressed_ptrs,
                                            const std::vector<size_t>& compressed_sizes,
                                            size_t num_images,
                                            size_t image_pixels_per_frame,
                                            uint16_t* output_buffer) {
  nvtx3::scoped_range range("GPU Decompression Pipeline", nvtx3::rgb{0, 0, 255}); // Blue

  if (num_images == 0 || num_images > BATCH_SIZE) {
    HOLOSCAN_LOG_ERROR("Invalid num_images: {} (max: {})", num_images, BATCH_SIZE);
    return false;
  }

  // ============================================================================
  // Step 1: H2D Transfer of Compressed Data (Async)
  // ============================================================================

  size_t total_compressed_size = 0;
  std::vector<size_t> device_offsets(num_images);  // Offsets in d_compressed_data_

  {
    nvtx3::scoped_range h2d_range("H2D Transfer", nvtx3::rgb{255, 255, 0}); // Yellow

    // Calculate total size and device offsets
    // Data is already contiguous in pinned host buffer from zmq_rx_operator
    for (size_t i = 0; i < num_images; ++i) {
      device_offsets[i] = total_compressed_size;
      total_compressed_size += compressed_sizes[i];
    }

    if (total_compressed_size > max_compressed_batch_size_) {
      HOLOSCAN_LOG_ERROR("Total compressed size {} exceeds maximum {}",
                        total_compressed_size, max_compressed_batch_size_);
      return false;
    }

    // Single contiguous H2D transfer from pinned memory
    // compressed_ptrs[0] points to the start of the contiguous data
    CUDA_CHECK(cudaMemcpyAsync(d_compressed_data_,
                                compressed_ptrs[0],
                                total_compressed_size,
                                cudaMemcpyHostToDevice, stream_));

    HOLOSCAN_LOG_DEBUG("  H2D transfer: {} bytes (single contiguous copy)", total_compressed_size);
  }

  // ============================================================================
  // Step 2: Build Compressed Pointer Arrays (Keep Full Data with Headers)
  // ============================================================================

  {
    nvtx3::scoped_range ptr_range("Build Compressed Pointer Arrays", nvtx3::rgb{200, 200, 0}); // Yellow

    // Build compressed pointer array
    // Skip 12-byte Dectris header to point at the BSLZ4 block headers (4-byte each)
    std::vector<const uint8_t*> h_compressed_ptrs(num_images);
    std::vector<size_t> h_compressed_sizes_no_dectris(num_images);

    for (size_t i = 0; i < num_images; ++i) {
      // Skip 12-byte Dectris header (8 bytes orig_size + 4 bytes block_size)
      // Point to the start of BSLZ4 data (which has 4-byte block headers)
      h_compressed_ptrs[i] = d_compressed_data_ + device_offsets[i] + 12;
      h_compressed_sizes_no_dectris[i] = compressed_sizes[i] - 12;  // Size without Dectris header

      if (i < 3) {
        HOLOSCAN_LOG_DEBUG("    Image {}: device_offset={}, dectris_header_skipped, bslz4_size={}",
                          i, device_offsets[i] + 12, h_compressed_sizes_no_dectris[i]);
      }
    }

    // Copy compressed pointers and sizes to device
    CUDA_CHECK(cudaMemcpyAsync((void*)d_compressed_ptrs_, h_compressed_ptrs.data(),
                                num_images * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_compressed_sizes_, h_compressed_sizes_no_dectris.data(),
                                num_images * sizeof(size_t), cudaMemcpyHostToDevice, stream_));

    HOLOSCAN_LOG_DEBUG("  Built compressed pointer arrays: {} images", num_images);
  }

  // ============================================================================
  // Step 3: GPU Header Parsing - Find Internal LZ4 Blocks
  // ============================================================================

  size_t image_size = image_pixels_per_frame * sizeof(uint16_t);

  // Allocate temporary GPU arrays for header parsing results. RAII guards keep
  // all error paths and CUDA_CHECK exceptions from leaking device memory.
  DeviceAllocation block_offsets_alloc;
  DeviceAllocation block_sizes_alloc;
  DeviceAllocation num_blocks_alloc;
  block_offsets_alloc.allocate(num_images * MAX_BLOCKS_PER_IMAGE * sizeof(size_t));
  block_sizes_alloc.allocate(num_images * MAX_BLOCKS_PER_IMAGE * sizeof(size_t));
  num_blocks_alloc.allocate(num_images * sizeof(int));

  size_t* d_block_offsets_temp = block_offsets_alloc.as<size_t>();
  size_t* d_block_sizes_temp = block_sizes_alloc.as<size_t>();
  int* d_num_blocks_temp = num_blocks_alloc.as<int>();

  {
    nvtx3::scoped_range parse_range("Parse BSLZ4 Headers (GPU)", nvtx3::rgb{128, 0, 255}); // Purple

    // Parse headers on GPU to find all internal LZ4 blocks
    cudaError_t err = parse_bslz4_headers_gpu(
        (const uint8_t**)d_compressed_ptrs_,
        d_compressed_sizes_,
        d_block_offsets_temp,
        d_block_sizes_temp,
        d_num_blocks_temp,
        num_images,
        MAX_BLOCKS_PER_IMAGE,
        stream_);

    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("GPU header parsing failed: {}", cudaGetErrorString(err));
      return false;
    }

    HOLOSCAN_LOG_DEBUG("  GPU header parsing complete");
  }

  // Get total number of blocks across all images (async copy with pinned memory)
  CUDA_CHECK(cudaMemcpyAsync(h_pinned_num_blocks_, d_num_blocks_temp,
                            num_images * sizeof(int), cudaMemcpyDeviceToHost, stream_));

  // Synchronize only when we need to read the value
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  size_t total_blocks = 0;
  for (size_t i = 0; i < num_images; ++i) {
    total_blocks += h_pinned_num_blocks_[i];
  }

  HOLOSCAN_LOG_DEBUG("  Total LZ4 blocks: {} (avg {:.1f} per image)",
                    total_blocks, (double)total_blocks / num_images);

  if (total_blocks == 0) {
    HOLOSCAN_LOG_ERROR("No LZ4 blocks found in compressed data");
    return false;
  }

  // ============================================================================
  // Step 4: Build nvcomp Parameter Arrays on GPU
  // ============================================================================

  // Allocate nvcomp arrays for total_blocks (not just num_images).
  DeviceAllocation nvcomp_in_ptrs_alloc;
  DeviceAllocation nvcomp_in_sizes_alloc;
  DeviceAllocation nvcomp_out_ptrs_alloc;
  DeviceAllocation nvcomp_out_sizes_alloc;
  DeviceAllocation nvcomp_statuses_alloc;

  nvcomp_in_ptrs_alloc.allocate(total_blocks * sizeof(const void*));
  nvcomp_in_sizes_alloc.allocate(total_blocks * sizeof(size_t));
  nvcomp_out_ptrs_alloc.allocate(total_blocks * sizeof(void*));
  nvcomp_out_sizes_alloc.allocate(total_blocks * sizeof(size_t));
  nvcomp_statuses_alloc.allocate(total_blocks * sizeof(nvcompStatus_t));

  const void** d_nvcomp_in_ptrs = nvcomp_in_ptrs_alloc.as<const void*>();
  size_t* d_nvcomp_in_sizes = nvcomp_in_sizes_alloc.as<size_t>();
  void** d_nvcomp_out_ptrs = nvcomp_out_ptrs_alloc.as<void*>();
  size_t* d_nvcomp_out_sizes = nvcomp_out_sizes_alloc.as<size_t>();
  nvcompStatus_t* d_nvcomp_statuses = nvcomp_statuses_alloc.as<nvcompStatus_t>();

  {
    nvtx3::scoped_range build_range("Build nvcomp Arrays (GPU)", nvtx3::rgb{255, 128, 255}); // Pink

    // OPTIMIZED: Calculate intermediate pointers directly on GPU (no H2D copy!)
    const size_t BLOCK_OUTPUT_SIZE = 8192;  // 4096 elements * 2 bytes (from benchmark)

    // Build nvcomp arrays on GPU
    cudaError_t err = build_nvcomp_arrays_gpu(
        (const uint8_t**)d_compressed_ptrs_,
        d_block_offsets_temp,
        d_block_sizes_temp,
        d_num_blocks_temp,
        d_intermediate_data_,  // OPTIMIZED: Pass base pointer directly
        image_size,             // OPTIMIZED: Stride between images
        num_images,
        MAX_BLOCKS_PER_IMAGE,
        BLOCK_OUTPUT_SIZE,
        total_blocks,
        d_nvcomp_in_ptrs,
        d_nvcomp_in_sizes,
        d_nvcomp_out_ptrs,
        d_nvcomp_out_sizes,
        stream_);

    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("GPU nvcomp array building failed: {}", cudaGetErrorString(err));
      return false;
    }

    HOLOSCAN_LOG_DEBUG("  Built nvcomp arrays for {} blocks", total_blocks);
  }

  // ============================================================================
  // Step 5: Query nvcomp Temp Size
  // ============================================================================

  size_t nvcomp_temp_size = 0;
  DeviceAllocation nvcomp_temp_alloc;

  {
    nvtx3::scoped_range query_range("Query nvcomp Temp Size", nvtx3::rgb{128, 128, 128}); // Gray

    const size_t BLOCK_OUTPUT_SIZE = 8192;
    const size_t max_total_uncompressed = total_blocks * BLOCK_OUTPUT_SIZE;

    // Use default decompression options
    nvcompBatchedLZ4DecompressOpts_t decompress_opts = nvcompBatchedLZ4DecompressDefaultOpts;

    NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
        total_blocks,  // num_chunks
        BLOCK_OUTPUT_SIZE,  // max_uncompressed_chunk_bytes
        decompress_opts,  // decompression options
        &nvcomp_temp_size,  // output: temp_bytes
        max_total_uncompressed));  // max_total_uncompressed_bytes

    nvcomp_temp_alloc.allocate(nvcomp_temp_size);

    HOLOSCAN_LOG_DEBUG("  Allocated nvcomp temp buffer: {} MB", nvcomp_temp_size / (1024 * 1024));
  }

  // ============================================================================
  // Step 6: LZ4 Decompression using nvcomp (Batch of Blocks)
  // ============================================================================

  {
    nvtx3::scoped_range lz4_range("LZ4 Decompression (nvcomp)", nvtx3::rgb{0, 128, 255}); // Blue

    HOLOSCAN_LOG_DEBUG("  Calling nvcompBatchedLZ4DecompressAsync:");
    HOLOSCAN_LOG_DEBUG("    total_blocks: {}", total_blocks);
    HOLOSCAN_LOG_DEBUG("    nvcomp_temp_size: {} bytes", nvcomp_temp_size);

    // Use default decompression options
    nvcompBatchedLZ4DecompressOpts_t decompress_opts = nvcompBatchedLZ4DecompressDefaultOpts;

    NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
        d_nvcomp_in_ptrs,          // device_compressed_chunk_ptrs
        d_nvcomp_in_sizes,         // device_compressed_chunk_bytes
        d_nvcomp_out_sizes,        // device_uncompressed_buffer_bytes (max output sizes)
        d_nvcomp_out_sizes,        // device_uncompressed_chunk_bytes (actual output sizes)
        total_blocks,              // num_chunks
        nvcomp_temp_alloc.get(),   // device_temp_ptr
        nvcomp_temp_size,          // temp_bytes
        d_nvcomp_out_ptrs,         // device_uncompressed_chunk_ptrs
        decompress_opts,           // decompress_opts (NEW in 5.0)
        d_nvcomp_statuses,         // device_statuses
        stream_));                 // stream

    HOLOSCAN_LOG_DEBUG("  LZ4 decompression launched for {} blocks", total_blocks);
  }

  // ============================================================================
  // Step 7: Bitshuffle Unshuffle
  // ============================================================================

  {
    nvtx3::scoped_range unshuffle_range("Bitshuffle Unshuffle", nvtx3::rgb{255, 128, 0}); // Orange

    // Build pointer arrays for bitshuffle (input from intermediate, output to final buffer)
    std::vector<uint8_t*> h_intermediate_ptrs(num_images);
    std::vector<uint8_t*> h_output_ptrs(num_images);

    for (size_t i = 0; i < num_images; ++i) {
      h_intermediate_ptrs[i] = d_intermediate_data_ + i * image_size;
      h_output_ptrs[i] = reinterpret_cast<uint8_t*>(output_buffer) + i * image_size;
    }

    DeviceAllocation intermediate_ptrs_alloc;
    DeviceAllocation output_ptrs_alloc;
    intermediate_ptrs_alloc.allocate(num_images * sizeof(uint8_t*));
    output_ptrs_alloc.allocate(num_images * sizeof(uint8_t*));

    uint8_t** d_intermediate_ptrs_array = intermediate_ptrs_alloc.as<uint8_t*>();
    uint8_t** d_output_ptrs_array = output_ptrs_alloc.as<uint8_t*>();

    CUDA_CHECK(cudaMemcpyAsync(d_intermediate_ptrs_array, h_intermediate_ptrs.data(),
                                num_images * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_output_ptrs_array, h_output_ptrs.data(),
                                num_images * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));

    // Call bitshuffle with pointer arrays
    size_t num_elements = image_size / sizeof(uint16_t);  // Number of uint16 elements per image
    cudaError_t err = bitshuffle::unshuffle_batch(
        d_intermediate_ptrs_array,
        d_output_ptrs_array,
        num_elements,
        sizeof(uint16_t),  // element_size
        num_images,        // batch_size
        stream_);

    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Bitshuffle unshuffle failed: {}", cudaGetErrorString(err));
      return false;
    }

    HOLOSCAN_LOG_DEBUG("  Bitshuffle unshuffle: {} images", num_images);
  }

  // ============================================================================
  // Step 8: Verify nvcomp Decompression Status
  // ============================================================================

  // OPTIMIZED: Make status verification async (optional for production)
  #ifdef ENABLE_STATUS_VERIFICATION
  {
    nvtx3::scoped_range verify_range("Verify Status", nvtx3::rgb{255, 255, 255}); // White

    // Use async copy to pinned memory (non-blocking)
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_statuses_, d_nvcomp_statuses,
                              total_blocks * sizeof(nvcompStatus_t),
                              cudaMemcpyDeviceToHost, stream_));

    // Synchronize to ensure status is ready
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    size_t num_failures = 0;
    for (size_t i = 0; i < total_blocks; ++i) {
      if (h_pinned_statuses_[i] != nvcompSuccess) {
        if (num_failures < 10) {  // Only log first 10 failures
          HOLOSCAN_LOG_ERROR("  Block {} decompression failed with status: {}", i, (int)h_pinned_statuses_[i]);
        }
        num_failures++;
      }
    }

    if (num_failures > 0) {
      HOLOSCAN_LOG_ERROR("GPU decompression failed: {}/{} blocks", num_failures, total_blocks);
      return false;
    }
  }
  #endif  // ENABLE_STATUS_VERIFICATION

  HOLOSCAN_LOG_DEBUG("GPU decompression complete: {} images, {} blocks", num_images, total_blocks);

  return true;
}

}  // namespace holoscan::ops
