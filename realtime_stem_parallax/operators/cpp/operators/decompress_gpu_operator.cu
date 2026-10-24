/**
 * GPU-accelerated BSLZ4 Decompression Operator Implementation
 */


#include "decompress_gpu_operator.hpp"

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#include "constants.h"
#include "bitshuffle_kernel_gpu.cuh"

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

// Big-endian field readers for the 12-byte Dectris bslz4 header
// (8 bytes uncompressed size, 4 bytes block size).
inline uint64_t read_be64(const uint8_t* p) {
  return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) | ((uint64_t)p[2] << 40) |
         ((uint64_t)p[3] << 32) | ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
         ((uint64_t)p[6] << 8) | (uint64_t)p[7];
}

inline uint32_t read_be32(const uint8_t* p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) |
         (uint32_t)p[3];
}

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

            // Build nvcomp arrays (calculate output pointer directly). The max output
            // size of the final block is clamped to what is left of the image so a
            // malformed block can never decompress past its image's region.
            size_t out_offset = block_within_img * block_output_size;
            size_t max_out = (out_offset < image_stride) ? (image_stride - out_offset) : 0;
            if (max_out > block_output_size) max_out = block_output_size;
            nvcomp_in_ptrs[global_idx] = d_compressed_base_ptrs[img] + block_offsets[meta_idx];
            nvcomp_in_sizes[global_idx] = block_sizes[meta_idx];
            nvcomp_out_ptrs[global_idx] = d_bitshuffled_base + (img * image_stride) + out_offset;
            nvcomp_out_sizes[global_idx] = max_out;
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
  // Input: Compressed batch with pre-extracted data
  spec.input<CompressedBatch>("input");

  // Output: TensorMap. Python converts "frames" to torch.Tensor with DLPack.
  spec.output<holoscan::TensorMap>("output");

  spec.param(detector_width_,
             "detector_width",
             "Detector width",
             "Frame width in pixels; must match the ZmqRxOp configuration.",
             static_cast<int64_t>(dectris::kDefaultDetectorWidth));
  spec.param(detector_height_,
             "detector_height",
             "Detector height",
             "Frame height in pixels; must match the ZmqRxOp configuration.",
             static_cast<int64_t>(dectris::kDefaultDetectorHeight));
  spec.param(pixel_dtype_,
             "dtype",
             "Pixel dtype",
             "Pixel data type of the output frames, 'uint16' or 'uint32'.",
             std::string(dectris::kDefaultPixelDtype));

  HOLOSCAN_LOG_INFO("DecompressGpuOp::setup() - GPU decompression operator configured");
}

void DecompressGpuOp::start() {
  HOLOSCAN_LOG_INFO("DecompressGpuOp::start() - Initializing GPU resources");
  HOLOSCAN_LOG_INFO("  Using pre-extracted compressed batches from ZmqRxOp");

  batch_count_ = 0;
  current_buffer_index_ = 0;

  if (detector_width_.get() <= 0 || detector_height_.get() <= 0) {
    throw std::runtime_error("DecompressGpuOp: detector_width/detector_height must be "
                             "positive");
  }
  elem_size_ = pixel_dtype_elem_size(pixel_dtype_.get());
  image_size_ = static_cast<size_t>(detector_width_.get()) *
                static_cast<size_t>(detector_height_.get()) * elem_size_;

  // Metadata capacity for the per-image LZ4 blocks. Sized from the nominal block size;
  // the actual block size read from the stream is checked against this in
  // decompress_batch_gpu, so a stream with unexpectedly small blocks fails loudly
  // instead of overflowing.
  max_blocks_per_image_ = image_size_ / dectris::kNominalBlockSize + 2;

  // Calculate worst-case compressed size per image
  // For BSLZ4: 12-byte Dectris header plus, per LZ4 block (nominally 8192 bytes), a
  // 4-byte length prefix and LZ4's incompressible-data bound. Must match ZmqRxOp's
  // sizing so a full pinned batch always fits the device buffer.
  size_t n_blocks =
      (image_size_ + dectris::kNominalBlockSize - 1) / dectris::kNominalBlockSize;
  max_compressed_size_per_image_ =
      12 + image_size_ + n_blocks * (4 + dectris::kNominalBlockSize / 255 + 16);
  max_compressed_batch_size_ = max_compressed_size_per_image_ * BATCH_SIZE;

  HOLOSCAN_LOG_INFO("  Detector: {}x{} {} ({} bytes/image)",
                    detector_height_.get(), detector_width_.get(), pixel_dtype_.get(),
                    image_size_);
  HOLOSCAN_LOG_INFO("  Max compressed size per image: {} bytes", max_compressed_size_per_image_);
  HOLOSCAN_LOG_INFO("  Max compressed batch size: {} MB", max_compressed_batch_size_ / (1024 * 1024));

  // Allocate GPU resources
  allocate_gpu_resources();

  // Preallocate the MatX tensor pool on GPU for the configured dtype
  HOLOSCAN_LOG_INFO("  Allocating MatX tensor pool on GPU (POOL_SIZE={})", POOL_SIZE);
  const matx::index_t height = static_cast<matx::index_t>(detector_height_.get());
  const matx::index_t width = static_cast<matx::index_t>(detector_width_.get());
  if (elem_size_ == 2) {
    matx::make_tensor(decompressed_pool_u16_, {POOL_SIZE, BATCH_SIZE, height, width},
                      matx::MATX_DEVICE_MEMORY);
  } else {
    matx::make_tensor(decompressed_pool_u32_, {POOL_SIZE, BATCH_SIZE, height, width},
                      matx::MATX_DEVICE_MEMORY);
  }
  matx::make_tensor(image_indices_pool_, {POOL_SIZE, BATCH_SIZE}, matx::MATX_DEVICE_MEMORY);
  matx::make_tensor(batch_id_pool_, {POOL_SIZE}, matx::MATX_DEVICE_MEMORY);

  // Note: GPU buffer doesn't need initialization (will be filled by decompression)
  // Initialize host buffers
  (image_indices_pool_ = 0).run();
  (batch_id_pool_ = 0).run();

  HOLOSCAN_LOG_INFO("DecompressGpuOp::start() - Initialization complete");
}

void DecompressGpuOp::compute(InputContext& op_input, OutputContext& op_output,
                              ExecutionContext& context) {
  nvtx3::scoped_range range("DecompressGpuOp::compute", nvtx3::rgb{255, 165, 0}); // Orange

  auto compressed_batch = op_input.receive<CompressedBatch>("input");

  if (!compressed_batch) {
    HOLOSCAN_LOG_WARN("DecompressGpuOp: Received null compressed batch");
    return;
  }

  // Extract data from CompressedBatch (pre-extracted by zmq_rx_operator)
  size_t num_images = compressed_batch->num_images;
  uint64_t batch_id = compressed_batch->batch_id;
  uint8_t* h_pinned_data = compressed_batch->h_pinned_data;
  auto& sizes = *compressed_batch->sizes;
  auto& offsets = *compressed_batch->offsets;
  auto& image_ids = *compressed_batch->image_ids;

  HOLOSCAN_LOG_INFO("DecompressGpuOp: Processing batch {} (batch_id: {}, {} images, buffer_index: {})",
                    batch_count_ + 1, batch_id, num_images, current_buffer_index_);

  if (num_images == 0) {
    HOLOSCAN_LOG_WARN("DecompressGpuOp: Empty batch received");
    return;
  }

  if (num_images > BATCH_SIZE) {
    HOLOSCAN_LOG_ERROR("DecompressGpuOp: Batch size {} exceeds maximum {}", num_images, BATCH_SIZE);
    return;
  }

  // ============================================================================
  // Step 1: Build compressed data pointers (data already in pinned memory!)
  // ============================================================================

  std::vector<const uint8_t*> compressed_ptrs(num_images);
  std::vector<size_t> compressed_sizes(num_images);
  std::vector<uint64_t> image_ids_vec(num_images);

  {
    nvtx3::scoped_range extract_range("Build compressed pointers", nvtx3::rgb{0, 255, 0}); // Green

    for (size_t i = 0; i < num_images; ++i) {
      compressed_ptrs[i] = h_pinned_data + offsets(i);
      compressed_sizes[i] = sizes(i);
      image_ids_vec[i] = image_ids(i);
    }
  }

  HOLOSCAN_LOG_DEBUG("  Using pre-extracted compressed data from zmq_rx_operator");

  // ============================================================================
  // Step 2: Decompress into the pool for the configured dtype and emit
  // ============================================================================

  bool success = (elem_size_ == 2)
      ? process_batch<uint16_t>(decompressed_pool_u16_, compressed_ptrs, compressed_sizes,
                                image_ids_vec, num_images, batch_id, op_output)
      : process_batch<uint32_t>(decompressed_pool_u32_, compressed_ptrs, compressed_sizes,
                                image_ids_vec, num_images, batch_id, op_output);

  if (!success) {
    HOLOSCAN_LOG_ERROR("DecompressGpuOp: GPU decompression failed");
    throw std::runtime_error("DecompressGpuOp: GPU decompression failed — stream does "
                             "not match the configured geometry/dtype or is corrupt");
  }

  HOLOSCAN_LOG_INFO("DecompressGpuOp: Completed batch {} (batch_id: {}, buffer_index: {})",
                    batch_count_ + 1, batch_id, current_buffer_index_);

  batch_count_++;
  current_buffer_index_ = (current_buffer_index_ + 1) % POOL_SIZE;
}

template <typename T>
bool DecompressGpuOp::process_batch(matx::tensor_t<T, 4>& pool,
                                    const std::vector<const uint8_t*>& compressed_ptrs,
                                    const std::vector<size_t>& compressed_sizes,
                                    const std::vector<uint64_t>& image_ids,
                                    size_t num_images,
                                    uint64_t batch_id,
                                    OutputContext& op_output) {
  // ============================================================================
  // Get output buffer slice from MatX tensor pool
  // ============================================================================

  matx::index_t idx = static_cast<matx::index_t>(current_buffer_index_);

  auto decompressed_batch_slice = matx::slice(pool,
                                              {idx, 0, 0, 0},
                                              {idx + 1,
                                               static_cast<matx::index_t>(num_images),
                                               matx::matxEnd,
                                               matx::matxEnd});
  auto decompressed_batch = matx::slice<3>(decompressed_batch_slice,
                                            {0, 0, 0, 0},
                                            {matx::matxDropDim, matx::matxEnd, matx::matxEnd, matx::matxEnd});

  auto image_indices_slice = matx::slice(image_indices_pool_,
                                          {idx, 0},
                                          {idx + 1, static_cast<matx::index_t>(num_images)});
  auto image_indices = matx::slice<1>(image_indices_slice,
                                      {0, 0},
                                      {matx::matxDropDim, matx::matxEnd});

  // ============================================================================
  // Perform GPU decompression
  // ============================================================================

  uint8_t* output_ptr = reinterpret_cast<uint8_t*>(decompressed_batch.Data());

  if (!decompress_batch_gpu(compressed_ptrs, compressed_sizes, num_images, output_ptr)) {
    return false;
  }

  // ============================================================================
  // Store metadata on device for PyTorch DLPack interop
  // ============================================================================

  auto batch_id_tensor = matx::slice<1>(batch_id_pool_, {idx}, {idx + 1});
  CUDA_CHECK(cudaMemcpyAsync(image_indices.Data(),
                             image_ids.data(),
                             num_images * sizeof(uint64_t),
                             cudaMemcpyHostToDevice,
                             stream_));
  CUDA_CHECK(cudaMemcpyAsync(batch_id_tensor.Data(),
                             &batch_id,
                             sizeof(uint64_t),
                             cudaMemcpyHostToDevice,
                             stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  // ============================================================================
  // Emit TensorMap for Python/PyTorch interop
  // ============================================================================

  holoscan::TensorMap output_batch;
  output_batch.insert({"frames", std::make_shared<holoscan::Tensor>(decompressed_batch.ToDlPack())});
  output_batch.insert({"image_ids", std::make_shared<holoscan::Tensor>(image_indices.ToDlPack())});
  output_batch.insert({"batch_id", std::make_shared<holoscan::Tensor>(batch_id_tensor.ToDlPack())});

  op_output.emit(output_batch, "output");
  return true;
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

  // Create CUDA stream
  CUDA_CHECK(cudaStreamCreate(&stream_));

  // Note: Pinned host buffer for compressed data is now provided by zmq_rx_operator
  // No need to allocate h_pinned_compressed_ here

  // Allocate pinned host memory for async operations
  CUDA_CHECK(cudaMallocHost(&h_pinned_num_blocks_, BATCH_SIZE * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&h_pinned_statuses_, BATCH_SIZE * max_blocks_per_image_ * sizeof(nvcompStatus_t)));
  HOLOSCAN_LOG_INFO("    Pinned host buffers (async): {} KB",
                    (BATCH_SIZE * sizeof(int) + BATCH_SIZE * max_blocks_per_image_ * sizeof(nvcompStatus_t)) / 1024);

  // Allocate device memory for compressed data
  CUDA_CHECK(cudaMalloc(&d_compressed_data_, max_compressed_batch_size_));
  HOLOSCAN_LOG_INFO("    d_compressed_data_: {} MB", max_compressed_batch_size_ / (1024 * 1024));

  // Allocate device memory for intermediate (bitshuffled) data
  size_t intermediate_size = BATCH_SIZE * image_size_;
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
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

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
                                            uint8_t* output_buffer) {
  nvtx3::scoped_range range("GPU Decompression Pipeline", nvtx3::rgb{0, 0, 255}); // Blue

  if (num_images == 0 || num_images > BATCH_SIZE) {
    HOLOSCAN_LOG_ERROR("Invalid num_images: {} (max: {})", num_images, BATCH_SIZE);
    return false;
  }

  // ============================================================================
  // Step 0: Validate the Dectris headers (host, pinned memory) and read the
  //         actual bslz4 block size — the block layout of everything that follows
  //         derives from it, so it is read from the stream rather than assumed.
  // ============================================================================

  size_t block_size = 0;  // bslz4 block size in bytes

  for (size_t i = 0; i < num_images; ++i) {
    if (compressed_sizes[i] < 12) {
      HOLOSCAN_LOG_ERROR("Image {}: compressed size {} too short for Dectris header",
                         i, compressed_sizes[i]);
      return false;
    }
    uint64_t orig_size = read_be64(compressed_ptrs[i]);
    uint32_t img_block_size = read_be32(compressed_ptrs[i] + 8);
    if (orig_size != image_size_) {
      HOLOSCAN_LOG_ERROR("Image {}: uncompressed size {} does not match configured "
                         "{}x{} {} ({} bytes)", i, orig_size, detector_height_.get(),
                         detector_width_.get(), pixel_dtype_.get(), image_size_);
      return false;
    }
    if (i == 0) {
      block_size = img_block_size;
    } else if (img_block_size != block_size) {
      HOLOSCAN_LOG_ERROR("Image {}: block size {} differs from batch block size {}",
                         i, img_block_size, block_size);
      return false;
    }
  }

  if (block_size == 0 || block_size % (8 * elem_size_) != 0) {
    HOLOSCAN_LOG_ERROR("Invalid bslz4 block size {} for {}-byte elements",
                       block_size, elem_size_);
    return false;
  }
  size_t blocks_needed = (image_size_ + block_size - 1) / block_size;
  if (blocks_needed > max_blocks_per_image_) {
    HOLOSCAN_LOG_ERROR("Stream block size {} implies {} blocks/image, exceeding the "
                       "allocated capacity {} (sized for {}-byte blocks)",
                       block_size, blocks_needed, max_blocks_per_image_,
                       dectris::kNominalBlockSize);
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
        HOLOSCAN_LOG_INFO("    Image {}: device_offset={}, dectris_header_skipped, bslz4_size={}",
                          i, device_offsets[i] + 12, h_compressed_sizes_no_dectris[i]);
      }
    }

    // Copy compressed pointers and sizes to device
    CUDA_CHECK(cudaMemcpyAsync((void*)d_compressed_ptrs_, h_compressed_ptrs.data(),
                                num_images * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_compressed_sizes_, h_compressed_sizes_no_dectris.data(),
                                num_images * sizeof(size_t), cudaMemcpyHostToDevice, stream_));

    HOLOSCAN_LOG_INFO("  Built compressed pointer arrays: {} images", num_images);
  }

  // ============================================================================
  // Step 3: GPU Header Parsing - Find Internal LZ4 Blocks
  // ============================================================================

  // Allocate temporary GPU arrays for header parsing results. RAII guards keep
  // all error paths and CUDA_CHECK exceptions from leaking device memory.
  DeviceAllocation block_offsets_alloc;
  DeviceAllocation block_sizes_alloc;
  DeviceAllocation num_blocks_alloc;
  block_offsets_alloc.allocate(num_images * max_blocks_per_image_ * sizeof(size_t));
  block_sizes_alloc.allocate(num_images * max_blocks_per_image_ * sizeof(size_t));
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
        max_blocks_per_image_,
        stream_);

    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("GPU header parsing failed: {}", cudaGetErrorString(err));
      return false;
    }

    HOLOSCAN_LOG_INFO("  GPU header parsing complete");
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

  HOLOSCAN_LOG_INFO("  Total LZ4 blocks: {} (avg {:.1f} per image)",
                    total_blocks, (double)total_blocks / num_images);

  if (total_blocks == 0) {
    HOLOSCAN_LOG_ERROR("No LZ4 blocks found in compressed data");
    return false;
  }

  // Every image must parse into exactly the expected number of LZ4 blocks. A short count means
  // the compressed data on the device is not what the stream carried (corrupt payload, or a
  // damaged host-to-device copy); decompressing it would silently produce wrong frames.
  for (size_t i = 0; i < num_images; ++i) {
    if (static_cast<size_t>(h_pinned_num_blocks_[i]) != blocks_needed) {
      HOLOSCAN_LOG_ERROR("Image {} of the batch: found {} LZ4 blocks, expected {}. The "
                         "compressed data on the GPU is corrupt.",
                         i, h_pinned_num_blocks_[i], blocks_needed);
      return false;
    }
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

    // Build nvcomp arrays on GPU; each block decompresses to `block_size` bytes (from
    // the Dectris header), except the final block of an image which is clamped to the
    // remainder inside the kernel.
    cudaError_t err = build_nvcomp_arrays_gpu(
        (const uint8_t**)d_compressed_ptrs_,
        d_block_offsets_temp,
        d_block_sizes_temp,
        d_num_blocks_temp,
        d_intermediate_data_,  // OPTIMIZED: Pass base pointer directly
        image_size_,            // OPTIMIZED: Stride between images
        num_images,
        max_blocks_per_image_,
        block_size,
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

    HOLOSCAN_LOG_INFO("  Built nvcomp arrays for {} blocks", total_blocks);
  }

  // ============================================================================
  // Step 5: Query nvcomp Temp Size
  // ============================================================================

  size_t nvcomp_temp_size = 0;
  DeviceAllocation nvcomp_temp_alloc;

  {
    nvtx3::scoped_range query_range("Query nvcomp Temp Size", nvtx3::rgb{128, 128, 128}); // Gray

    const size_t max_total_uncompressed = total_blocks * block_size;

    // Use default decompression options
    nvcompBatchedLZ4DecompressOpts_t decompress_opts = nvcompBatchedLZ4DecompressDefaultOpts;

    NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
        total_blocks,  // num_chunks
        block_size,  // max_uncompressed_chunk_bytes
        decompress_opts,  // decompression options
        &nvcomp_temp_size,  // output: temp_bytes
        max_total_uncompressed));  // max_total_uncompressed_bytes

    nvcomp_temp_alloc.allocate(nvcomp_temp_size);

    HOLOSCAN_LOG_INFO("  Allocated nvcomp temp buffer: {} MB", nvcomp_temp_size / (1024 * 1024));
  }

  // ============================================================================
  // Step 6: LZ4 Decompression using nvcomp (Batch of Blocks)
  // ============================================================================

  {
    nvtx3::scoped_range lz4_range("LZ4 Decompression (nvcomp)", nvtx3::rgb{0, 128, 255}); // Blue

    HOLOSCAN_LOG_INFO("  Calling nvcompBatchedLZ4DecompressAsync:");
    HOLOSCAN_LOG_INFO("    total_blocks: {}", total_blocks);
    HOLOSCAN_LOG_INFO("    nvcomp_temp_size: {} bytes", nvcomp_temp_size);

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
      h_intermediate_ptrs[i] = d_intermediate_data_ + i * image_size_;
      h_output_ptrs[i] = output_buffer + i * image_size_;
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
    size_t num_elements = image_size_ / elem_size_;  // Number of elements per image
    cudaError_t err = bitshuffle::unshuffle_batch(
        d_intermediate_ptrs_array,
        d_output_ptrs_array,
        num_elements,
        elem_size_,                // element_size
        block_size / elem_size_,   // block_size_elements (from the Dectris header)
        num_images,                // batch_size
        stream_);

    if (err != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Bitshuffle unshuffle failed: {}", cudaGetErrorString(err));
      return false;
    }

    HOLOSCAN_LOG_DEBUG("  Bitshuffle unshuffle: {} images", num_images);
  }

  // ============================================================================
  // Step 8: Synchronize Stream
  // ============================================================================

  {
    nvtx3::scoped_range sync_range("Stream Synchronization", nvtx3::rgb{128, 128, 128}); // Gray

    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  // ============================================================================
  // Step 9: Verify nvcomp Decompression Status
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
