/**
 * GPU-accelerated BSLZ4 Decompression Operator
 *
 * This operator performs BSLZ4 decompression entirely on the GPU using:
 * - Pre-extracted bslz4 payloads from ZmqRxOp
 * - H2D transfer of compressed data
 * - GPU header parsing (12-byte Dectris headers)
 * - nvcomp for LZ4 decompression
 * - Custom CUDA kernel for bitshuffle unshuffle
 *
 * Detector geometry and pixel dtype ('uint16' or 'uint32') are runtime parameters and
 * must match the ZmqRxOp configuration; the per-image Dectris headers are validated
 * against them, and the bslz4 block size is read from those headers rather than assumed.
 *
 * Input:  CompressedBatch from ZmqRxOp
 * Output: TensorMap with decompressed frames on GPU in the configured dtype. Python
 *         sinks convert the "frames" tensor to torch.Tensor with torch.from_dlpack().
 */

#ifndef DECOMPRESS_GPU_OPERATOR_HPP
#define DECOMPRESS_GPU_OPERATOR_HPP

#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/lz4.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/core/domain/tensor_map.hpp>
#include <matx.h>

#include "constants.h"
#include "pipeline_types.hpp"

namespace holoscan::ops {

class DecompressGpuOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DecompressGpuOp)

  DecompressGpuOp() = default;
  ~DecompressGpuOp();

  // Pool size for circular buffer of output tensors
  static constexpr size_t POOL_SIZE = dectris::kPoolSize;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  // ============================================================================
  // Configuration
  // ============================================================================

  Parameter<int64_t> detector_width_;
  Parameter<int64_t> detector_height_;
  Parameter<std::string> pixel_dtype_;

  // Derived from the parameters in start()
  size_t elem_size_ = 0;               // Bytes per pixel (2 or 4)
  size_t image_size_ = 0;              // Uncompressed image size in bytes
  size_t max_blocks_per_image_ = 0;    // For metadata allocation; checked against the
                                       // actual block size read from the stream

  // ============================================================================
  // GPU Memory Management
  // ============================================================================

  // Device buffers
  uint8_t* d_compressed_data_ = nullptr;      // Compressed input buffer
  uint8_t* d_intermediate_data_ = nullptr;    // LZ4 output (bitshuffled)

  // Reusable compressed input parameter arrays (all on device)
  const void** d_compressed_ptrs_ = nullptr;  // Pointers to each compressed image (const for nvcomp API)
  size_t* d_compressed_sizes_ = nullptr;      // Size of each compressed image

  // ============================================================================
  // Host Memory (Pinned for fast H2D transfer)
  // ============================================================================

  // Note: h_pinned_compressed_ is now provided by zmq_rx_operator via CompressedBatch
  // No longer owned by this operator

  int* h_pinned_num_blocks_ = nullptr;        // Pinned host buffer for num_blocks (async copy)
  nvcompStatus_t* h_pinned_statuses_ = nullptr; // Pinned host buffer for status verification (async)

  // ============================================================================
  // MatX Tensor Pool (GPU memory)
  // ============================================================================

  // Pool of decompressed image batches on GPU: [POOL_SIZE, BATCH_SIZE, H, W]. MatX
  // tensors are typed at compile time, so there is one pool per supported dtype and
  // only the configured one is allocated in start().
  matx::tensor_t<uint16_t, 4> decompressed_pool_u16_;
  matx::tensor_t<uint32_t, 4> decompressed_pool_u32_;
  matx::tensor_t<uint64_t, 2> image_indices_pool_;           // [POOL_SIZE, BATCH_SIZE]
  matx::tensor_t<uint64_t, 1> batch_id_pool_;                // [POOL_SIZE]

  // ============================================================================
  // CUDA Resources
  // ============================================================================

  cudaStream_t stream_ = nullptr;

  // ============================================================================
  // Buffer Management
  // ============================================================================

  size_t current_buffer_index_ = 0;
  size_t batch_count_ = 0;

  // Sizing parameters
  size_t max_compressed_size_per_image_ = 0;  // Worst-case compressed size
  size_t max_compressed_batch_size_ = 0;       // Worst-case for full batch

  // ============================================================================
  // Helper Methods
  // ============================================================================

  /**
   * Allocate all GPU buffers and resources
   */
  void allocate_gpu_resources();

  /**
   * Free all GPU buffers and resources
   */
  void free_gpu_resources();

  /**
   * Decompress one batch into the current slice of `pool` and emit the TensorMap.
   * Templated on the pixel type; the pools are compile-time typed while the dtype is
   * chosen at runtime, so compute() dispatches here per configured dtype.
   */
  template <typename T>
  bool process_batch(matx::tensor_t<T, 4>& pool,
                     const std::vector<const uint8_t*>& compressed_ptrs,
                     const std::vector<size_t>& compressed_sizes,
                     const std::vector<uint64_t>& image_ids,
                     size_t num_images,
                     uint64_t batch_id,
                     OutputContext& op_output);

  /**
   * Perform GPU decompression pipeline for a batch
   */
  bool decompress_batch_gpu(const std::vector<const uint8_t*>& compressed_ptrs,
                            const std::vector<size_t>& compressed_sizes,
                            size_t num_images,
                            uint8_t* output_buffer);
};

}  // namespace holoscan::ops

#endif  // DECOMPRESS_GPU_OPERATOR_HPP
