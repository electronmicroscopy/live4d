/**
 * @file bitshuffle_kernel.cuh
 * @brief CUDA kernel for bitshuffle unshuffle operation
 *
 * Bitshuffle is a bit-level rearrangement algorithm that improves compression.
 * It transposes data at the bit level, grouping bits of the same significance
 * across multiple elements together.
 *
 * For example, with 4 uint16_t values:
 *   Before: [AAAA BBBB CCCC DDDD] (each letter represents a bit from one value)
 *   After:  [A0A1A2A3 B0B1B2B3 C0C1C2C3 D0D1D2D3] (bits of same position grouped)
 *
 * unshuffle_batch performs the inverse operation on GPU; shuffle_batch performs the
 * forward operation for compression. Both use bitshuffle's 4096-element blocks.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace bitshuffle {

/**
 * @brief Batch unshuffle - process multiple images in parallel
 *
 * @param d_input_ptrs Array of input pointers (one per image)
 * @param d_output_ptrs Array of output pointers (one per image)
 * @param num_elements Number of elements per image
 * @param element_size Size of each element
 * @param batch_size Number of images
 * @param stream CUDA stream
 * @return cudaError_t
 */
cudaError_t unshuffle_batch(
    uint8_t** d_input_ptrs,
    uint8_t** d_output_ptrs,
    size_t num_elements,
    size_t element_size,
    size_t batch_size,
    cudaStream_t stream = 0
);

/**
 * @brief Batch shuffle of contiguous images into contiguous bit-plane blocks
 *
 * Image i occupies bytes [i * num_elements * element_size, (i + 1) * ...) of both buffers.
 * The output matches bitshuffle's bshuf_trans_bit_elem applied per 4096-element block,
 * with a final partial block; num_elements must be a multiple of 8.
 */
cudaError_t shuffle_batch(
    const uint8_t* d_input,
    uint8_t* d_output,
    size_t num_elements,
    size_t element_size,
    size_t batch_size,
    cudaStream_t stream = 0
);

} // namespace bitshuffle
