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
 * This kernel performs the inverse operation (unshuffle) on GPU.
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
 * @param element_size Size of each element in bytes
 * @param block_size_elements Elements per bslz4 block = block size in bytes (from the
 *                            Dectris header, nominally 8192) / element_size
 * @param batch_size Number of images
 * @param stream CUDA stream
 * @return cudaError_t
 */
cudaError_t unshuffle_batch(
    uint8_t** d_input_ptrs,
    uint8_t** d_output_ptrs,
    size_t num_elements,
    size_t element_size,
    size_t block_size_elements,
    size_t batch_size,
    cudaStream_t stream = 0
);

} // namespace bitshuffle
