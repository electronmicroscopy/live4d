/**
 * @file bitshuffle_kernel.cu
 * @brief CUDA kernel implementation for bitshuffle unshuffle
 */

#include "bitshuffle_kernel_gpu.cuh"

namespace bitshuffle {

/**
 * @brief Batch unshuffle kernel - each block processes elements from one image
 */
__global__ void bitunshuffle_batch_kernel(
    uint8_t** __restrict__ input_ptrs,
    uint8_t** __restrict__ output_ptrs,
    size_t num_elements_per_image,
    size_t element_size,
    size_t block_size_elements,
    size_t batch_size
) {
    // Determine which image this block is working on
    size_t image_idx = blockIdx.x / ((num_elements_per_image + blockDim.x - 1) / blockDim.x);
    if (image_idx >= batch_size) return;

    // Determine which element within the image
    size_t block_within_image = blockIdx.x % ((num_elements_per_image + blockDim.x - 1) / blockDim.x);
    size_t elem_idx = block_within_image * blockDim.x + threadIdx.x;

    if (elem_idx >= num_elements_per_image) return;

    const uint8_t* input = input_ptrs[image_idx];
    uint8_t* output = output_ptrs[image_idx];

    // Bitshuffle block-aware logic (same as single-image kernel). The bslz4 block size is
    // constant in *bytes* (nominally 8192), so the element count per block depends on the
    // element size and comes in as a parameter.
    const size_t BLOCK_SIZE_ELEMENTS = block_size_elements;
    const size_t BLOCK_SIZE_BYTES = BLOCK_SIZE_ELEMENTS * element_size;

    const size_t block_idx = elem_idx / BLOCK_SIZE_ELEMENTS;
    const size_t elem_within_block = elem_idx % BLOCK_SIZE_ELEMENTS;
    const size_t block_offset_bytes = block_idx * BLOCK_SIZE_BYTES;

    size_t block_num_elements = BLOCK_SIZE_ELEMENTS;
    if (block_idx == num_elements_per_image / BLOCK_SIZE_ELEMENTS) {
        block_num_elements = num_elements_per_image % BLOCK_SIZE_ELEMENTS;
        if (block_num_elements == 0) block_num_elements = BLOCK_SIZE_ELEMENTS;
    }

    const size_t bytes_per_plane = block_num_elements / 8;

    // Process each byte of the element
    for (size_t byte_in_elem = 0; byte_in_elem < element_size; byte_in_elem++) {
        uint8_t result_byte = 0;

        for (size_t bit = 0; bit < 8; bit++) {
            const size_t bit_plane = byte_in_elem * 8 + bit;
            const size_t bit_plane_start_byte = bit_plane * bytes_per_plane;
            const size_t byte_within_plane = elem_within_block / 8;
            const size_t bit_offset = elem_within_block % 8;

            const size_t byte_pos = block_offset_bytes + bit_plane_start_byte + byte_within_plane;

            const uint8_t bit_value = (input[byte_pos] >> bit_offset) & 1;
            result_byte |= (bit_value << bit);
        }

        output[elem_idx * element_size + byte_in_elem] = result_byte;
    }
}

cudaError_t unshuffle_batch(
    uint8_t** d_input_ptrs,
    uint8_t** d_output_ptrs,
    size_t num_elements,
    size_t element_size,
    size_t block_size_elements,
    size_t batch_size,
    cudaStream_t stream
) {
    if (num_elements == 0 || batch_size == 0) {
        return cudaSuccess;
    }

    // Bitshuffle operates on bit planes of 8-element groups, both for full blocks and
    // for the final short block (num_elements % block_size_elements).
    if (num_elements % 8 != 0 || block_size_elements % 8 != 0 || block_size_elements == 0) {
        return cudaErrorInvalidValue;
    }

    // Launch enough blocks to cover all images and all elements
    const int threads_per_block = 256;
    int blocks_per_image = (num_elements + threads_per_block - 1) / threads_per_block;
    int total_blocks = blocks_per_image * batch_size;

    bitunshuffle_batch_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
        d_input_ptrs, d_output_ptrs, num_elements, element_size, block_size_elements,
        batch_size
    );

    return cudaGetLastError();
}

} // namespace bitshuffle
