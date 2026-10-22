#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * Message type enumeration for Dectris ZMQ stream messages.
 * Used for fast detection of message type from CBOR "type" field.
 */
typedef enum {
    ZMQ_MSG_START = 0,
    ZMQ_MSG_IMAGE = 1,
    ZMQ_MSG_END = 2,
    ZMQ_MSG_UNKNOWN = 3
} zmq_message_type_t;

/**
 * Fast-path CBOR parser for Dectris IMAGE messages.
 *
 * This parser exploits the highly regular structure of Dectris detector
 * messages to quickly locate compressed BSLZ4 data without full CBOR parsing.
 *
 * Expected structure (from simulate_zmq_stream.py):
 *   IMAGE = {
 *     "data": {
 *       "threshold_1": CBORTag(40, [shape, CBORTag(69, CBORTag(56500, ["bslz4", elem_size, bytes]))])
 *     }
 *   }
 *
 * CBORTag 56500 is the Dectris compression tag defined at:
 * https://github.com/dectris/documentation/blob/main/cbor/dectris-compression-tag.md
 */

/**
 * Fast detection of Dectris ZMQ stream message type.
 *
 * Searches for the CBOR "type" field and extracts its value without full parsing.
 * Optimized for speed with early termination and minimal memory access.
 *
 * @param msg_data  CBOR-encoded message
 * @param msg_size  Size of message in bytes
 * @return Message type enum (START, IMAGE, END, or UNKNOWN)
 */
zmq_message_type_t fast_detect_message_type(
    const uint8_t* msg_data,
    size_t msg_size);

// Read the unsigned series_id from a tagged or bare Stream2 map.
int fast_extract_series_id(const uint8_t* data, size_t size, uint64_t* series_id);

/**
 * Fast extraction of compressed BSLZ4 data from Dectris IMAGE message.
 *
 * This function directly locates the compressed data by searching for the
 * Dectris compression tag (56500) without full CBOR traversal.
 * Also extracts the image_id field from the CBOR map.
 *
 * @param msg_data        CBOR-encoded IMAGE message
 * @param msg_size        Size of message in bytes
 * @param compressed_out  Output: Pointer to compressed data (within msg_data)
 * @param compressed_size_out Output: Size of compressed data
 * @param original_size_out   Output: Original uncompressed size
 * @param elem_size_out       Output: Element size (2 for uint16)
 * @param image_id_out        Output: Image ID from CBOR (can be NULL if not needed)
 * @return 0 on success, -1 on error
 */
int fast_extract_compressed_bslz4(
    const uint8_t* msg_data,
    size_t msg_size,
    const uint8_t** compressed_out,
    size_t* compressed_size_out,
    size_t* original_size_out,
    size_t* elem_size_out,
    uint64_t* image_id_out);

#if defined(__cplusplus)
}
#endif
