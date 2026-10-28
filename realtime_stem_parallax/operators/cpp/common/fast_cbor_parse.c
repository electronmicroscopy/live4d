#include "fast_cbor_parse.h"

#include <string.h>
#include <stdio.h>

// CBOR major types (top 3 bits)
#define CBOR_MAJOR_TYPE_MASK 0xE0
#define CBOR_MAJOR_UNSIGNED  0x00
#define CBOR_MAJOR_NEGATIVE  0x20
#define CBOR_MAJOR_BYTES     0x40
#define CBOR_MAJOR_TEXT      0x60
#define CBOR_MAJOR_ARRAY     0x80
#define CBOR_MAJOR_MAP       0xA0
#define CBOR_MAJOR_TAG       0xC0
#define CBOR_MAJOR_SIMPLE    0xE0

// CBOR additional info (bottom 5 bits)
#define CBOR_ADDL_INFO_MASK  0x1F

// Dectris compression tag: 56500 = 0xDCB4
// Encoded as: 0xD9 (tag with 2-byte uint16) 0xDC 0xB4
static const uint8_t DECTRIS_TAG_PATTERN[3] = {0xD9, 0xDC, 0xB4};

/**
 * Read CBOR unsigned integer (variable length).
 * Returns the value and advances the pointer.
 */
static int read_cbor_uint(const uint8_t** data, const uint8_t* end, uint64_t* value) {
    if (*data >= end) return -1;

    uint8_t initial_byte = **data;
    uint8_t additional_info = initial_byte & CBOR_ADDL_INFO_MASK;
    (*data)++;

    if (additional_info < 24) {
        *value = additional_info;
        return 0;
    } else if (additional_info == 24) {
        // 1-byte uint8
        if (*data >= end) return -1;
        *value = **data;
        (*data)++;
        return 0;
    } else if (additional_info == 25) {
        // 2-byte uint16
        if (*data + 2 > end) return -1;
        *value = ((uint64_t)(*data)[0] << 8) | (*data)[1];
        (*data) += 2;
        return 0;
    } else if (additional_info == 26) {
        // 4-byte uint32
        if (*data + 4 > end) return -1;
        *value = ((uint64_t)(*data)[0] << 24) | ((uint64_t)(*data)[1] << 16) |
                 ((uint64_t)(*data)[2] << 8) | (*data)[3];
        (*data) += 4;
        return 0;
    } else if (additional_info == 27) {
        // 8-byte uint64
        if (*data + 8 > end) return -1;
        *value = ((uint64_t)(*data)[0] << 56) | ((uint64_t)(*data)[1] << 48) |
                 ((uint64_t)(*data)[2] << 40) | ((uint64_t)(*data)[3] << 32) |
                 ((uint64_t)(*data)[4] << 24) | ((uint64_t)(*data)[5] << 16) |
                 ((uint64_t)(*data)[6] << 8) | (*data)[7];
        (*data) += 8;
        return 0;
    }

    return -1;
}

/**
 * Read CBOR byte string length and get pointer to data.
 */
static int read_cbor_bytes(const uint8_t** data, const uint8_t* end,
                          const uint8_t** bytes_out, size_t* len_out) {
    if (*data >= end) return -1;

    uint8_t initial_byte = **data;
    uint8_t major_type = initial_byte & CBOR_MAJOR_TYPE_MASK;

    if (major_type != CBOR_MAJOR_BYTES) {
        fprintf(stderr, "Expected byte string, got major type 0x%02x\n", major_type);
        return -1;
    }

    uint64_t length;
    if (read_cbor_uint(data, end, &length) != 0) {
        return -1;
    }

    if (*data + length > end) {
        fprintf(stderr, "Byte string extends beyond message: %lu + %lu > %lu\n",
                (unsigned long)(*data - end + (*data - end)),
                (unsigned long)length,
                (unsigned long)0);
        return -1;
    }

    *bytes_out = *data;
    *len_out = (size_t)length;
    (*data) += length;

    return 0;
}

/**
 * Fast detection of Dectris ZMQ stream message type.
 * Searches for the CBOR "type" field and returns START, IMAGE, END, or UNKNOWN.
 *
 * Strategy:
 * 1. Validate CBOR map structure (first byte should be 0xAx)
 * 2. Search for CBOR text string "type": 0x64 + "type"
 * 3. Extract and compare the value ("start", "image", "end")
 * 4. Limit search to first 1KB to avoid scanning large image data
 */
zmq_message_type_t fast_detect_message_type(
    const uint8_t* msg_data,
    size_t msg_size)
{
    // Early validation with minimal checks
    if (msg_size < 10) {
        return ZMQ_MSG_UNKNOWN;
    }

    // Skip any leading CBOR tags before the map. Real DECTRIS Stream2 messages
    // arrive wrapped in the self-describe tag 55799 (bytes d9 d9 f7); simulator
    // messages may carry the bare map.
    const uint8_t* head = msg_data;
    const uint8_t* head_end = msg_data + msg_size;
    while (head < head_end && (*head & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_TAG) {
        uint64_t tag_value;
        if (read_cbor_uint(&head, head_end, &tag_value) != 0) {
            return ZMQ_MSG_UNKNOWN;
        }
    }

    // Quick CBOR map validation
    if (head >= head_end || ((*head & 0xe0) != 0xa0)) {
        return ZMQ_MSG_UNKNOWN;
    }

    // Optimized search strategy: look for "type" field in CBOR format
    // CBOR text string "type": 0x64 (major type 3, length 4) + "type" bytes
    const uint8_t type_cbor[] = {0x64, 0x74, 0x79, 0x70, 0x65};
    const size_t type_cbor_len = 5;

    // Limit search range to avoid scanning entire large messages
    // Most CBOR messages have "type" field near the beginning. The bounds checks below keep
    // every read inside the message, so the whole window can be searched (a small `end`
    // message with "type" as a later key is still recognized).
    size_t search_limit = (msg_size < 1024) ? msg_size : 1024;

    // Fast linear search with early termination
    for (size_t i = 0; i < search_limit; i++) {
        // Use direct byte comparison for maximum speed
        if (msg_data[i] == type_cbor[0] &&
            i + type_cbor_len < msg_size &&
            memcmp(msg_data + i, type_cbor, type_cbor_len) == 0) {

            // Found "type" field, extract value
            size_t value_start = i + type_cbor_len;

            if (value_start >= msg_size) break;

            uint8_t value_type = msg_data[value_start];

            // Check if it's a text string (major type 3)
            if ((value_type & 0xe0) == 0x60) {
                uint8_t length = value_type & 0x1f;
                value_start++;

                // Handle only simple length encodings (0-23)
                if (length <= 0x17 && value_start + length <= msg_size) {
                    // Direct comparison for known message types
                    if (length == 5) {
                        if (memcmp(msg_data + value_start, "start", 5) == 0) {
                            return ZMQ_MSG_START;
                        }
                        if (memcmp(msg_data + value_start, "image", 5) == 0) {
                            return ZMQ_MSG_IMAGE;
                        }
                    } else if (length == 3) {
                        if (memcmp(msg_data + value_start, "end", 3) == 0) {
                            return ZMQ_MSG_END;
                        }
                    }
                }
            }
            break; // Found "type" field, no need to continue searching
        }
    }

    return ZMQ_MSG_UNKNOWN;
}

/**
 * Extract an unsigned-integer field from a CBOR map by key.
 * Scans the message for the text-string key (keys of 1-23 characters) and reads its value.
 */
int fast_extract_uint_field(const uint8_t* msg_data, size_t msg_size, const char* key,
                            uint64_t* value_out) {
    const uint8_t* end = msg_data + msg_size;
    const uint8_t* ptr = msg_data;

    const size_t key_len = strlen(key);
    if (key_len == 0 || key_len > 23) return -1;
    const uint8_t key_head = (uint8_t)(CBOR_MAJOR_TEXT | key_len);  // text string, short length

    while (ptr + key_len + 1 < end) {
        if (*ptr == key_head && memcmp(ptr + 1, key, key_len) == 0) {
            // Found the key, next should be the value
            ptr += 1 + key_len;

            // Read the unsigned integer value
            if ((*ptr & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_UNSIGNED) {
                uint64_t value;
                if (read_cbor_uint(&ptr, end, &value) == 0) {
                    *value_out = value;
                    return 0;  // Success!
                }
            }
            return -1;
        }
        ptr++;
    }

    return -1;  // Not found
}

/**
 * Parse the content of a Dectris compression tag (56500), starting right after the tag bytes:
 * the array [algorithm "bslz4", elem_size, compressed_bytes]. Returns 0 and fills the outputs,
 * or -1 with the reason in *why.
 */
static int parse_dectris_payload(
    const uint8_t* ptr,
    const uint8_t* end,
    const uint8_t** compressed_out,
    size_t* compressed_size_out,
    size_t* original_size_out,
    size_t* elem_size_out,
    const char** why)
{
    // An array with 3 elements: [algorithm, elem_size, bytes]
    if (ptr >= end) {
        *why = "unexpected end after the Dectris tag";
        return -1;
    }
    uint8_t array_byte = *ptr;
    if ((array_byte & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_ARRAY ||
        (array_byte & CBOR_ADDL_INFO_MASK) != 3) {
        *why = "no 3-element array after the Dectris tag";
        return -1;
    }
    ptr++;

    // Element 0: algorithm (text string, must be "bslz4")
    if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_TEXT) {
        *why = "no algorithm string after the Dectris tag";
        return -1;
    }
    uint64_t algo_len;
    if (read_cbor_uint(&ptr, end, &algo_len) != 0 || ptr + algo_len > end) {
        *why = "malformed algorithm string";
        return -1;
    }
    if (algo_len != 5 || memcmp(ptr, "bslz4", 5) != 0) {
        *why = "the compression is not bslz4";
        return -1;
    }
    ptr += algo_len;

    // Element 1: elem_size (unsigned integer)
    if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED) {
        *why = "no element size after the algorithm";
        return -1;
    }
    uint64_t elem_size;
    if (read_cbor_uint(&ptr, end, &elem_size) != 0) {
        *why = "malformed element size";
        return -1;
    }

    // Element 2: compressed_bytes (byte string)
    const uint8_t* compressed_bytes;
    size_t compressed_len;
    if (read_cbor_bytes(&ptr, end, &compressed_bytes, &compressed_len) != 0) {
        *why = "malformed compressed byte string";
        return -1;
    }

    // Dectris header (first 12 bytes): original_size (big-endian uint64), block_size (uint32)
    if (compressed_len < 12) {
        *why = "compressed data too short for the Dectris header";
        return -1;
    }
    uint64_t orig_size =
        ((uint64_t)compressed_bytes[0] << 56) |
        ((uint64_t)compressed_bytes[1] << 48) |
        ((uint64_t)compressed_bytes[2] << 40) |
        ((uint64_t)compressed_bytes[3] << 32) |
        ((uint64_t)compressed_bytes[4] << 24) |
        ((uint64_t)compressed_bytes[5] << 16) |
        ((uint64_t)compressed_bytes[6] << 8) |
        ((uint64_t)compressed_bytes[7]);

    *elem_size_out = (size_t)elem_size;
    *original_size_out = (size_t)orig_size;
    // The full compressed data, including the 12-byte header
    *compressed_out = compressed_bytes;
    *compressed_size_out = compressed_len;
    return 0;
}

/**
 * Fast path: Find Dectris compression tag (56500) and extract compressed data.
 *
 * Strategy:
 * 1. Extract image_id from the CBOR map
 * 2. Scan for the 3-byte pattern 0xD9 0xDC 0xB4 (Dectris tag 56500)
 * 3. Parse the following array: [algorithm, elem_size, compressed_bytes]
 * 4. Extract the compressed bytes directly
 *
 * The three tag bytes can also occur inside other values of the message (the DCU's timestamps
 * hit them about once per million frames), so a match whose content is not a valid payload is
 * skipped and the scan goes on.
 */
int fast_extract_compressed_bslz4(
    const uint8_t* msg_data,
    size_t msg_size,
    const uint8_t** compressed_out,
    size_t* compressed_size_out,
    size_t* original_size_out,
    size_t* elem_size_out,
    uint64_t* image_id_out)
{
    const uint8_t* end = msg_data + msg_size;
    const uint8_t* ptr = msg_data;

    // Extract image_id if requested. Without it the frame cannot be placed in the scan, so
    // the message is rejected (it used to land on scan position 0).
    if (image_id_out != NULL) {
        if (fast_extract_uint_field(msg_data, msg_size, "image_id", image_id_out) != 0) {
            fprintf(stderr, "image_id not found in image message\n");
            return -1;
        }
    }

    // Scan for the Dectris compression tag pattern; the first match with a valid payload wins.
    const char* why = "Dectris compression tag (56500) not found in message";
    while (ptr + sizeof(DECTRIS_TAG_PATTERN) <= end) {
        if (memcmp(ptr, DECTRIS_TAG_PATTERN, sizeof(DECTRIS_TAG_PATTERN)) == 0 &&
            parse_dectris_payload(ptr + sizeof(DECTRIS_TAG_PATTERN), end, compressed_out,
                                  compressed_size_out, original_size_out, elem_size_out,
                                  &why) == 0) {
            return 0;
        }
        ptr++;
    }

    fprintf(stderr, "No bslz4 payload in image message: %s\n", why);
    return -1;
}

/**
 * Fast path: Find the multi-dim array tag (40) and extract the image dimensions.
 *
 * Strategy: scan for the canonical tag-40 encoding (0xD8 0x28) and verify the structure
 * that must follow — array(2) of [dims, typed-array], with dims itself an array(2) of
 * unsigned integers — so an accidental 0xD8 0x28 in unrelated bytes is skipped rather
 * than misparsed.
 */
int fast_extract_image_dims(
    const uint8_t* msg_data,
    size_t msg_size,
    uint64_t* dim0_out,
    uint64_t* dim1_out)
{
    const uint8_t* end = msg_data + msg_size;
    const uint8_t* ptr = msg_data;

    while (ptr + 2 <= end) {
        if (ptr[0] == 0xD8 && ptr[1] == 0x28) {
            const uint8_t* p = ptr + 2;
            /* Tag content: array(2) = [dims, typed-array] */
            if (p < end && *p == 0x82) {
                p++;
                /* dims: array(2) of unsigned ints (a detector frame is 2-D) */
                if (p < end && (*p & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_ARRAY &&
                    (*p & CBOR_ADDL_INFO_MASK) == 2) {
                    p++;
                    uint64_t dim0, dim1;
                    if (p < end && (*p & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_UNSIGNED &&
                        read_cbor_uint(&p, end, &dim0) == 0 &&
                        p < end && (*p & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_UNSIGNED &&
                        read_cbor_uint(&p, end, &dim1) == 0) {
                        *dim0_out = dim0;
                        *dim1_out = dim1;
                        return 0;
                    }
                }
            }
        }
        ptr++;
    }

    fprintf(stderr, "Multi-dim array tag (40) not found in message\n");
    return -1;
}
