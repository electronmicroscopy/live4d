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

static int skip_cbor(const uint8_t** ptr, const uint8_t* end, unsigned depth) {
    if (*ptr >= end || depth > 32) return -1;
    const uint8_t major = **ptr >> 5;
    uint64_t n;
    if (read_cbor_uint(ptr, end, &n)) return -1;
    if (major == 2 || major == 3) {
        if (n > (uint64_t)(end - *ptr)) return -1;
        *ptr += n;
    } else if (major == 4 || major == 5 || major == 6) {
        if (major == 5) {
            if (n > (uint64_t)(end - *ptr) / 2) return -1;
            n *= 2;
        }
        if (major == 6) n = 1;
        if (n > (uint64_t)(end - *ptr)) return -1;
        while (n--) if (skip_cbor(ptr, end, depth + 1)) return -1;
    }
    return 0;
}

int fast_extract_series_id(const uint8_t* data, size_t size, uint64_t* series_id) {
    const uint8_t* ptr = data;
    const uint8_t* end = data + size;
    uint64_t n;
    while (ptr < end && (*ptr >> 5) == 6)
        if (read_cbor_uint(&ptr, end, &n)) return -1;
    if (ptr >= end || (*ptr >> 5) != 5 || read_cbor_uint(&ptr, end, &n)) return -1;
    while (n--) {
        uint64_t len;
        if (ptr >= end || (*ptr >> 5) != 3 || read_cbor_uint(&ptr, end, &len)
            || len > (uint64_t)(end - ptr)) return -1;
        const int matches = len == 9 && memcmp(ptr, "series_id", 9) == 0;
        ptr += len;
        if (matches) {
            if (ptr >= end || (*ptr >> 5) != 0) return -1;
            return read_cbor_uint(&ptr, end, series_id);
        }
        if (skip_cbor(&ptr, end, 0)) return -1;
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
    if (head >= head_end || ((*head & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_MAP)) {
        return ZMQ_MSG_UNKNOWN;
    }

    // Optimized search strategy: look for "type" field in CBOR format
    // CBOR text string "type": 0x64 (major type 3, length 4) + "type" bytes
    const uint8_t type_cbor[] = {0x64, 0x74, 0x79, 0x70, 0x65};
    const size_t type_cbor_len = 5;

    // Limit search range to avoid scanning entire large messages
    // Most CBOR messages have "type" field near the beginning
    size_t search_limit = (msg_size < 1024) ? msg_size : 1024;
    if (search_limit > type_cbor_len + 20) {
        search_limit -= type_cbor_len + 20;
    }

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
 * Extract image_id from CBOR map.
 * Scans the message for "image_id" key and extracts its value.
 */
static int extract_image_id(const uint8_t* msg_data, size_t msg_size, uint64_t* image_id_out) {
    const uint8_t* end = msg_data + msg_size;
    const uint8_t* ptr = msg_data;

    // Look for the text string "image_id" (9 bytes: 0x68 + "image_id")
    const char* image_id_key = "image_id";
    const size_t key_len = 8;

    while (ptr + key_len + 1 < end) {
        // Check if this is a text string with length 8
        if (*ptr == 0x68) {  // CBOR text string, length 8
            if (memcmp(ptr + 1, image_id_key, key_len) == 0) {
                // Found "image_id" key, next should be the value
                ptr += 1 + key_len;

                if (ptr >= end) return -1;

                // Read the unsigned integer value
                if ((*ptr & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_UNSIGNED) {
                    uint64_t value;
                    if (read_cbor_uint(&ptr, end, &value) == 0) {
                        *image_id_out = value;
                        return 0;  // Success!
                    }
                }
                return -1;
            }
        }
        ptr++;
    }

    return -1;  // Not found
}

/**
 * Fast path: Find Dectris compression tag (56500) and extract compressed data.
 *
 * Strategy:
 * 1. Extract image_id from the CBOR map
 * 2. Scan for the 3-byte pattern 0xD9 0xDC 0xB4 (Dectris tag 56500)
 * 3. Parse the following array: [algorithm, elem_size, compressed_bytes]
 * 4. Extract the compressed bytes directly
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

    // Extract image_id if requested
    if (image_id_out != NULL) {
        if (extract_image_id(msg_data, msg_size, image_id_out) != 0) {
            // If image_id extraction fails, set to 0 and continue
            // (compressed data extraction is more critical)
            *image_id_out = 0;
        }
    }

    // Scan for Dectris compression tag pattern
    while (ptr + sizeof(DECTRIS_TAG_PATTERN) <= end) {
        if (memcmp(ptr, DECTRIS_TAG_PATTERN, sizeof(DECTRIS_TAG_PATTERN)) == 0) {
            // Found tag 56500!
            ptr += sizeof(DECTRIS_TAG_PATTERN);

            // Next should be an array with 3 elements: [algorithm, elem_size, bytes]
            if (ptr >= end) {
                fprintf(stderr, "Unexpected end after Dectris tag\n");
                return -1;
            }

            uint8_t array_byte = *ptr;
            if ((array_byte & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_ARRAY) {
                fprintf(stderr, "Expected array after Dectris tag, got 0x%02x\n", array_byte);
                return -1;
            }

            uint8_t array_length = array_byte & CBOR_ADDL_INFO_MASK;
            if (array_length != 3) {
                fprintf(stderr, "Expected array of length 3, got %d\n", array_length);
                return -1;
            }
            ptr++;

            // Element 0: algorithm (text string, should be "bslz4")
            if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_TEXT) {
                fprintf(stderr, "Expected text string for algorithm\n");
                return -1;
            }

            uint64_t algo_len;
            if (read_cbor_uint(&ptr, end, &algo_len) != 0) {
                fprintf(stderr, "Failed to read algorithm string length\n");
                return -1;
            }

            if (ptr + algo_len > end) {
                fprintf(stderr, "Algorithm string extends beyond message\n");
                return -1;
            }

            // Verify it's "bslz4"
            if (algo_len != 5 || memcmp(ptr, "bslz4", 5) != 0) {
                fprintf(stderr, "Expected 'bslz4' algorithm, got something else\n");
                return -1;
            }
            ptr += algo_len;

            // Element 1: elem_size (unsigned integer)
            if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED) {
                fprintf(stderr, "Expected unsigned int for elem_size\n");
                return -1;
            }

            uint64_t elem_size;
            if (read_cbor_uint(&ptr, end, &elem_size) != 0) {
                fprintf(stderr, "Failed to read elem_size\n");
                return -1;
            }
            *elem_size_out = (size_t)elem_size;

            // Element 2: compressed_bytes (byte string)
            const uint8_t* compressed_bytes;
            size_t compressed_len;
            if (read_cbor_bytes(&ptr, end, &compressed_bytes, &compressed_len) != 0) {
                fprintf(stderr, "Failed to read compressed bytes\n");
                return -1;
            }

            // Extract Dectris header (first 12 bytes):
            // - 8 bytes: original_size (big-endian uint64)
            // - 4 bytes: block_size (big-endian uint32)
            if (compressed_len < 12) {
                fprintf(stderr, "Compressed data too short for Dectris header: %zu bytes\n", compressed_len);
                return -1;
            }

            // Read original size from Dectris header
            uint64_t orig_size =
                ((uint64_t)compressed_bytes[0] << 56) |
                ((uint64_t)compressed_bytes[1] << 48) |
                ((uint64_t)compressed_bytes[2] << 40) |
                ((uint64_t)compressed_bytes[3] << 32) |
                ((uint64_t)compressed_bytes[4] << 24) |
                ((uint64_t)compressed_bytes[5] << 16) |
                ((uint64_t)compressed_bytes[6] << 8) |
                ((uint64_t)compressed_bytes[7]);

            *original_size_out = (size_t)orig_size;

            // Return pointer to the full compressed data (including 12-byte header)
            *compressed_out = compressed_bytes;
            *compressed_size_out = compressed_len;

            return 0;  // Success!
        }

        ptr++;
    }

    fprintf(stderr, "Dectris compression tag (56500) not found in message\n");
    return -1;
}
