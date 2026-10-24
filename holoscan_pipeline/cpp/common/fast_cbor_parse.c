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

int fast_extract_start_metadata(const uint8_t* data, size_t size,
                                dectris_start_metadata_t* metadata) {
    if (data == NULL || metadata == NULL || size == 0 ||
        fast_detect_message_type(data, size) != ZMQ_MSG_START) return -1;

    const uint8_t* ptr = data;
    const uint8_t* end = data + size;
    uint64_t n;
    while (ptr < end && (*ptr & CBOR_MAJOR_TYPE_MASK) == CBOR_MAJOR_TAG)
        if (read_cbor_uint(&ptr, end, &n)) return -1;
    if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_MAP ||
        read_cbor_uint(&ptr, end, &n))
        return -1;

    unsigned found = 0;
    for (uint64_t i = 0; i < n; ++i) {
        if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_TEXT) return -1;
        uint64_t key_size;
        if (read_cbor_uint(&ptr, end, &key_size) || key_size > (uint64_t)(end - ptr)) return -1;
        const uint8_t* key = ptr;
        ptr += key_size;

        if (key_size == 9 && memcmp(key, "series_id", 9) == 0) {
            if ((found & 1) || ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED ||
                read_cbor_uint(&ptr, end, &metadata->series_id)) return -1;
            found |= 1;
        } else if (key_size == 12 && memcmp(key, "image_size_x", 12) == 0) {
            uint64_t value;
            if ((found & 2) || ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED ||
                read_cbor_uint(&ptr, end, &value) || value > UINT32_MAX) return -1;
            metadata->image_size_x = (uint32_t)value;
            found |= 2;
        } else if (key_size == 12 && memcmp(key, "image_size_y", 12) == 0) {
            uint64_t value;
            if ((found & 4) || ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED ||
                read_cbor_uint(&ptr, end, &value) || value > UINT32_MAX) return -1;
            metadata->image_size_y = (uint32_t)value;
            found |= 4;
        } else if (key_size == 11 && memcmp(key, "image_dtype", 11) == 0) {
            if ((found & 8) || ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_TEXT) return -1;
            uint64_t value_size;
            if (read_cbor_uint(&ptr, end, &value_size) || value_size > (uint64_t)(end - ptr) ||
                value_size != 6 || memcmp(ptr, "uint16", 6) != 0) return -1;
            ptr += value_size;
            found |= 8;
        } else if (skip_cbor(&ptr, end, 0)) {
            return -1;
        }
    }

    if (ptr != end || found != 15) return -1;
    const uint32_t dim = metadata->image_size_x;
    if (metadata->image_size_y != dim || (dim != 96 && dim != 192)) return -1;
    return 0;
}

/* Locate a map member structurally. Integer/byte contents may contain tag bytes. */
static int map_value(const uint8_t* ptr, const uint8_t* end, const char* key,
                     const uint8_t** value, const uint8_t** value_end) {
    uint64_t n;
    while (ptr < end && (*ptr >> 5) == 6)
        if (read_cbor_uint(&ptr, end, &n)) return -1;
    if (ptr >= end || (*ptr >> 5) != 5 || read_cbor_uint(&ptr, end, &n)) return -1;
    const size_t wanted = strlen(key);
    while (n--) {
        uint64_t length;
        if (ptr >= end || (*ptr >> 5) != 3 || read_cbor_uint(&ptr, end, &length) ||
            length > (uint64_t)(end - ptr)) return -1;
        const int matches = length == wanted && !memcmp(ptr, key, wanted);
        ptr += length;
        const uint8_t* start = ptr;
        if (skip_cbor(&ptr, end, 0)) return -1;
        if (matches) { *value = start; *value_end = ptr; return 0; }
    }
    return -1;
}

static int expect_cbor(const uint8_t** ptr, const uint8_t* end,
                       uint8_t major, uint64_t expected) {
    uint64_t value;
    return *ptr >= end || (**ptr >> 5) != major ||
           read_cbor_uint(ptr, end, &value) || value != expected ? -1 : 0;
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

    if (length > (uint64_t)(end - *data)) {
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

/* Parse the supported channel envelope without scanning image bytes for tags.
 * In particular, image_id=55336 encodes D8 28 inside an integer, not a tensor tag.
 */
int fast_extract_compressed_bslz4(
    const uint8_t* msg_data, size_t msg_size,
    const uint8_t** compressed_out, size_t* compressed_size_out,
    size_t* original_size_out, size_t* elem_size_out,
    uint32_t* image_size_x_out, uint32_t* image_size_y_out,
    uint64_t* image_id_out)
{
    const uint8_t *ptr, *end, *channels, *channels_end;
    uint64_t value, height, width;
    if (!msg_data || !msg_size || !compressed_out || !compressed_size_out ||
        !original_size_out || !elem_size_out || !image_size_x_out || !image_size_y_out)
        return -1;
    if (map_value(msg_data, msg_data + msg_size, "image_id", &ptr, &end) ||
        ptr >= end || (*ptr >> 5) != 0 || read_cbor_uint(&ptr, end, &value) || ptr != end)
        return -1;
    if (image_id_out) *image_id_out = value;
    if (map_value(msg_data, msg_data + msg_size, "data", &channels, &channels_end) ||
        map_value(channels, channels_end, "threshold_1", &ptr, &end) ||
        expect_cbor(&ptr, end, 6, 40) || expect_cbor(&ptr, end, 4, 2) ||
        expect_cbor(&ptr, end, 4, 2) ||
        ptr >= end || (*ptr >> 5) != 0 || read_cbor_uint(&ptr, end, &height) ||
        ptr >= end || (*ptr >> 5) != 0 || read_cbor_uint(&ptr, end, &width) ||
        height > UINT32_MAX || width > UINT32_MAX ||
        expect_cbor(&ptr, end, 6, 69) || expect_cbor(&ptr, end, 6, 56500) ||
        expect_cbor(&ptr, end, 4, 3) || expect_cbor(&ptr, end, 3, 5) ||
        end - ptr < 5 || memcmp(ptr, "bslz4", 5))
        return -1;
    ptr += 5;
    if (ptr >= end || (*ptr >> 5) != 0 || read_cbor_uint(&ptr, end, &value) ||
        value > SIZE_MAX) return -1;
    *elem_size_out = (size_t)value;
    if (read_cbor_bytes(&ptr, end, compressed_out, compressed_size_out) ||
        ptr != end || *compressed_size_out < 12) return -1;
    const uint8_t* bytes = *compressed_out;
    uint64_t original = 0;
    for (unsigned i = 0; i < 8; ++i) original = (original << 8) | bytes[i];
    if (original > SIZE_MAX) return -1;
    *original_size_out = (size_t)original;
    *image_size_x_out = (uint32_t)width;
    *image_size_y_out = (uint32_t)height;
    return 0;
}
