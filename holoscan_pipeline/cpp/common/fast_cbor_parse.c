#include "fast_cbor_parse.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct {
    char* ptr;
    size_t left;  // Includes the terminating NUL.
} text_out_t;

static void text_append(text_out_t* out, const char* text, size_t size) {
    if (out->left <= 1) return;
    if (size > out->left - 1) size = out->left - 1;
    memcpy(out->ptr, text, size);
    out->ptr += size;
    out->left -= size;
    *out->ptr = '\0';
}

static void text_printf(text_out_t* out, const char* format, ...) {
    char buffer[64];
    va_list args;
    va_start(args, format);
    const int size = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    if (size > 0) text_append(out, buffer, (size_t)size < sizeof(buffer) ? (size_t)size
                                                                          : sizeof(buffer) - 1);
}

static void text_double(text_out_t* out, double value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.15g", value);
    if (strtod(buffer, NULL) != value) snprintf(buffer, sizeof(buffer), "%.17g", value);
    text_append(out, buffer, strlen(buffer));
}

static double half_to_double(uint16_t half) {
    const int exponent = (half >> 10) & 0x1F;
    const int mantissa = half & 0x3FF;
    const double value = exponent == 0  ? ldexp(mantissa, -24)
                         : exponent == 31 ? (mantissa ? NAN : INFINITY)
                                          : ldexp(mantissa + 1024, exponent - 25);
    return (half & 0x8000) ? -value : value;
}

// Render one definite-length CBOR item in diagnostic notation, truncating long output.
static int render_cbor(const uint8_t** ptr, const uint8_t* end, text_out_t* out, unsigned depth) {
    if (*ptr >= end || depth > 32) return -1;
    const uint8_t major = **ptr >> 5;
    const uint8_t info = **ptr & CBOR_ADDL_INFO_MASK;
    uint64_t n;
    if (read_cbor_uint(ptr, end, &n)) return -1;
    switch (major) {
    case 0:
        text_printf(out, "%llu", (unsigned long long)n);
        return 0;
    case 1:
        if (n < (uint64_t)INT64_MAX) text_printf(out, "%lld", -1LL - (long long)n);
        else text_printf(out, "-1-%llu", (unsigned long long)n);
        return 0;
    case 2:
        if (n > (uint64_t)(end - *ptr)) return -1;
        text_append(out, "h'", 2);
        for (uint64_t i = 0; i < n; ++i) text_printf(out, "%02x", (*ptr)[i]);
        text_append(out, "'", 1);
        *ptr += n;
        return 0;
    case 3:
        if (n > (uint64_t)(end - *ptr)) return -1;
        text_append(out, "\"", 1);
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t c = (*ptr)[i];
            if (c == '"' || c == '\\') text_printf(out, "\\%c", c);
            else if (c < 0x20 || c == 0x7F) text_printf(out, "\\x%02x", c);
            else text_append(out, (const char*)&c, 1);
        }
        text_append(out, "\"", 1);
        *ptr += n;
        return 0;
    case 4:
    case 5:
        text_append(out, major == 4 ? "[" : "{", 1);
        for (uint64_t i = 0; i < n; ++i) {
            if (i) text_append(out, ", ", 2);
            if (render_cbor(ptr, end, out, depth + 1)) return -1;
            if (major == 5) {
                text_append(out, ": ", 2);
                if (render_cbor(ptr, end, out, depth + 1)) return -1;
            }
        }
        text_append(out, major == 4 ? "]" : "}", 1);
        return 0;
    case 6:
        text_printf(out, "%llu(", (unsigned long long)n);
        if (render_cbor(ptr, end, out, depth + 1)) return -1;
        text_append(out, ")", 1);
        return 0;
    default:
        if (info == 20) text_append(out, "false", 5);
        else if (info == 21) text_append(out, "true", 4);
        else if (info == 22) text_append(out, "null", 4);
        else if (info == 23) text_append(out, "undefined", 9);
        else if (info <= 24) text_printf(out, "simple(%llu)", (unsigned long long)n);
        else if (info == 25) text_double(out, half_to_double((uint16_t)n));
        else if (info == 26) {
            const uint32_t bits = (uint32_t)n;
            float value;
            memcpy(&value, &bits, sizeof(value));
            text_double(out, value);
        } else {
            double value;
            memcpy(&value, &n, sizeof(value));
            text_double(out, value);
        }
        return 0;
    }
}

int fast_visit_header_appendix(const uint8_t* data, size_t size,
                               header_appendix_visitor_t visit, void* context) {
    if (data == NULL || visit == NULL) return -1;
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
        const int matches = len == 15 && memcmp(ptr, "header_appendix", 15) == 0;
        ptr += len;
        if (!matches) {
            if (skip_cbor(&ptr, end, 0)) return -1;
            continue;
        }

        char key[256], value[1024];
        text_out_t key_out = {key, sizeof(key)};
        text_out_t value_out = {value, sizeof(value)};
        key[0] = value[0] = '\0';
        if (ptr >= end || (*ptr >> 5) != 5) {
            if (render_cbor(&ptr, end, &value_out, 0)) return -1;
            visit(context, NULL, value);
            return 0;
        }
        uint64_t entries;
        if (read_cbor_uint(&ptr, end, &entries)) return -1;
        while (entries--) {
            key_out = (text_out_t){key, sizeof(key)};
            value_out = (text_out_t){value, sizeof(value)};
            key[0] = value[0] = '\0';
            // Show text keys bare; other key types keep their diagnostic form.
            if (ptr < end && (*ptr >> 5) == 3) {
                if (read_cbor_uint(&ptr, end, &len) || len > (uint64_t)(end - ptr)) return -1;
                text_append(&key_out, (const char*)ptr, (size_t)len);
                ptr += len;
            } else if (render_cbor(&ptr, end, &key_out, 0)) {
                return -1;
            }
            if (render_cbor(&ptr, end, &value_out, 0)) return -1;
            visit(context, key, value);
        }
        return 0;
    }
    return 1;
}

static int extract_image_shape(const uint8_t* data, size_t size,
                               uint32_t* width_out, uint32_t* height_out) {
    const uint8_t* ptr = data;
    const uint8_t* end = data + size;
    while (ptr + 2 <= end) {
        // Tag 40 bytes can also occur inside other values, e.g. image_id 55336 (0x19 0xD8 0x28);
        // a candidate that is not a [[height, width], data] array is skipped.
        if (ptr[0] == 0xD8 && ptr[1] == 0x28) {
            const uint8_t* candidate = ptr + 2;
            uint64_t wrapper_size;
            uint64_t shape_size, width, height;
            if (candidate >= end || (*candidate & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_ARRAY ||
                read_cbor_uint(&candidate, end, &wrapper_size) || wrapper_size != 2 ||
                candidate >= end || (*candidate & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_ARRAY) {
                ++ptr;  // the bytes D8 28 inside an earlier value (e.g. a timestamp), not tag 40
                continue;
            }
            if (read_cbor_uint(&candidate, end, &shape_size) || shape_size != 2 ||
                candidate >= end || (*candidate & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED ||
                read_cbor_uint(&candidate, end, &width) || width > UINT32_MAX ||
                candidate >= end || (*candidate & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED ||
                read_cbor_uint(&candidate, end, &height) || height > UINT32_MAX ||
                skip_cbor(&candidate, end, 0)) {
                ++ptr;
                continue;
            }
            *width_out = (uint32_t)width;
            *height_out = (uint32_t)height;
            return 0;
        }
        ++ptr;
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
    uint32_t* image_size_x_out,
    uint32_t* image_size_y_out,
    uint64_t* image_id_out)
{
    const uint8_t* end = msg_data + msg_size;
    const uint8_t* ptr = msg_data;

    if (image_size_x_out == NULL || image_size_y_out == NULL ||
        extract_image_shape(msg_data, msg_size, image_size_x_out, image_size_y_out) != 0)
        return -1;

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
            // Found tag 56500 (or its bytes inside an earlier value: then keep scanning).
            const uint8_t* resume = ptr + 1;
            ptr += sizeof(DECTRIS_TAG_PATTERN);

            // Next should be an array with 3 elements: [algorithm, elem_size, bytes]
            if (ptr >= end) {
                ptr = resume;  // not the tag
                continue;
            }

            uint8_t array_byte = *ptr;
            if ((array_byte & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_ARRAY) {
                ptr = resume;  // not the tag
                continue;
            }

            uint8_t array_length = array_byte & CBOR_ADDL_INFO_MASK;
            if (array_length != 3) {
                ptr = resume;  // not the tag
                continue;
            }
            ptr++;

            // Element 0: algorithm (text string, should be "bslz4")
            if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_TEXT) {
                ptr = resume;  // not the tag
                continue;
            }

            uint64_t algo_len;
            if (read_cbor_uint(&ptr, end, &algo_len) != 0) {
                ptr = resume;  // not the tag
                continue;
            }

            if (ptr + algo_len > end) {
                ptr = resume;  // not the tag
                continue;
            }

            // Verify it's "bslz4"
            if (algo_len != 5 || memcmp(ptr, "bslz4", 5) != 0) {
                ptr = resume;  // not the tag
                continue;
            }
            ptr += algo_len;

            // Element 1: elem_size (unsigned integer)
            if (ptr >= end || (*ptr & CBOR_MAJOR_TYPE_MASK) != CBOR_MAJOR_UNSIGNED) {
                ptr = resume;  // not the tag
                continue;
            }

            uint64_t elem_size;
            if (read_cbor_uint(&ptr, end, &elem_size) != 0) {
                ptr = resume;  // not the tag
                continue;
            }
            *elem_size_out = (size_t)elem_size;

            // Element 2: compressed_bytes (byte string)
            const uint8_t* compressed_bytes;
            size_t compressed_len;
            if (read_cbor_bytes(&ptr, end, &compressed_bytes, &compressed_len) != 0) {
                ptr = resume;  // not the tag
                continue;
            }

            // Extract Dectris header (first 12 bytes):
            // - 8 bytes: original_size (big-endian uint64)
            // - 4 bytes: block_size (big-endian uint32)
            if (compressed_len < 12) {
                ptr = resume;  // not the tag
                continue;
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
