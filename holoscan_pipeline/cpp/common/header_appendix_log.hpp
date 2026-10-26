#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/logger/logger.hpp>

#include "fast_cbor_parse.h"

namespace dectris {

// Log START header_appendix entries; called once per series, outside the image path.
inline void log_header_appendix(uint64_t series_id, const uint8_t* data, size_t size) {
  const auto visit = [](void* context, const char* key, const char* value) {
    const auto id = *static_cast<const uint64_t*>(context);
    if (key) HOLOSCAN_LOG_INFO("series {} header_appendix[{}] = {}", id, key, value);
    else HOLOSCAN_LOG_INFO("series {} header_appendix = {}", id, value);
  };
  const int status = fast_visit_header_appendix(data, size, visit, &series_id);
  if (status == 1) HOLOSCAN_LOG_INFO("series {} has no header_appendix", series_id);
  else if (status != 0) HOLOSCAN_LOG_WARN("series {} has malformed header_appendix", series_id);
}

}  // namespace dectris
