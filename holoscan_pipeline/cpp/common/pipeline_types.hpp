#pragma once

#include "compressed_batch_schema_traits.hpp"
#include "decoded_batch_schema_traits.hpp"

namespace holoscan::ops {
using CompressedBatch = ::dectris::schema::compressed::CompressedBatchT;
using DecodedBatch = ::dectris::schema::decoded::DecodedBatchT;
}
