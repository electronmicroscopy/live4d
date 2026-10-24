#pragma once

#include <cstddef>

namespace holoscan::ops::dectris {

// Version of these operators, exposed to Python as dectris_holoscan_ops.__version__ (the
// runner prints it, so a log shows which build ran).
constexpr const char* kOperatorsVersion = "2.0.1";

// Default batch size when the runtime `batch_size` parameter is left unset.
constexpr std::size_t kDefaultBatchSize = 2048;

// Hard ceiling on the batch size. It sizes the preallocated pinned/GPU pools in both
// ZmqRxOp and DecompressGpuOp, so the runtime `batch_size` parameter may be any positive
// value up to this (e.g. 1024 = one scan line of a 1024-wide scan, or 2048). Memory for the
// full ceiling is allocated at every start, whatever batch_size is. To allow larger
// batches, raise this and rebuild; the pools grow ~148 KB per unit for 96x96 uint16.
constexpr std::size_t kMaxBatchSize = 2048;
constexpr std::size_t kPoolSize = 3;

// Defaults only — detector geometry and pixel dtype are runtime parameters on the
// operators and are validated against what the stream actually carries.
constexpr std::size_t kDefaultDetectorWidth = 96;
constexpr std::size_t kDefaultDetectorHeight = 96;
constexpr const char* kDefaultPixelDtype = "uint16";

// Nominal bslz4 block size in bytes (bitshuffle's default target). Used for worst-case
// buffer sizing; the actual block size is read from each image's 12-byte Dectris header.
constexpr std::size_t kNominalBlockSize = 8192;

constexpr const char* kDefaultZmqEndpoint = "tcp://127.0.0.1:5555";

}  // namespace holoscan::ops::dectris

namespace holoscan::ops {

constexpr unsigned int BATCH_SIZE = static_cast<unsigned int>(dectris::kMaxBatchSize);

}  // namespace holoscan::ops
