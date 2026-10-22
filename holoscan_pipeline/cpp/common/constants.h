#pragma once

#include <cstddef>

namespace holoscan::ops::dectris {

constexpr std::size_t kMaxBatchSize = 1000;

constexpr std::size_t kDetectorWidth = 96;
constexpr std::size_t kDetectorHeight = 96;
constexpr std::size_t kDetectorPixels = kDetectorHeight * kDetectorWidth;

constexpr const char* kDefaultZmqEndpoint = "tcp://127.0.0.1:5555";

}  // namespace holoscan::ops::dectris

namespace holoscan::ops {

constexpr unsigned int BATCH_SIZE = static_cast<unsigned int>(dectris::kMaxBatchSize);
constexpr unsigned int DETECTOR_WIDTH = static_cast<unsigned int>(dectris::kDetectorWidth);
constexpr unsigned int DETECTOR_HEIGHT = static_cast<unsigned int>(dectris::kDetectorHeight);
constexpr unsigned int DETECTOR_SIZE = static_cast<unsigned int>(dectris::kDetectorPixels);

}  // namespace holoscan::ops
