#pragma once

#include <cstddef>

namespace holoscan::ops::dectris {

constexpr std::size_t kMaxBatchSize = 1000;

constexpr std::size_t kDetectorWidth = 96;
constexpr std::size_t kDetectorHeight = 96;
constexpr std::size_t kDetectorPixels = kDetectorHeight * kDetectorWidth;
constexpr std::size_t kMaxDetectorWidth = 192;
constexpr std::size_t kMaxDetectorHeight = 192;
constexpr std::size_t kMaxDetectorPixels = kMaxDetectorHeight * kMaxDetectorWidth;

constexpr const char* kDefaultZmqEndpoint = "tcp://127.0.0.1:5555";

}  // namespace holoscan::ops::dectris

namespace holoscan::ops {

constexpr unsigned int BATCH_SIZE = static_cast<unsigned int>(dectris::kMaxBatchSize);
constexpr unsigned int DETECTOR_WIDTH = static_cast<unsigned int>(dectris::kDetectorWidth);
constexpr unsigned int DETECTOR_HEIGHT = static_cast<unsigned int>(dectris::kDetectorHeight);
constexpr unsigned int DETECTOR_SIZE = static_cast<unsigned int>(dectris::kDetectorPixels);
constexpr unsigned int MAX_DETECTOR_WIDTH = static_cast<unsigned int>(dectris::kMaxDetectorWidth);
constexpr unsigned int MAX_DETECTOR_HEIGHT = static_cast<unsigned int>(dectris::kMaxDetectorHeight);
constexpr unsigned int MAX_DETECTOR_SIZE = static_cast<unsigned int>(dectris::kMaxDetectorPixels);

}  // namespace holoscan::ops
