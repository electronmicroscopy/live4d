#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>

namespace dectris {

// Bounds publications ahead of the slowest of up to kMaxConsumers downstream callbacks.
class BatchWindow {
 public:
  static constexpr uint32_t kMaxConsumers = 4;

  explicit BatchWindow(uint32_t capacity, uint32_t consumers = 1) noexcept
      : capacity_(capacity), consumers_(std::clamp(consumers, 1U, kMaxConsumers)) {}

  bool admits(uint64_t next_batch_id) const noexcept {
    uint64_t released = released_[0].load(std::memory_order_acquire);
    for (uint32_t i = 1; i < consumers_; ++i)
      released = std::min(released, released_[i].load(std::memory_order_acquire));
    return next_batch_id <= released || next_batch_id - released <= capacity_;
  }

  void release(uint64_t batch_id, uint32_t consumer = 0) noexcept {
    if (consumer >= consumers_) return;
    auto& slot = released_[consumer];
    uint64_t current = slot.load(std::memory_order_relaxed);
    while (current < batch_id &&
           !slot.compare_exchange_weak(current, batch_id, std::memory_order_release,
                                       std::memory_order_relaxed)) {}
  }

 private:
  const uint32_t capacity_;
  const uint32_t consumers_;
  std::atomic<uint64_t> released_[kMaxConsumers]{};
};

inline BatchWindow* access(uint64_t handle) noexcept {
  return reinterpret_cast<BatchWindow*>(static_cast<uintptr_t>(handle));
}

// Lets the last of several END consumers request finite-mode completion, which does not
// drain other consumers' queues.
class CompletionLatch {
 public:
  explicit CompletionLatch(uint32_t parties) noexcept : remaining_(parties) {}

  // True for exactly one call: the one that brings the count to zero.
  bool arrive() noexcept { return remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1; }

 private:
  std::atomic<int64_t> remaining_;
};

inline CompletionLatch* access_latch(uint64_t handle) noexcept {
  return reinterpret_cast<CompletionLatch*>(static_cast<uintptr_t>(handle));
}

}  // namespace dectris

extern "C" {
void* dectris_batch_window_create(uint32_t capacity, uint32_t consumers);
void dectris_batch_window_destroy(void* handle);
void dectris_batch_window_release(void* handle, uint64_t batch_id, uint32_t consumer);
void* dectris_completion_latch_create(uint32_t parties);
void dectris_completion_latch_destroy(void* handle);
bool dectris_completion_latch_arrive(void* handle);
}
