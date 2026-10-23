#pragma once

#include <atomic>
#include <cstdint>

namespace dectris {

class BatchWindow {
 public:
  explicit BatchWindow(uint32_t capacity) noexcept : capacity_(capacity) {}

  bool admits(uint64_t next_batch_id) const noexcept {
    const uint64_t released = released_.load(std::memory_order_acquire);
    return next_batch_id <= released || next_batch_id - released <= capacity_;
  }

  void release(uint64_t batch_id) noexcept {
    uint64_t current = released_.load(std::memory_order_relaxed);
    while (current < batch_id &&
           !released_.compare_exchange_weak(current, batch_id, std::memory_order_release,
                                            std::memory_order_relaxed)) {}
  }

 private:
  const uint32_t capacity_;
  std::atomic<uint64_t> released_{0};
};

inline BatchWindow* access(uint64_t handle) noexcept {
  return reinterpret_cast<BatchWindow*>(static_cast<uintptr_t>(handle));
}

}  // namespace dectris

extern "C" {
void* dectris_batch_window_create(uint32_t capacity);
void dectris_batch_window_destroy(void* handle);
void dectris_batch_window_release(void* handle, uint64_t batch_id);
}
