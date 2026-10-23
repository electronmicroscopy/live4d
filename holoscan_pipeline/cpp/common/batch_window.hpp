#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace dectris {

class BatchWindow {
 public:
  BatchWindow(uint32_t capacity, uint32_t consumers = 1)
      : capacity_(capacity), consumer_count_(consumers) {
    if (capacity == 0 || consumers == 0 || consumers > 64) {
      throw std::invalid_argument("invalid batch window dimensions");
    }
    slots_.reset(new Slot[capacity]);
    complete_mask_ = consumers == 64 ? UINT64_MAX : (uint64_t{1} << consumers) - 1;
  }

  bool admits(uint64_t next_batch_id) const noexcept {
    if (next_batch_id == 0) return false;
    const uint64_t released = released_.load(std::memory_order_acquire);
    return next_batch_id <= released || next_batch_id - released <= capacity_;
  }

  bool acknowledge(uint32_t consumer, uint64_t batch_id) noexcept {
    if (consumer >= consumer_count_ || batch_id == 0) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t released = released_.load(std::memory_order_relaxed);
    if (batch_id <= released) return true;
    if (batch_id - released > capacity_) return false;

    Slot& slot = slots_[(batch_id - 1) % capacity_];
    if (slot.batch_id != 0 && slot.batch_id != batch_id) return false;
    if (slot.batch_id == 0) slot.batch_id = batch_id;
    slot.acknowledgements |= uint64_t{1} << consumer;

    while (released != UINT64_MAX) {
      const uint64_t next = released + 1;
      Slot& next_slot = slots_[(next - 1) % capacity_];
      if (next_slot.batch_id != next || next_slot.acknowledgements != complete_mask_) break;
      next_slot.batch_id = 0;
      next_slot.acknowledgements = 0;
      released = next;
    }
    released_.store(released, std::memory_order_release);
    return true;
  }

  void release(uint64_t batch_id) noexcept { acknowledge(0, batch_id); }

 private:
  struct Slot {
    uint64_t batch_id{};
    uint64_t acknowledgements{};
  };

  const uint32_t capacity_;
  const uint32_t consumer_count_;
  uint64_t complete_mask_{};
  std::unique_ptr<Slot[]> slots_;
  std::atomic<uint64_t> released_{0};
  std::mutex mutex_;
};

inline BatchWindow* access(uint64_t handle) noexcept {
  return reinterpret_cast<BatchWindow*>(static_cast<uintptr_t>(handle));
}

}  // namespace dectris

extern "C" {
void* dectris_batch_window_create(uint32_t capacity);
void* dectris_batch_window_create_for_consumers(uint32_t capacity, uint32_t consumers);
void dectris_batch_window_destroy(void* handle);
void dectris_batch_window_release(void* handle, uint64_t batch_id);
int dectris_batch_window_acknowledge(void* handle, uint32_t consumer, uint64_t batch_id);
}
