#include "batch_window.hpp"

namespace {

void* create_batch_window(uint32_t capacity, uint32_t consumers) noexcept {
  try {
    return new dectris::BatchWindow(capacity, consumers);
  } catch (...) {
    return nullptr;
  }
}

}  // namespace

extern "C" void* dectris_batch_window_create(uint32_t capacity) {
  return create_batch_window(capacity, 1);
}

extern "C" void* dectris_batch_window_create_for_consumers(uint32_t capacity,
                                                            uint32_t consumers) {
  return create_batch_window(capacity, consumers);
}

extern "C" void dectris_batch_window_destroy(void* handle) {
  delete static_cast<dectris::BatchWindow*>(handle);
}

extern "C" void dectris_batch_window_release(void* handle, uint64_t batch_id) {
  if (handle) static_cast<dectris::BatchWindow*>(handle)->release(batch_id);
}

extern "C" int dectris_batch_window_acknowledge(void* handle, uint32_t consumer,
                                                uint64_t batch_id) {
  if (!handle) return 0;
  return static_cast<dectris::BatchWindow*>(handle)->acknowledge(consumer, batch_id) ? 1 : 0;
}
