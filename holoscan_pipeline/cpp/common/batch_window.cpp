#include "batch_window.hpp"

extern "C" void* dectris_batch_window_create(uint32_t capacity) {
  if (capacity == 0) return nullptr;
  return new dectris::BatchWindow(capacity);
}

extern "C" void dectris_batch_window_destroy(void* handle) {
  delete static_cast<dectris::BatchWindow*>(handle);
}

extern "C" void dectris_batch_window_release(void* handle, uint64_t batch_id) {
  if (handle) static_cast<dectris::BatchWindow*>(handle)->release(batch_id);
}
