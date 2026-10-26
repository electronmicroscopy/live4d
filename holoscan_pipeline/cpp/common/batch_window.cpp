#include "batch_window.hpp"

extern "C" void* dectris_batch_window_create(uint32_t capacity, uint32_t consumers) {
  if (capacity == 0 || consumers == 0 || consumers > dectris::BatchWindow::kMaxConsumers)
    return nullptr;
  return new dectris::BatchWindow(capacity, consumers);
}

extern "C" void dectris_batch_window_destroy(void* handle) {
  delete static_cast<dectris::BatchWindow*>(handle);
}

extern "C" void dectris_batch_window_release(void* handle, uint64_t batch_id, uint32_t consumer) {
  if (handle) static_cast<dectris::BatchWindow*>(handle)->release(batch_id, consumer);
}

extern "C" void* dectris_completion_latch_create(uint32_t parties) {
  if (parties == 0) return nullptr;
  return new dectris::CompletionLatch(parties);
}

extern "C" void dectris_completion_latch_destroy(void* handle) {
  delete static_cast<dectris::CompletionLatch*>(handle);
}

extern "C" bool dectris_completion_latch_arrive(void* handle) {
  return handle && static_cast<dectris::CompletionLatch*>(handle)->arrive();
}
