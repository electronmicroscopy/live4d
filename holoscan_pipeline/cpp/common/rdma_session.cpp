#include "rdma_session.hpp"

extern "C" void* dectris_rdma_session_create() {
  try { return new dectris::RdmaSession; } catch (...) { return nullptr; }
}

extern "C" void dectris_rdma_session_destroy(void* handle) {
  delete static_cast<dectris::RdmaSession*>(handle);
}

extern "C" uint64_t dectris_rdma_session_attach(void* handle, uint64_t qp,
                                               uint64_t credit, uint32_t rkey) {
  return handle ? static_cast<dectris::RdmaSession*>(handle)->attach(qp, credit, rkey) : 0;
}

extern "C" int dectris_rdma_session_detach(void* handle, uint64_t generation) {
  return handle && static_cast<dectris::RdmaSession*>(handle)->detach(generation);
}

extern "C" dectris::SessionSnapshot dectris_rdma_session_snapshot(void* handle) {
  return handle ? static_cast<dectris::RdmaSession*>(handle)->snapshot()
                : dectris::SessionSnapshot{};
}
