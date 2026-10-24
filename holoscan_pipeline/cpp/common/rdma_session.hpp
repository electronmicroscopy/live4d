#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>

namespace dectris {

// Process-local mailbox. The controller owns QPs; only the source callback polls them.
// A detached acknowledgment is the boundary after which their owner may destroy them.
enum class SessionPhase : uint64_t {
  kIdle, kAttaching, kAttached, kDrained, kDetaching, kDetached, kStopped
};

struct SessionSnapshot {
  uint64_t generation{}, phase{}, attached{}, drained{}, detached{};
  uint64_t series_id{}, batch_id{}, position{}, safe_credit{}, sent_credit{};
  uint64_t writes_outstanding{}, starts{}, stops{};
  uint64_t attached_ns{}, drained_ns{}, detached_ns{};
};

class RdmaSession {
 public:
  enum class CommandKind { kNone, kAttach, kDetach };
  struct Command {
    CommandKind kind{CommandKind::kNone};
    uint64_t generation{}, qp{}, credit_address{};
    uint32_t credit_rkey{};
  };

  uint64_t attach(uint64_t qp, uint64_t credit_address, uint32_t credit_rkey) {
    std::lock_guard lock(mutex_);
    if (!qp || !credit_address || command_.kind != CommandKind::kNone ||
        (state_.phase != uint64_t(SessionPhase::kIdle) &&
         state_.phase != uint64_t(SessionPhase::kDetached))) return 0;
    const auto generation = ++state_.generation;
    command_ = {CommandKind::kAttach, generation, qp, credit_address, credit_rkey};
    state_.phase = uint64_t(SessionPhase::kAttaching);
    return generation;
  }

  bool detach(uint64_t generation) {
    std::lock_guard lock(mutex_);
    if (state_.generation != generation || state_.phase != uint64_t(SessionPhase::kDrained) ||
        command_.kind != CommandKind::kNone) return false;
    command_ = {CommandKind::kDetach, generation};
    state_.phase = uint64_t(SessionPhase::kDetaching);
    return true;
  }

  Command take_command() {
    std::lock_guard lock(mutex_);
    const auto result = command_;
    command_ = {};
    return result;
  }

  SessionSnapshot snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
  }

  void started() {
    std::lock_guard lock(mutex_);
    ++state_.starts;
  }

  void attached(uint64_t generation, uint64_t batch_id) {
    std::lock_guard lock(mutex_);
    state_.attached = generation;
    state_.batch_id = batch_id;
    state_.position = state_.safe_credit = state_.sent_credit = state_.series_id = 0;
    state_.writes_outstanding = 0;
    state_.attached_ns = timestamp();
    state_.phase = uint64_t(SessionPhase::kAttached);
  }

  void drained(uint64_t generation, uint64_t series, uint64_t batch_id,
               uint64_t position, uint64_t safe_credit, uint64_t sent_credit) {
    std::lock_guard lock(mutex_);
    if (state_.phase != uint64_t(SessionPhase::kAttached) || state_.generation != generation)
      return;
    state_.drained = generation;
    state_.series_id = series;
    state_.batch_id = batch_id;
    state_.position = position;
    state_.safe_credit = safe_credit;
    state_.sent_credit = sent_credit;
    state_.writes_outstanding = 0;
    state_.drained_ns = timestamp();
    state_.phase = uint64_t(SessionPhase::kDrained);
  }

  void detached(uint64_t generation) {
    std::lock_guard lock(mutex_);
    state_.detached = generation;
    state_.detached_ns = timestamp();
    state_.phase = uint64_t(SessionPhase::kDetached);
  }

  void stopped() {
    std::lock_guard lock(mutex_);
    ++state_.stops;
    command_ = {};
    state_.phase = uint64_t(SessionPhase::kStopped);
  }

 private:
  static uint64_t timestamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  mutable std::mutex mutex_;
  Command command_;
  SessionSnapshot state_;
};

}  // namespace dectris

extern "C" {
void* dectris_rdma_session_create();
void dectris_rdma_session_destroy(void* handle);
uint64_t dectris_rdma_session_attach(void* handle, uint64_t qp, uint64_t credit, uint32_t rkey);
int dectris_rdma_session_detach(void* handle, uint64_t generation);
dectris::SessionSnapshot dectris_rdma_session_snapshot(void* handle);
}
