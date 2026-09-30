// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <stdexcept>

// Callers hold the graph mutex. Scheduling may advance ahead of the execute
// queue; an open host transaction and an open device transaction are distinct.
class NativeSegmentedEpochs {
 public:
  uint64_t scheduled() const { return scheduled_; }
  uint64_t executing() const { return executing_; }
  void beginScheduled(uint64_t epoch) {
    if (!epoch || scheduled_) throw std::invalid_argument("Segmented prefix already scheduled");
    scheduled_ = epoch;
  }
  void finishScheduled(uint64_t epoch) {
    if (!epoch || scheduled_ != epoch) throw std::invalid_argument("Segmented suffix scheduling mismatch");
    scheduled_ = 0;
  }
  void beginExecuting(uint64_t epoch, uint64_t completed, uint64_t scheduled) {
    if (executing_ || !ordered(epoch, completed, scheduled))
      throw std::invalid_argument("Segmented prefix execution is out of order");
    executing_ = epoch;
  }
  void finishExecuting(uint64_t epoch, uint64_t completed, uint64_t scheduled) {
    if (executing_ != epoch || !ordered(epoch, completed, scheduled))
      throw std::invalid_argument("Segmented suffix execution is out of order");
    executing_ = 0;
  }
 private:
  static bool ordered(uint64_t epoch, uint64_t completed, uint64_t scheduled) {
    return epoch && completed != UINT64_MAX && epoch == completed + 1 && epoch <= scheduled;
  }
  uint64_t scheduled_ = 0, executing_ = 0;
};
