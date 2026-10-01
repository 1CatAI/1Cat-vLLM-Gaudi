// SPDX-License-Identifier: Apache-2.0
#include "tools/communication/tp2_native_segmented_epochs.h"
#include <cassert>
#include <iostream>

template<class F> void rejects(F action) {
  try { action(); } catch (const std::invalid_argument&) { return; }
  assert(false && "Invalid replay ordering accepted");
}
int main() {
  NativeSegmentedEpochs state;
  // Queue two complete tokens before the execute thread runs the first one.
  state.beginScheduled(1);
  rejects([&] { state.beginScheduled(2); });
  rejects([&] { state.finishScheduled(2); });
  state.finishScheduled(1);
  state.beginScheduled(2);
  state.finishScheduled(2);
  assert(!state.scheduled() && !state.executing());
  rejects([&] { state.beginExecuting(2, 0, 2); });
  state.beginExecuting(1, 0, 2);
  rejects([&] { state.beginExecuting(2, 1, 2); });
  rejects([&] { state.finishExecuting(2, 0, 2); });
  state.finishExecuting(1, 0, 2);
  state.beginExecuting(2, 1, 2);
  // The frontend may open a future prefix while this suffix is executing.
  state.beginScheduled(3);
  state.finishExecuting(2, 1, 3);
  assert(state.scheduled() == 3 && !state.executing());
  state.finishScheduled(3);
  state.beginExecuting(3, 2, 3);
  state.finishExecuting(3, 2, 3);
  rejects([&] { state.finishExecuting(3, 2, 3); });
  rejects([&] { state.beginExecuting(4, 3, 3); });
  rejects([&] { state.beginExecuting(0, UINT64_MAX, 0); });
  std::cout << "QUEUED_SEGMENTED_REPLAY_FIFO_EXACT\n";
}
