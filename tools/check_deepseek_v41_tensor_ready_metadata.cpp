// SPDX-License-Identifier: Apache-2.0
// CPU-only epoch/ownership capability checks, never a speed qualification.
#include "communication/dsv41_tensor_ready_signals.h"
#include <cassert>
#include <iostream>

int main() {
  const auto recipe = dsv41TensorReadyDeltas({3, 5, 9, 11}, {2, 0, 2, 0},
                                            {0, 2}, {1, 3}, {0, 0});
  assert((recipe == std::vector<uint64_t>{0, 0}));
  const auto early = dsv41TensorReadyDeltas({3, 5, 9, 11}, {2, 0, 2, 0},
                                           {0, 2}, {1, 3}, {1, 2});
  assert((early == std::vector<uint64_t>{2, 2}));
  for (uint64_t epoch : {uint64_t(100), uint64_t(10100), uint64_t(20100)}) {
    assert(dsv41TensorReadyTarget(epoch + 3, early[0]) == epoch + 1);
    assert(dsv41TensorReadyTarget(epoch + 9, early[1]) == epoch + 7);
  }
  unsigned rejected = 0;
  const auto fails = [&](auto fn) {
    try { fn(); } catch (const std::invalid_argument&) { ++rejected; return; }
    throw std::logic_error("Invalid signal metadata was accepted");
  };
  fails([] { dsv41TensorReadyDeltas({2, 4}, {0, 0}, {0}, {1}, {1}); });
  fails([] { dsv41TensorReadyDeltas({2, 4}, {1, 0}, {0}, {0}, {1}); });
  fails([] { dsv41TensorReadyDeltas({2, 4}, {2, 0}, {0}, {1}, {1}); });
  fails([] { dsv41TensorReadyDeltas({2, 4}, {1, 0}, {UINT32_MAX}, {1}, {1}); });
  fails([] { dsv41TensorReadyTarget(2, 2); });
  fails([] { dsv41TensorReadyTarget(uint64_t(1) << 60, 0); });
  assert(rejected == 6);
  std::cout << "Recipe defaults, two early signals, three replay epochs and six invalid contracts passed\n";
}
