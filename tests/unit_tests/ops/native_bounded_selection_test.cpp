// SPDX-License-Identifier: Apache-2.0
#include "native_bounded_selection.h"
#include <cassert>

int main() {
  const std::vector<uint32_t> minimum{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 1, 2, 0};
  for (uint32_t bound = 0; bound <= 8; ++bound) {
    const auto selected = NativeBoundedSelection::prepare(minimum, {0, 9, 9}, {9, 10, 13}, bound, 8);
    assert(selected.producers.size() == 3 && selected.consumers.size() == 3);
    assert(selected.segments[selected.producers[0]] == 0);
    assert(selected.segments[selected.producers[1]] == 9);
    assert(selected.segments[selected.producers[2]] == 9);
    assert(selected.segments[selected.consumers[0]] == 9);
    assert(selected.segments[selected.consumers[1]] == 10);
    assert(selected.segments[selected.consumers[2]] == 13);
    assert(selected.segments.size() == 4 + bound + (bound > 2 ? 2 : bound));
  }
  auto external = NativeBoundedSelection::prepare(minimum, {UINT32_MAX}, {0}, 0, 8);
  assert(external.producers[0] == UINT32_MAX && external.consumers[0] == 0);
  auto rejected = [](auto fn) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    assert(false);
  };
  rejected([&] { NativeBoundedSelection::prepare(minimum, {1}, {9}, 0, 8); });
  rejected([&] { NativeBoundedSelection::prepare(minimum, {0}, {2}, 0, 8); });
  rejected([&] { NativeBoundedSelection::prepare(minimum, {0}, {9}, 9, 8); });
  rejected([&] { NativeBoundedSelection::prepare({0, 10, 0}, {0}, {2}, 1, 8); });
  const std::vector<uint32_t> dual{0,1,2,3,4,9,0};
  for (uint32_t bound = 0; bound <= 8; ++bound) {
    auto selected = NativeBoundedSelection::prepare(dual, {0}, {6}, bound, 8);
    assert(selected.segments.front() == 0 && selected.segments.back() == 6);
    assert(selected.segments.size() == (bound <= 4 ? 2 + bound : 3));
    if (bound > 4) assert(selected.segments[1] == 5);
  }
  rejected([&] { NativeBoundedSelection::prepare(dual, {5}, {6}, 8, 8); });
  rejected([&] { NativeBoundedSelection::prepare({0, 1}, {}, {}, 0, 8); });
  rejected([&] { NativeBoundedSelection::prepare(minimum, {0, 9}, {13, 10}, 8, 8); });
  rejected([&] { NativeBoundedSelection::prepare(minimum, {13}, {9}, 8, 8); });
}
