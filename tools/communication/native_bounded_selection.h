// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Cold-only selection and dependency remapping. Zero marks mandatory work;
// 1..limit mark independent tile recipes. Optional tag 9 selects the complete
// scorer instead of the prefix when bound > 4. Consumers mask unused outputs.
// State writers, collective endpoints and the terminal recipe are mandatory.
struct NativeBoundedSelection {
  std::vector<uint32_t> segments, producers, consumers;

  static NativeBoundedSelection prepare(const std::vector<uint32_t>& minimum,
                                       const std::vector<uint32_t>& producers,
                                       const std::vector<uint32_t>& consumers,
                                       uint32_t bound, uint32_t limit) {
    if (minimum.empty() || minimum.size() >= UINT32_MAX || !limit || limit > 8 || bound > limit ||
        minimum.front() || minimum.back() || producers.size() != consumers.size())
      throw std::invalid_argument("Invalid bounded segment contract");
    NativeBoundedSelection result;
    const bool hasWhole = std::find(minimum.begin(), minimum.end(), 9) != minimum.end();
    if (hasWhole && limit != 8) throw std::invalid_argument("Whole scorer requires eight-tile capacity");
    std::vector<uint32_t> inverse(minimum.size(), UINT32_MAX);
    for (uint32_t i = 0; i < minimum.size(); ++i) {
      const auto tag = minimum[i];
      if (tag > limit && tag != 9) throw std::invalid_argument("Tile bound exceeds prepared capacity");
      const bool selected = !tag || (hasWhole && bound > 4 ? tag == 9 : tag <= bound);
      if (selected) {
        inverse[i] = result.segments.size();
        result.segments.push_back(i);
      }
    }
    for (size_t i = 0; i < consumers.size(); ++i) {
      const auto producer = producers[i], consumer = consumers[i];
      if (consumer >= minimum.size() || minimum[consumer] || (i && consumer < consumers[i - 1]) ||
          (producer != UINT32_MAX && (producer >= consumer || minimum[producer])))
        throw std::invalid_argument("Collective endpoint cannot be omitted or reordered");
      result.producers.push_back(producer == UINT32_MAX ? UINT32_MAX : inverse[producer]);
      result.consumers.push_back(inverse[consumer]);
    }
    return result;
  }
};
