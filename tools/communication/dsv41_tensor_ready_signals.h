// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Cold capability metadata only; no recipe, network or replay implementation.
// Native RecipeLauncher reserves external-tensor completions before its
// ordinary commands. The external execution ordinal is one-based. The final
// completion still owns all recipe buffers and CCB retirement.
inline std::vector<uint64_t> dsv41TensorReadyDeltas(
    const std::vector<uint64_t>& segmentEnds,
    const std::vector<uint64_t>& externalCounts,
    const std::vector<uint32_t>& producers,
    const std::vector<uint32_t>& consumers,
    const std::vector<uint32_t>& ordinals) {
  if (segmentEnds.empty() || segmentEnds.size() != externalCounts.size() ||
      producers.empty() || producers.size() != consumers.size() || producers.size() != ordinals.size())
    throw std::invalid_argument("Tensor-ready metadata coverage differs");
  uint64_t previous = 0;
  for (size_t index = 0; index < segmentEnds.size(); ++index) {
    if (segmentEnds[index] <= previous || segmentEnds[index] - previous <= externalCounts[index])
      throw std::invalid_argument("External tensor signals must precede a real final recipe completion");
    previous = segmentEnds[index];
  }
  std::vector<uint64_t> result(producers.size());
  for (size_t index = 0; index < producers.size(); ++index) {
    const auto producer = producers[index];
    if (producer == std::numeric_limits<uint32_t>::max()) {
      if (ordinals[index]) throw std::invalid_argument("External graph input has no captured tensor-ready signal");
      continue;
    }
    if (producer >= segmentEnds.size() || consumers[index] >= segmentEnds.size() || producer >= consumers[index])
      throw std::invalid_argument("Communication must retain a later real consumer");
    if (!ordinals[index]) continue;
    if (ordinals[index] > externalCounts[producer])
      throw std::invalid_argument("Requested tensor signal is absent from the captured recipe");
    const uint64_t start = producer ? segmentEnds[producer-1] : 0;
    result[index] = segmentEnds[producer] - (start + ordinals[index]);
  }
  return result;
}

inline uint64_t dsv41TensorReadyTarget(uint64_t recipeTarget, uint64_t delta) {
  if (recipeTarget <= delta || recipeTarget >= (uint64_t(1) << 60))
    throw std::invalid_argument("Tensor-ready target is stale or overflows its long SO");
  return recipeTarget - delta;
}
