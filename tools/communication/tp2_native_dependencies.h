// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct NativeBufferRange {
  uint64_t address = 0, bytes = 0;
  bool overlaps(const NativeBufferRange& other) const {
    if (!bytes || !other.bytes) return false;
    if (address > UINT64_MAX - bytes || other.address > UINT64_MAX - other.bytes)
      throw std::invalid_argument("Native binding range overflow");
    return address < other.address + other.bytes && other.address < address + bytes;
  }
};

struct NativeDependencyNode {
  bool exchange = false;
  std::vector<NativeBufferRange> inputs, outputs;
};

struct NativeCollectiveDependency {
  uint32_t producer = UINT32_MAX, consumer = UINT32_MAX, lastConsumer = UINT32_MAX;
  NativeBufferRange input, output;
};

struct NativeInputPrefix {
  uint32_t computes = UINT32_MAX;
  size_t collectives = 0;
};

inline NativeInputPrefix prepareNativeInputPrefix(
    const std::vector<NativeDependencyNode>& nodes, const std::vector<NativeBufferRange>& lateInputs,
    const std::vector<NativeCollectiveDependency>& dependencies) {
  if (lateInputs.empty() || dependencies.empty())
    throw std::invalid_argument("Native input prefix requires late input bindings and TP dependencies");
  NativeInputPrefix prefix;
  std::vector<bool> used(lateInputs.size(), false);
  uint32_t segment = 0;
  for (const auto& node : nodes) {
    for (size_t i = 0; i < lateInputs.size(); ++i) {
      if (!lateInputs[i].bytes) throw std::invalid_argument("Native late input is empty");
      for (const auto& output : node.outputs)
        if (output.overlaps(lateInputs[i]))
          throw std::invalid_argument("Native graph mutates a late input binding");
      for (const auto& input : node.inputs)
        if (input.overlaps(lateInputs[i])) {
          prefix.computes = std::min(prefix.computes, segment);
          used[i] = true;
        }
    }
    if (!node.exchange) ++segment;
  }
  if (!prefix.computes || prefix.computes == UINT32_MAX ||
      std::find(used.begin(), used.end(), false) != used.end())
    throw std::invalid_argument("Native graph has no verified late-input-free prefix");
  for (const auto& dependency : dependencies) {
    // A device-side producer can live outside the captured compute graph while
    // its exchange still feeds a prefix compute. Publish that collective with
    // the prefix; stream input readiness protects its external source.
    const bool producedInPrefix =
        dependency.producer != UINT32_MAX && dependency.producer < prefix.computes;
    const bool consumedInPrefix = dependency.consumer < prefix.computes;
    if (!producedInPrefix && !consumedInPrefix) break;
    ++prefix.collectives;
  }
  if (!prefix.collectives || prefix.collectives == dependencies.size())
    throw std::invalid_argument("Native input boundary does not split TP publication");
  for (size_t i = prefix.collectives; i < dependencies.size(); ++i)
    if (dependencies[i].consumer < prefix.computes)
      throw std::invalid_argument("Native input prefix consumes an unpublished exchange");
  return prefix;
}

inline std::vector<NativeCollectiveDependency> prepareNativeDependencies(
    const std::vector<NativeDependencyNode>& nodes) {
  std::vector<uint32_t> segments(nodes.size());
  uint32_t count = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    segments[i] = count;
    if (!nodes[i].exchange) ++count;
  }
  std::vector<NativeCollectiveDependency> result;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (!node.exchange) continue;
    if (node.inputs.size() != 1 || node.outputs.size() != 1 || !node.inputs[0].bytes ||
        node.inputs[0].bytes != node.outputs[0].bytes || node.inputs[0].overlaps(node.outputs[0]))
      throw std::invalid_argument("Native explicit dependency requires disjoint peer bindings");
    NativeCollectiveDependency dep;
    dep.input = node.inputs[0]; dep.output = node.outputs[0];
    for (size_t j = 0; j < i; ++j)
      for (const auto& output : nodes[j].outputs) {
        if (output.overlaps(dep.input)) {
          if (nodes[j].exchange) throw std::invalid_argument("Peer producer must be a compute segment");
          dep.producer = segments[j];
        }
      }
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      for (const auto& input : nodes[j].inputs) {
        if (input.overlaps(dep.output)) {
          if (nodes[j].exchange) throw std::invalid_argument("Peer consumer must be a compute segment");
          if (dep.consumer == UINT32_MAX) dep.consumer = segments[j];
          dep.lastConsumer = segments[j];
        }
      }
      for (const auto& output : nodes[j].outputs) {
        if (dep.consumer == UINT32_MAX && (output.overlaps(dep.input) || output.overlaps(dep.output)))
          throw std::invalid_argument("Independent segment overwrites an in-flight collective binding");
      }
    }
    if (dep.consumer == UINT32_MAX || (dep.producer != UINT32_MAX && dep.producer >= dep.consumer) ||
        (!result.empty() && dep.consumer < result.back().consumer))
      throw std::invalid_argument("Native dependency is missing or out of collective order");
    result.push_back(dep);
  }
  return result;
}
