// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct NativeNodeKind {
  bool exchange;
  bool peerOnly;
};

struct NativeGraphTopology {
  static bool supportsV41SegmentedPrefix(size_t groups, size_t collectives, bool externalPrefix) {
    // PP0 keeps embedding + 40 layer exchanges + two Engram exchanges.
    // Its three long-context indexers each add two real exchanges; the
    // producer/consumer analysis still discovers and checks the exact cut.
    return groups == 5 && (collectives == 43 || collectives == 45 ||
                           collectives == 47 || collectives == 49) && !externalPrefix;
  }

  static bool supportsV41Dependencies(size_t groups, size_t collectives, bool externalPrefix) {
    // Each long-context indexer contributes two real AllGathers. PP0 has
    // three indexers plus two Engram exchanges (and optionally embedding);
    // PP1 has five indexers. Exact coverage is still checked by prepare().
    const bool stageCollectives = collectives == 40 || (collectives >= 42 && collectives <= 50);
    return !externalPrefix &&
        ((groups == 5 && stageCollectives) ||
         (groups == 1 && collectives >= 8 && collectives <= 16 && collectives % 2 == 0));
  }

  size_t prefixNodes = 0;
  uint32_t computeCount = 0;
  size_t externalCollectives = 0;
  std::vector<uint32_t> consumers;

  static NativeGraphTopology prepare(const std::vector<NativeNodeKind>& nodes, size_t groups,
                                     bool externalInputCompletion = false,
                                     size_t expectedCollectives = 0, bool externalPrefix = false) {
    const bool explicitTopology = expectedCollectives != 0;
    if (!groups || (!explicitTopology && groups != 1 && groups != 8))
      throw std::invalid_argument("Invalid native group coverage");
    const size_t expected = explicitTopology ? expectedCollectives : groups * 16;
    const bool hasPrefix = explicitTopology ? externalPrefix : groups == 8;
    NativeGraphTopology result;
    size_t total = 0;
    for (size_t index = 0; index < nodes.size(); ++index) {
      const auto node = nodes[index];
      if (node.peerOnly && !node.exchange) throw std::invalid_argument("Peer node must exchange");
      if (!node.exchange) continue;
      ++total;
      if (hasPrefix && !result.prefixNodes) {
        result.prefixNodes = index + 1;
        result.externalCollectives = 1;
      }
    }
    if (total != expected + result.externalCollectives)
      throw std::invalid_argument("Native graph collective coverage differs");
    for (size_t index = result.prefixNodes; index < nodes.size(); ++index) {
      const auto node = nodes[index];
      if (node.exchange) {
        if ((!result.computeCount && !externalInputCompletion) ||
            (!explicitTopology && !result.consumers.empty() && result.consumers.back() == result.computeCount))
          throw std::invalid_argument("Native exchange requires a distinct compute producer and consumer");
        result.consumers.push_back(result.computeCount);
      }
      if (!node.peerOnly) ++result.computeCount;
    }
    if (result.consumers.size() != expected || result.consumers.back() >= result.computeCount)
      throw std::invalid_argument("Native exchange has no compiled consumer");
    return result;
  }
};
