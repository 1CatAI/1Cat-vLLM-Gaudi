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
  size_t prefixNodes = 0;
  uint32_t computeCount = 0;
  size_t externalCollectives = 0;
  std::vector<uint32_t> consumers;

  static NativeGraphTopology prepare(const std::vector<NativeNodeKind>& nodes, size_t groups,
                                     bool externalInputCompletion = false) {
    if (groups != 1 && groups != 8) throw std::invalid_argument("Invalid native group coverage");
    NativeGraphTopology result;
    size_t total = 0;
    for (size_t index = 0; index < nodes.size(); ++index) {
      const auto node = nodes[index];
      if (node.peerOnly && !node.exchange) throw std::invalid_argument("Peer node must exchange");
      if (!node.exchange) continue;
      ++total;
      if (groups == 8 && !result.prefixNodes) {
        result.prefixNodes = index + 1;
        result.externalCollectives = 1;
      }
    }
    if (total != groups * 16 + result.externalCollectives)
      throw std::invalid_argument("Native graph collective coverage differs");
    for (size_t index = result.prefixNodes; index < nodes.size(); ++index) {
      const auto node = nodes[index];
      if (node.exchange) {
        if ((!result.computeCount && !externalInputCompletion) ||
            (!result.consumers.empty() && result.consumers.back() == result.computeCount))
          throw std::invalid_argument("Native exchange requires a distinct compute producer and consumer");
        result.consumers.push_back(result.computeCount);
      }
      if (!node.peerOnly) ++result.computeCount;
    }
    if (result.consumers.size() != groups * 16 || result.consumers.back() >= result.computeCount)
      throw std::invalid_argument("Native exchange has no compiled consumer");
    return result;
  }
};
