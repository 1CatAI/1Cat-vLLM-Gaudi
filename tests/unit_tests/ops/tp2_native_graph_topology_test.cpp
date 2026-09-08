// SPDX-License-Identifier: Apache-2.0
#include "tools/communication/tp2_native_graph_topology.h"
#include <cassert>
#include <iostream>

static std::vector<NativeNodeKind> chain(size_t groups, bool peer) {
  std::vector<NativeNodeKind> nodes;
  if (groups == 8) nodes = {{false, false}, {true, peer}};
  for (size_t index = 0; index < groups * 16; ++index) {
    nodes.push_back({false, false});
    nodes.push_back({true, peer});
  }
  nodes.push_back({false, false});
  return nodes;
}

int main() {
  for (size_t groups : {1, 8}) for (bool peer : {false, true}) {
    const auto nodes = chain(groups, peer);
    const auto layout = NativeGraphTopology::prepare(nodes, groups);
    assert(layout.prefixNodes == (groups == 8 ? 2 : 0));
    assert(layout.externalCollectives == (groups == 8 ? 1 : 0));
    assert(layout.consumers.size() == groups * 16);
    assert(layout.computeCount == groups * 16 * (peer ? 1 : 2) + 1);
    for (uint32_t index = 0; index < layout.consumers.size(); ++index)
      assert(layout.consumers[index] == index * (peer ? 1 : 2) + 1);
  }
  auto reject = [](const auto& nodes) {
    try { NativeGraphTopology::prepare(nodes, 1); }
    catch (const std::invalid_argument&) { return; }
    assert(false && "Invalid dependency topology was accepted");
  };
  auto missing_consumer = chain(1, true); missing_consumer.pop_back(); reject(missing_consumer);
  auto missing_producer = chain(1, true); missing_producer.erase(missing_producer.begin()); reject(missing_producer);
  const auto external = NativeGraphTopology::prepare(missing_producer, 1, true);
  assert(external.computeCount == 16 && external.consumers.front() == 0 && external.consumers.back() == 15);
  auto adjacent_peers = chain(1, true); adjacent_peers.erase(adjacent_peers.begin() + 2); reject(adjacent_peers);
  auto invalid_kind = chain(1, true); invalid_kind[0].peerOnly = true; reject(invalid_kind);
  auto dropped_collective = chain(1, true); dropped_collective[1] = {false, false}; reject(dropped_collective);
  auto mixed = chain(1, true); mixed[3].peerOnly = false;
  const auto layout = NativeGraphTopology::prepare(mixed, 1);
  assert(layout.computeCount == 18 && layout.consumers[1] == 2 && layout.consumers[2] == 4);
  std::cout << "NATIVE_TOPOLOGY_EXACT\n";
}
