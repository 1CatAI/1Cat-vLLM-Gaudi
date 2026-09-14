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
  assert(NativeGraphTopology::supportsV41SegmentedPrefix(5, 43, false));
  for (size_t collectives : {3, 8, 40, 42, 44})
    assert(!NativeGraphTopology::supportsV41SegmentedPrefix(5, collectives, false));
  assert(!NativeGraphTopology::supportsV41SegmentedPrefix(5, 43, true));
  assert(!NativeGraphTopology::supportsV41SegmentedPrefix(1, 43, false));
  for (size_t collectives : {40, 42, 43}) {
    assert(NativeGraphTopology::supportsV41Dependencies(5, collectives, false));
    assert(!NativeGraphTopology::supportsV41Dependencies(5, collectives, true));
    std::vector<NativeNodeKind> nodes;
    for (size_t index = 0; index < collectives; ++index) {
      nodes.push_back({false, false});
      nodes.push_back({true, true});
    }
    nodes.push_back({false, false});
    const auto layout = NativeGraphTopology::prepare(nodes, 5, false, collectives, false);
    assert(layout.externalCollectives == 0 && layout.consumers.size() == collectives);
    assert(layout.computeCount == collectives + 1);
    nodes.erase(nodes.begin() + 1);
    bool rejected = false;
    try { NativeGraphTopology::prepare(nodes, 5, false, collectives, false); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  assert(NativeGraphTopology::supportsV41Dependencies(1, 8, false));
  assert(!NativeGraphTopology::supportsV41Dependencies(5, 41, false));
  assert(!NativeGraphTopology::supportsV41Dependencies(5, 44, false));
  assert(!NativeGraphTopology::supportsV41Dependencies(4, 43, false));
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
  std::vector<NativeNodeKind> v4;
  for (size_t index = 0; index < 86; ++index) {
    v4.push_back({false, false});
    v4.push_back({true, true});
  }
  v4.push_back({false, false});
  const auto v4_layout = NativeGraphTopology::prepare(v4, 6, false, 86, false);
  assert(v4_layout.computeCount == 87 && v4_layout.consumers.size() == 86);
  assert(v4_layout.prefixNodes == 0 && v4_layout.externalCollectives == 0);
  const auto shared_consumer = NativeGraphTopology::prepare(
      {{false, false}, {true, true}, {true, true}, {false, false}}, 1, false, 2, false);
  assert(shared_consumer.computeCount == 2);
  assert(shared_consumer.consumers == std::vector<uint32_t>({1, 1}));
  auto incomplete_v4 = v4;
  incomplete_v4.erase(incomplete_v4.begin() + 1);
  bool rejected = false;
  try { NativeGraphTopology::prepare(incomplete_v4, 6, false, 86, false); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  std::cout << "NATIVE_TOPOLOGY_EXACT\n";
}
