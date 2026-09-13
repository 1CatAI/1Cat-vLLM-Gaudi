// SPDX-License-Identifier: Apache-2.0
#include "tools/communication/tp2_native_dependencies.h"
#include <cassert>
#include <iostream>

int main() {
  NativeBufferRange x{100, 16}, peer{200, 16}, residual{300, 64}, control{400, 32}, out{500, 64};
  std::vector<NativeDependencyNode> nodes{
      {false, {residual}, {x}}, {true, {x}, {peer}},
      {false, {residual}, {control}}, {false, {x, peer, control}, {out}},
      {false, {peer, out}, {{600, 16}}}};
  auto plan = prepareNativeDependencies(nodes);
  assert(plan.size() == 1 && plan[0].producer == 0 && plan[0].consumer == 2 && plan[0].lastConsumer == 3);
  auto adjacent = nodes; adjacent.erase(adjacent.begin() + 2);
  assert(prepareNativeDependencies(adjacent)[0].consumer == 1);
  auto alias = nodes; alias[3].inputs[1] = {208, 8};
  assert(prepareNativeDependencies(alias)[0].consumer == 2);
  auto external = nodes; external.erase(external.begin());
  assert(prepareNativeDependencies(external)[0].producer == UINT32_MAX);
  auto shared = nodes;
  shared.insert(shared.begin() + 2, {true, {x}, {{700, 16}}});
  shared[4].inputs.push_back({700, 16});
  const auto pairs = prepareNativeDependencies(shared);
  assert(pairs.size() == 2 && pairs[0].consumer == pairs[1].consumer);
  auto rejects = [](const auto& bad) {
    try { prepareNativeDependencies(bad); } catch (const std::invalid_argument&) { return; }
    assert(false && "Unsafe dependency accepted");
  };
  auto overwritten = nodes; overwritten[2].outputs.push_back({104, 4}); rejects(overwritten);
  overwritten = nodes; overwritten[2].outputs.push_back(peer); rejects(overwritten);
  auto missing = nodes; missing.resize(3); rejects(missing);
  auto same = nodes; same[1].outputs[0] = x; rejects(same);
  std::cout << "TP dependencies: independent work, shared consumers, aliases and overwrite guards passed\n";
}
