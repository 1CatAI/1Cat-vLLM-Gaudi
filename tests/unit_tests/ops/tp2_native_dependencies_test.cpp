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
  NativeBufferRange token{1000, 4}, embed{1100, 32}, embedPeer{1200, 32}, attention{1300, 32};
  NativeBufferRange attentionPeer{1400, 32}, mhc{1500, 32}, late{1600, 64}, ffn{1700, 32};
  NativeBufferRange unpacked{1800, 32}, ffnPeer{1900, 32}, engramPeer{2000, 32};
  // Production segment 3 fuses layer-0 FFN with layer-1 Engram unpacking.
  std::vector<NativeDependencyNode> fused{
      {false, {token}, {embed}}, {true, {embed}, {embedPeer}},
      {false, {embed, embedPeer}, {attention}}, {true, {attention}, {attentionPeer}},
      {false, {embed}, {mhc}},
      {false, {attention, attentionPeer, mhc, {late.address + 16, 32}}, {ffn, unpacked}},
      {true, {ffn}, {ffnPeer}}, {true, {unpacked}, {engramPeer}},
      {false, {mhc}, {{2100, 32}}},
      {false, {ffn, ffnPeer, unpacked, engramPeer}, {{2200, 32}}}};
  const auto fusedDeps = prepareNativeDependencies(fused);
  const auto prefix = prepareNativeInputPrefix(fused, {late}, fusedDeps);
  assert(prefix.computes == 3 && prefix.collectives == 2);
  assert(fusedDeps[3].producer == 3 && fusedDeps[3].consumer == 5);
  assert(fusedDeps[1].consumer == prefix.computes);  // Retained suffix wait.
  auto rejectsPrefix = [&](const auto& bad, const auto& lateInputs) {
    try { prepareNativeInputPrefix(bad, lateInputs, fusedDeps); }
    catch (const std::invalid_argument&) { return; }
    assert(false && "Unsafe input prefix accepted");
  };
  rejectsPrefix(fused, std::vector<NativeBufferRange>{{3000, 16}});
  rejectsPrefix(fused, std::vector<NativeBufferRange>{});
  auto earlyRead = fused; earlyRead[0].inputs.push_back(late);
  rejectsPrefix(earlyRead, std::vector<NativeBufferRange>{late});
  auto inputWrite = fused; inputWrite[4].outputs.push_back({late.address + 8, 4});
  rejectsPrefix(inputWrite, std::vector<NativeBufferRange>{late});

  // Device Engram produces layer-1 rows outside the captured graph. Its TP
  // exchange must still be published before a prefix consumer, while a later
  // layer-14 input remains the actual split boundary.
  auto deviceEngram = fused;
  deviceEngram[5].inputs.pop_back();
  deviceEngram[5].outputs.pop_back();
  deviceEngram.push_back({false, {{2200, 32}, late}, {{2300, 32}}});
  deviceEngram.push_back({true, {{2300, 32}}, {{2400, 32}}});
  deviceEngram.push_back({false, {{2300, 32}, {2400, 32}}, {{2500, 32}}});
  const auto deviceDependencies = prepareNativeDependencies(deviceEngram);
  assert(deviceDependencies[3].producer == UINT32_MAX && deviceDependencies[3].consumer == 5);
  const auto devicePrefix = prepareNativeInputPrefix(deviceEngram, {late}, deviceDependencies);
  assert(devicePrefix.computes == 6 && devicePrefix.collectives == 4);
  std::cout << "TP dependencies: independent work, shared consumers, aliases and overwrite guards passed\n";
}
