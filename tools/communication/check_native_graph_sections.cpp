// SPDX-License-Identifier: Apache-2.0
// No device acquisition: exercise the graph clones used by eager input copies.
#include "synapse_api.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("Synapse status " + std::to_string(status));
}

int main() {
  check(synInitialize());
  synGraphHandle graph = nullptr;
  check(synGraphCreateEager(&graph, synDeviceGaudi2));
  synTensor tensors[2] = {};
  synSectionHandle original[2] = {};
  synTensorGeometry geometry = {};
  geometry.dims = 1;
  geometry.sizes[0] = 8;
  for (unsigned i = 0; i < 2; ++i) {
    check(synSectionCreate(&original[i], 0, graph));
    check(synTensorHandleCreate(&tensors[i], graph, DATA_TENSOR, i ? "output" : "input"));
    check(synTensorSetGeometry(tensors[i], &geometry, synGeometryMaxSizes));
    check(synTensorSetDeviceDataType(tensors[i], syn_type_single));
    check(synTensorAssignToSection(tensors[i], original[i], 0));
  }
  check(synNodeCreate(graph, tensors, tensors + 1, 1, 1, nullptr, 0, "memcpy", "copy", nullptr, nullptr));
  unsigned invalidHandles = 0, retainedHandles = 0;
  for (unsigned iteration = 0; iteration < 64; ++iteration) {
    uint32_t tensorCount = 0, nodeCount = 0;
    synGraphHandle clone = nullptr;
    check(synGraphDuplicate(graph, &clone, nullptr, &tensorCount, nullptr, &nodeCount));
    std::vector<synTensorHandleMap> tensorMap(tensorCount);
    std::vector<synNodeHandleMap> nodeMap(nodeCount);
    check(synGraphDuplicate(graph, &clone, tensorMap.data(), &tensorCount, nodeMap.data(), &nodeCount));
    std::vector<synSectionHandle> handles;
    for (const auto& entry : tensorMap) {
      synSectionHandle section = nullptr;
      uint64_t offset = 0;
      check(synTensorGetSection(entry.newHandle, &section, &offset));
      invalidHandles += section == nullptr || section == original[0] || section == original[1];
      if (section) handles.push_back(section);
    }
    check(synGraphDestroy(clone));
    for (auto section : handles) {
      bool persistent = false;
      retainedHandles += synSectionGetPersistent(section, &persistent) == synSuccess;
    }
  }
  // A fresh empty section must reuse a retired clone slot, rather than advance
  // the million-entry pool on every duplication.
  synSectionHandle probe = nullptr;
  check(synSectionCreate(&probe, 0, graph));
  const uint64_t slot = reinterpret_cast<uint64_t>(probe) & 0xffffffffULL;
  check(synSectionDestroy(probe));
  check(synGraphDestroy(graph));
  check(synDestroy());
  const bool passed = invalidHandles == 0 && retainedHandles == 0 && slot < 8;
  std::cout << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"clones\":64,\"invalid_section_handles\":" << invalidHandles
            << ",\"retained_section_handles\":" << retainedHandles
            << ",\"next_slot\":" << slot << "}\n";
  return passed ? 0 : 1;
}
