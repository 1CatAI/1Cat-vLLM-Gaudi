// SPDX-License-Identifier: Apache-2.0
// Exercise the real compiler eligibility pass without allocating a device.
#include "graph_compiler/passes/allocate_tensors.cpp"
#include "gaudi_graph.h"
#include "graph_editor.h"
#include "node_factory.h"
#include <cassert>
#include <iostream>

int main()
{
    assert(GCFG_ENABLE_PERSISTENT_INPUT_REUSE.value());
    for (bool sameSection : {false, true})
    for (uint64_t outputOffset : {0, 8, 64})
    for (bool externalOutput : {false, true})
    {
        const TSize shape[] = {8};
        auto input = std::make_shared<Tensor>(1, shape, syn_type_float);
        auto first = std::make_shared<Tensor>(1, shape, syn_type_float);
        auto second = std::make_shared<Tensor>(1, shape, syn_type_float);
        auto output = std::make_shared<Tensor>(1, shape, syn_type_float);
        for (auto tensor : {input, output})
            tensor->setMemoryDescriptor(synMemoryDescriptor(true));
        input->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
        output->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + (sameSection ? 0 : 1));
        input->setMemorySectionOffset(0);
        output->setMemorySectionOffset(outputOffset);
        input->setDramOffset(0x10000);
        output->setDramOffset(0x10000 + outputOffset);
        input->setTensorAsReusable(true);
        input->enforceOutput(externalOutput);
        GaudiGraph graph;
        GraphEditor::addNode(graph, NodeFactory::createNode(
            {input}, {first}, nullptr, NodeFactory::memcpyNodeTypeName, "read_input"));
        GraphEditor::addNode(graph, NodeFactory::createNode(
            {first}, {second}, nullptr, NodeFactory::memcpyNodeTypeName, "temporary"));
        GraphEditor::addNode(graph, NodeFactory::createNode(
            {second}, {output}, nullptr, NodeFactory::memcpyNodeTypeName, "later_output"));
        graph.getGraphAnnotation().memoryCoherence = std::make_shared<TensorCoherenceMapping>(graph);
        const auto selected = gatherReusablePersistentTensors(graph);
        const bool canReuse = std::find(selected.begin(), selected.end(), input) != selected.end();
        const bool overlap = sameSection && outputOffset < 32;
        assert(canReuse == (!overlap && !externalOutput));
    }
    std::cout << "persistent input reuse: overlapping later outputs protected; disjoint ranges remain reusable\n";
}
