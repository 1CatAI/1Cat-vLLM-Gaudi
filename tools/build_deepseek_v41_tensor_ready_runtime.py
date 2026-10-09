# SPDX-License-Identifier: Apache-2.0
"""Add an opt-in cold tensor-ready plan API to one private runtime unit.

Default replay, communication and C1 source are unchanged. The new API
replaces selected NIC producer completion offsets only after proving the
external signals exist in the captured producer and retaining final retirement.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--runtime-source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--audit-tensor-bindings', action='store_true',
                        help='Print captured input/output bindings during the cold capability query only')
    args = parser.parse_args()
    parent, runtime, build = args.parent.resolve(), args.runtime_source.resolve(), args.build.resolve()
    build.mkdir(parents=True, exist_ok=False)
    common = build / 'source/synapse/src/runtime/scal/common'
    common.mkdir(parents=True)
    copied = {
        'native_compute_program.hpp': parent / 'source/native_compute_program.hpp',
        'stream_compute_scal.cpp': parent / 'source/stream_compute_scal.cpp',
        'native_compute_graph.hpp': runtime / 'synapse/src/runtime/scal/common/native_compute_graph.hpp',
        'dsv41_tensor_ready_signals.h': Path(__file__).with_name('communication') / 'dsv41_tensor_ready_signals.h',
    }
    fingerprints = {str(path): digest(path) for path in copied.values()}
    for name, source in copied.items():
        shutil.copy2(source, common / name)
    header = common / 'native_compute_graph.hpp'
    before = header.read_text()
    anchor = '    synStatus close();\n'
    assert before.count(anchor) == 1
    after = before.replace(anchor, anchor + '''    synStatus configureTensorReadySignals(
                                         const uint32_t* producers, const uint32_t* consumers,
                                         const uint32_t* ordinals, uint64_t count);
    synStatus getTensorReadySegments(uint64_t* ends, uint64_t* externalCounts, uint64_t count) const;
''')
    header.write_text(after)
    source = common / 'stream_compute_scal.cpp'
    original = source.read_text()
    addition = '''
// DSpark isolated cold capability. No class/plan layout change or C1 default.
synStatus NativeComputeGraph::getTensorReadySegments(uint64_t* ends, uint64_t* externalCounts,
                                                     uint64_t count) const
{
    if (m_state != State::INSTANTIATED || m_replayActive || m_replayCount || !m_jointPlan ||
        !ends || !externalCounts || count != m_segments.size() || m_jointPlan->collectiveSplit ||
        count != m_jointPlan->program.segmentCompletionOffsets.size()) return synInvalidArgument;
    for (size_t index = 0; index < count; ++index)
    {
        ends[index] = m_jointPlan->program.segmentCompletionOffsets[index];
        externalCounts[index] = m_segments[index].commands.completionIncrementsBeforeCommands;
    }
    return synSuccess;
}

synStatus NativeComputeGraph::configureTensorReadySignals(const uint32_t* producers,
    const uint32_t* consumers, const uint32_t* ordinals, uint64_t count)
{
    if (!producers || !consumers || !ordinals || !count || count > 128 ||
        !m_jointPlan || count != m_jointPlan->program.producerOffsets.size()) return synInvalidArgument;
    std::vector<uint64_t> ends(m_segments.size()), externalCounts(m_segments.size());
    if (getTensorReadySegments(ends.data(), externalCounts.data(), ends.size()) != synSuccess)
        return synInvalidArgument;
    try
    {
        const auto deltas = dsv41TensorReadyDeltas(ends, externalCounts, {producers, producers+count},
                                                  {consumers, consumers+count}, {ordinals, ordinals+count});
        auto adjusted = m_jointPlan->program.producerOffsets;
        for (size_t index = 0; index < count; ++index)
        {
            const uint64_t expected = producers[index] == UINT32_MAX ? 0 : ends.at(producers[index]);
            if (adjusted[index] != expected) return synInvalidArgument;
            adjusted[index] -= deltas[index];
        }
        m_jointPlan->program.producerOffsets.swap(adjusted);
        return synSuccess;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "NATIVE_TENSOR_READY_REJECT reason=%s\\n", error.what());
        return synInvalidArgument;
    }
}

extern "C" __attribute__((visibility("default"))) synStatus
synNativeComputeGraphGetTensorReadySegmentsV1(synNativeComputeGraphHandle handle,
                                             uint64_t* ends, uint64_t* counts, uint64_t size)
{
    if (!handle || !handle->graph) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    return handle->graph->getTensorReadySegments(ends, counts, size);
}

extern "C" __attribute__((visibility("default"))) synStatus
synNativeComputeGraphConfigureTensorReadySignalsV1(synNativeComputeGraphHandle handle,
    const uint32_t* producers, const uint32_t* consumers, const uint32_t* ordinals, uint64_t count)
{
    if (!handle || !handle->graph) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    return handle->graph->configureTensorReadySignals(producers, consumers, ordinals, count);
}
'''
    if args.audit_tensor_bindings:
        audit_anchor = '''        ends[index] = m_jointPlan->program.segmentCompletionOffsets[index];'''
        audit = '''        for (const auto& binding : m_segments[index].bindings)
        {
            std::fprintf(stderr, "NATIVE_BINDING_AUDIT segment=%zu recipe=%llu tensor=%llu "
                                 "type=%u address=0x%llx sizes=",
                         index, static_cast<unsigned long long>(m_segments[index].recipeSequence),
                         static_cast<unsigned long long>(binding.tensorId), unsigned(binding.tensorType),
                         static_cast<unsigned long long>(binding.deviceAddress));
            for (size_t dim = 0; dim < HABANA_DIM_MAX; ++dim)
                std::fprintf(stderr, "%s%llu", dim ? "," : "", static_cast<unsigned long long>(binding.sizes[dim]));
            std::fprintf(stderr, "\\n");
        }
'''
        assert addition.count(audit_anchor) == 1
        addition = addition.replace(audit_anchor, audit+audit_anchor)
    source.write_text(original.replace('#include "stream_compute_scal.hpp"',
                                       '#include "stream_compute_scal.hpp"\n#include "dsv41_tensor_ready_signals.h"')
                      + addition)
    patches = []
    for name, old in (('native_compute_graph.hpp', before), ('stream_compute_scal.cpp', original)):
        patches.extend(difflib.unified_diff(old.splitlines(True), (common / name).read_text().splitlines(True),
                                            fromfile='a/'+name, tofile='b/'+name))
    (build / 'source.patch').write_text(''.join(patches))
    commands = json.loads((parent / 'commands.json').read_text())
    compile_command = list(commands['compile'])
    compile_command[1:1] = ['-I'+str(build / 'source/synapse/src')]
    compile_command[compile_command.index('-o')+1] = str(build / 'stream_compute_scal.cpp.o')
    compile_command[compile_command.index('-c')+1] = str(source)
    link = list(commands['link'])
    link[link.index('-o')+1] = str(build / 'libSynapse.so')
    replaced = 0
    for index, value in enumerate(link):
        if value.endswith('/stream_compute_scal.cpp.o'):
            link[index] = str(build / 'stream_compute_scal.cpp.o')
            replaced += 1
    if replaced != 1:
        raise ValueError('Parent link must contain exactly one native compute runtime unit')
    (build / 'commands.json').write_text(json.dumps(dict(compile=compile_command, link=link), indent=2)+'\n')
    with (build / 'build.log').open('w') as log:
        for command in (compile_command, link):
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    for path, sha in fingerprints.items():
        if digest(Path(path)) != sha:
            raise RuntimeError('Shared prerequisite changed while building the immutable capability')
    proof = dict(parent=str(parent / 'libSynapse.so'), parent_sha256=digest(parent / 'libSynapse.so'),
                 library=str(build / 'libSynapse.so'), sha256=digest(build / 'libSynapse.so'),
                 source_prerequisites=fingerprints, source_patch_sha256=digest(build / 'source.patch'),
                 status='built_unqualified', default_enabled=False, class_layout_changed=False,
                 whole_recipe_retirement_unchanged=True, hcl_unchanged=True, performance_qualified=False)
    (build / 'RESULT.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
