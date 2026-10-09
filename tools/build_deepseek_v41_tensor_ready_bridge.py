# SPDX-License-Identifier: Apache-2.0
"""Expose the explicit cold tensor-ready API in an isolated DSpark bridge."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-bridge', type=Path, required=True)
    parser.add_argument('--runtime-proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bridge-source', type=Path, required=True)
    parser.add_argument('--cmake-build-directory', type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base_bridge.resolve(), args.output.resolve()
    metadata = json.loads(base.with_suffix('.abi.json').read_text())
    runtime = json.loads(args.runtime_proof.read_text())
    if digest(base) != metadata['binary_sha256'] or digest(Path(runtime['library'])) != runtime['sha256']:
        raise ValueError('Locked bridge/runtime changed')
    cmake = args.cmake_build_directory.resolve()
    if not (cmake / '_deps/abseil-cpp-src/absl/strings/str_format.h').is_file():
        # Recover the exact parent dependency root before compilation. A
        # retained CMake build can have its dependencies in a sibling build.
        flags = next(line.split(' = ', 1)[1] for line in (base.parent / 'build.ninja').read_text().splitlines()
                     if line.startswith('cflags = '))
        roots = [Path(arg[2:]) for arg in shlex.split(flags) if arg.startswith('-I')]
        compatible = [path for path in roots
                      if (path / '_deps/abseil-cpp-src/absl/strings/str_format.h').is_file()]
        if len(compatible) != 1:
            raise ValueError('Recover one exact parent CMake dependency root')
        cmake = compatible[0]
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    for path, sha in metadata['adapter_sources'].items():
        path = Path(path)
        if digest(path) != sha:
            raise ValueError('Locked source changed: '+str(path))
        shutil.copy2(path, source / path.name)
    header = source / 'tp2_native_decode_graph.h'
    original = header.read_text()
    anchor = '  void capture(std::vector<std::shared_ptr<PreparedGroupPlan>> plans,\n'
    methods = '''  void configureTensorReadySignals(std::vector<uint32_t> ordinals) {
    TORCH_CHECK(state_.load() == State::Created && !ordinals.empty(),
                "Tensor-ready signals must be configured on a new graph");
    tensor_ready_ordinals_ = std::move(ordinals);
  }

  std::vector<std::vector<uint64_t>> tensorReadyInfo() const { return tensor_ready_info_; }

'''
    assert original.count(anchor) == 1
    text = original.replace(anchor, methods+anchor)
    anchor = '      state_.store(State::Instantiated);\n      prepareCompletionAddresses();'
    integration = '''      if (!tensor_ready_ordinals_.empty()) {
        TORCH_CHECK(usesJointPlan() && !bounded_tiles_ && !segmentedPrefixConfigured() &&
                    tensor_ready_ordinals_.size() == prepared_consumers_.size(),
                    "Tensor-ready capability requires one complete joint graph");
        using Inspect = synStatus (*)(void*, uint64_t*, uint64_t*, uint64_t);
        using Configure = synStatus (*)(void*, const uint32_t*, const uint32_t*, const uint32_t*, uint64_t);
        const auto inspect = reinterpret_cast<Inspect>(
            dlsym(RTLD_DEFAULT, "synNativeComputeGraphGetTensorReadySegmentsV1"));
        const auto configure = reinterpret_cast<Configure>(
            dlsym(RTLD_DEFAULT, "synNativeComputeGraphConfigureTensorReadySignalsV1"));
        TORCH_CHECK(inspect && configure, "Tensor-ready capability requires its private runtime API");
        std::vector<uint64_t> ends(segment_count_.load()), externalCounts(segment_count_.load());
        checkSynapse(inspect(syn_graph_, ends.data(), externalCounts.data(), ends.size()),
                     "synNativeComputeGraphGetTensorReadySegmentsV1");
        auto producers = prepared_producers_;
        if (producers.empty()) {
          for (const auto consumer : prepared_consumers_) {
            TORCH_CHECK(consumer > 0, "Implicit tensor-ready producer cannot be outside this graph");
            producers.push_back(consumer-1);
          }
        }
        checkSynapse(configure(syn_graph_, producers.data(), prepared_consumers_.data(),
                               tensor_ready_ordinals_.data(), producers.size()),
                     "synNativeComputeGraphConfigureTensorReadySignalsV1");
        tensor_ready_info_ = {ends, externalCounts,
                              {producers.begin(), producers.end()},
                              {prepared_consumers_.begin(), prepared_consumers_.end()},
                              {tensor_ready_ordinals_.begin(), tensor_ready_ordinals_.end()}};
      }
'''
    assert text.count(anchor) == 1
    text = text.replace(anchor, integration+anchor)
    anchor = '  RuntimeApis::SynGraph syn_graph_ = nullptr;\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '  std::vector<uint32_t> tensor_ready_ordinals_;\n'
                        '  std::vector<std::vector<uint64_t>> tensor_ready_info_;\n'+anchor)
    header.write_text(text)
    cpp = source / 'tp2_fused_ar_norm_bridge.cpp'
    original_cpp = cpp.read_text()
    anchor = '      .def("configure_topology", &tp2_native::NativeDecodeGraph::configureTopology)'
    assert original_cpp.count(anchor) == 1
    cpp.write_text(original_cpp.replace(anchor, anchor+'\n'
        '      .def("configure_tensor_ready_signals", &tp2_native::NativeDecodeGraph::configureTensorReadySignals)\n'
        '      .def("tensor_ready_info", &tp2_native::NativeDecodeGraph::tensorReadyInfo)'))
    patch = []
    for name, before in (('tp2_native_decode_graph.h', original), ('tp2_fused_ar_norm_bridge.cpp', original_cpp)):
        patch.extend(difflib.unified_diff(before.splitlines(True), (source / name).read_text().splitlines(True),
                                          fromfile='a/'+name, tofile='b/'+name))
    (output / 'source.patch').write_text(''.join(patch))
    builder = Path(__file__).with_name('communication') / 'build_tp2_fused_ar_norm_bridge.py'
    shutil.copy2(builder, source / builder.name)
    command = [sys.executable, str(source / builder.name), '--bridge-source', str(args.bridge_source.resolve()),
               '--cmake-build-directory', str(cmake),
               '--build-directory', str(output / 'build')]
    for argument, key in (('backend-library', 'backend'), ('native-hccl-library', 'native_hccl')):
        item = metadata['bridge_dependencies'][key]
        if digest(Path(item['path'])) != item['sha256']:
            raise ValueError('Bridge dependency changed')
        command += ['--'+argument, item['path']]
    hcl = metadata['native_dependencies']['hcl']
    if digest(Path(hcl['path'])) != hcl['sha256']:
        raise ValueError('HCL dependency changed')
    command += ['--synapse-library', runtime['library'], '--hcl-library', hcl['path']]
    environment = dict(os.environ, MAX_JOBS='4')
    environment['LD_LIBRARY_PATH'] = ':'.join((str(Path(runtime['library']).parent),
        str(Path(hcl['path']).parent), str(Path(metadata['bridge_dependencies']['backend']['path']).parent),
        environment.get('LD_LIBRARY_PATH', '')))
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = dict(base_bridge=str(base), base_sha256=metadata['binary_sha256'], runtime=runtime['library'],
                 runtime_sha256=runtime['sha256'], source_patch_sha256=digest(output / 'source.patch'),
                 default_enabled=False, peer_exchange_unchanged=True, final_completion_unchanged=True,
                 status='built_unqualified', command=command)
    (output / 'RESULT.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
