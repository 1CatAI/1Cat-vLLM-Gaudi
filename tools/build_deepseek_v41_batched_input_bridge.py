# SPDX-License-Identifier: Apache-2.0
"""Private DSpark staging capability; preserve the serving adapter and transport.

The installed Bridge already exposes its transfer-manifest API. This replaces
N D2D submissions and producer events with one, retaining every source wait,
destination-consumer wait and tensor lifetime. It does not pack communication.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-abi", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmake-build-directory", type=Path, required=True)
    args = parser.parse_args()
    abi = json.loads(args.baseline_abi.read_text())
    for path, sha in abi["adapter_sources"].items():
        if digest(Path(path)) != sha:
            raise ValueError("The installed adapter's source changed: " + path)
    output = args.output.resolve()
    source = output / "source"
    if output.exists():
        raise FileExistsError("Use a fresh isolated capability build")
    shutil.copytree(args.baseline_source, source)
    header = source / "tp2_native_decode_graph.h"
    old = header.read_text()
    new = old
    public = "  void stageFixedInputs("
    if old.count(public) != 1:
        raise ValueError("Inspect the installed native input API before changing it")
    method = '''  void enableBatchedInputStaging() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load()==State::Instantiated && expected_groups_==1,
                "Batch input staging requires an instantiated DSpark protocol graph");
    batched_input_staging_=true;
  }
  uint64_t inputStagingSubmissions() const {return input_staging_submissions_.load();}

'''
    new = new.replace(public, method + public, 1)
    marker = "    for (const auto& copy : copies) {\n      // This helper retains"
    if new.count(marker) != 1:
        raise ValueError("Inspect staging's source/lifetime contract before changing it")
    batch = '''    if(batched_input_staging_ && copies.size()>1) {
      synapse_helpers::device::transfer_manifest transfers;
      transfers.reserve(copies.size());
      uint64_t bytes=0;
      for(const auto& copy:copies) {
        transfers.push_back({address(copy.source),address(copy.destination),
          reinterpret_cast<synapse_helpers::device_ptr>(copy.source.storage().data_ptr().get()),
          reinterpret_cast<synapse_helpers::device_ptr>(copy.destination.storage().data_ptr().get()),copy.bytes});
        bytes+=copy.bytes;
      }
      // The existing API retains address locks and source dependencies, then
      // registers one completion producer for every destination. The callback
      // holds the backend tensors until that completion, including views.
      auto result=device.copy_data_within_device(transfers,[held=copies]() mutable {held.clear();});
      checkSynapse(result.status,"native DSpark batched D2D input staging");
      input_update_copies_.fetch_add(copies.size());
      input_update_bytes_.fetch_add(bytes);
      input_staging_submissions_.fetch_add(1);
      return;
    }
'''
    new = new.replace(marker, batch + marker, 1)
    marker = "      input_update_copies_.fetch_add(1);"
    if new.count(marker) != 1:
        raise ValueError("Installed input copy accounting changed")
    new = new.replace(marker, "      input_staging_submissions_.fetch_add(1);\n" + marker, 1)
    marker = "  std::atomic<uint64_t> input_update_copies_{0};"
    new = new.replace(marker, "  bool batched_input_staging_=false;\n"
                      "  std::atomic<uint64_t> input_staging_submissions_{0};\n" + marker, 1)
    header.write_text(new)
    binding = source / "tp2_fused_ar_norm_bridge.cpp"
    old_binding = binding.read_text()
    marker = '      .def("stage_fixed_inputs", &tp2_native::NativeDecodeGraph::stageFixedInputs)'
    if old_binding.count(marker) != 1:
        raise ValueError("Installed binding differs from the supported input API")
    new_binding = old_binding.replace(marker,
        '      .def("enable_batched_input_staging", &tp2_native::NativeDecodeGraph::enableBatchedInputStaging)\n'
        '      .def("input_staging_submissions", &tp2_native::NativeDecodeGraph::inputStagingSubmissions)\n' + marker)
    binding.write_text(new_binding)
    patch = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile="installed/header", tofile="private/header"))
    patch += "".join(difflib.unified_diff(old_binding.splitlines(True), new_binding.splitlines(True),
                                         fromfile="installed/binding", tofile="private/binding"))
    (output / "private-input-staging.patch").write_text(patch)
    api_header = next(Path(path) for path in abi["headers"] if path.endswith("habana_eager/graph_storage.h"))
    command = [sys.executable, str(source / "build_tp2_fused_ar_norm_bridge.py"),
               "--bridge-source", str(api_header.parent.parent),
               "--cmake-build-directory", str(args.cmake_build_directory.resolve()),
               "--build-directory", str(output / "build")]
    for group, key, flag in (("bridge_dependencies", "backend", "--backend-library"),
                             ("bridge_dependencies", "native_hccl", "--native-hccl-library"),
                             ("native_dependencies", "synapse", "--synapse-library"),
                             ("native_dependencies", "hcl", "--hcl-library")):
        record = abi[group][key]
        if digest(Path(record["path"])) != record["sha256"]:
            raise ValueError("Installed dependency changed: " + key)
        command += [flag, record["path"]]
    (output / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    runtime_dirs = [str(Path(abi["native_dependencies"][key]["path"]).resolve().parent)
                    for key in ("synapse", "hcl")]
    runtime_dirs += [str(Path(abi["bridge_dependencies"]["backend"]["path"]).resolve().parent)]
    library_path = os.pathsep.join((*runtime_dirs, os.environ.get("LD_LIBRARY_PATH", "")))
    subprocess.run(command, check=True, env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", MAX_JOBS="1",
                                                LD_LIBRARY_PATH=library_path))
    for path, sha in abi["adapter_sources"].items():
        if digest(Path(path)) != sha:
            raise ValueError("The original adapter was changed during the private build")
    (output / "BUILD_PROOF.json").write_text(json.dumps(dict(
        baseline_abi=str(args.baseline_abi.resolve()), default_enabled=False,
        shared_replay_and_communication_sources_unchanged=True,
        runtime_api="Existing synapse_helpers::device transfer_manifest, no new runtime/driver API",
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest()), indent=2) + "\n")


if __name__ == "__main__":
    main()
