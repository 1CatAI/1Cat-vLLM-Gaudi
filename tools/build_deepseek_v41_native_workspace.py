# SPDX-License-Identifier: Apache-2.0
"""Build an isolated native runtime with graph-local scratch reuse.

Only ordered segments within one graph share scratch. Allocations remain
owned by that graph's existing segment resources until final completion.
Neither captured addresses nor graph class layouts are changed after capture.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_source(original):
    anchor = "    status         = segment.memory->initialize(globalSize, arcSize, workspaceSize);"
    if original.count(anchor) != 1:
        raise ValueError("Unsupported native segment allocation implementation")
    # No graph layout change: existing segment resources own growing arenas.
    candidate = original.replace("#include <limits>\n", "#include <limits>\n#include <cstdlib>\n")
    candidate = candidate.replace(anchor, '''    const char* reuseMode = std::getenv("VLLM_HPU_DSV41_NATIVE_WORKSPACE_REUSE");
    const bool reuseScratch = reuseMode && reuseMode[0] == '1' && reuseMode[1] == '\\0';
    uint64_t scratchAddress = 0;
    uint64_t allocationSize = workspaceSize;
    if (reuseScratch && workspaceSize)
    {
        // Recipes execute in capture order on this graph's compute stream.
        // Only scratch is shared; program memory and launch tensors are not.
        for (const auto& previous : m_segments)
        {
            const auto bytes = previous.memory->workspaceAllocationBytes();
            if (bytes >= 8191 && bytes - 8191 >= workspaceSize)
            {
                scratchAddress = previous.memory->workspaceDeviceAddress();
                break;
            }
        }
        if (scratchAddress)
            allocationSize = 0;
        else
        {
            // Retain each old arena rather than relocating captured pointers.
            // Geometric growth bounds retained capacity below twice the max.
            allocationSize = 8192;
            while (allocationSize < workspaceSize) allocationSize <<= 1;
        }
    }
    if (allocationSize && allocationSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes)
    {
        status = synOutOfDeviceMemory;
        m_state = State::INVALID;
        return nullptr;
    }
    status = segment.memory->initialize(globalSize, arcSize, allocationSize);''')
    old = "    segment.workspaceAddress = workspaceSize ? segment.memory->workspaceDeviceAddress() : workspaceAddress;"
    if candidate.count(old) != 1:
        raise ValueError("Unsupported native scratch binding implementation")
    candidate = candidate.replace(old, '''    segment.workspaceAddress = scratchAddress ? scratchAddress :
        (workspaceSize ? segment.memory->workspaceDeviceAddress() : workspaceAddress);
    if (reuseScratch && workspaceSize)
        std::fprintf(stderr, "NATIVE_WORKSPACE_REUSE requested=%llu allocated=%llu reused=%u\\n",
            static_cast<unsigned long long>(workspaceSize),
            static_cast<unsigned long long>(allocationSize), unsigned(scratchAddress != 0));''')
    # Check retained capacity only when a new allocation is required.
    old_limit = '''    if (m_workspaceAllocationBytes > maximumWorkspaceBytes || workspaceSize > maximumWorkspaceBytes ||
        (workspaceSize && workspaceSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes))'''
    if candidate.count(old_limit) != 1:
        raise ValueError("Unsupported native workspace bound")
    candidate = candidate.replace(old_limit, '''    if (m_workspaceAllocationBytes > maximumWorkspaceBytes || workspaceSize > maximumWorkspaceBytes)''')
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    args = parser.parse_args()
    parent, build = args.parent.resolve(), args.build.resolve()
    build.mkdir(parents=True, exist_ok=False)
    common = build / "source/synapse/src/runtime/scal/common"
    common.mkdir(parents=True)
    prerequisites = [parent / "source/stream_compute_scal.cpp", parent / "source/native_compute_program.hpp"]
    fingerprints = {str(p): digest(p) for p in prerequisites}
    original = prerequisites[0].read_text()
    patched = patch_source(original)
    source = common / "stream_compute_scal.cpp"
    source.write_text(patched)
    (common / "native_compute_program.hpp").write_bytes(prerequisites[1].read_bytes())
    (build / "source.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), patched.splitlines(True), fromfile="a/stream_compute_scal.cpp",
        tofile="b/stream_compute_scal.cpp")))
    commands = json.loads((parent / "commands.json").read_text())
    compile_command = list(commands["compile"])
    compile_command[1:1] = ["-I" + str(build / "source/synapse/src")]
    compile_command[compile_command.index("-o") + 1] = str(build / "stream_compute_scal.cpp.o")
    compile_command[compile_command.index("-c") + 1] = str(source)
    baseline_link = list(commands["link"])
    baseline = build / "parent-relinked.so"
    baseline_link[baseline_link.index("-o") + 1] = str(baseline)
    link = list(commands["link"])
    link[link.index("-o") + 1] = str(build / "libSynapse.so")
    indices = [i for i, value in enumerate(link) if value.endswith("/stream_compute_scal.cpp.o")]
    if len(indices) != 1:
        raise ValueError("Parent must link one native compute runtime unit")
    link[indices[0]] = str(build / "stream_compute_scal.cpp.o")
    (build / "commands.json").write_text(json.dumps(dict(compile=compile_command, link=link), indent=2) + "\n")
    with (build / "build.log").open("w") as log:
        for command in (baseline_link, compile_command, link):
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    parent_sha = digest(parent / "libSynapse.so")
    if digest(baseline) != parent_sha:
        raise RuntimeError("Parent link inputs no longer reproduce the qualified runtime")
    for path, sha in fingerprints.items():
        if digest(Path(path)) != sha:
            raise RuntimeError("Shared prerequisite changed during private build")
    proof = dict(schema=1, kind="native_ordered_scratch_v1",
                 parent=str(parent / "libSynapse.so"), parent_sha256=parent_sha,
                 parent_relinked_sha256=digest(baseline),
                 library=str(build / "libSynapse.so"), sha256=digest(build / "libSynapse.so"),
                 source_prerequisites=fingerprints, status="built_unqualified", default_enabled=False,
                 class_layout_changed=False, final_completion_retirement_unchanged=True,
                 source_patch_sha256=digest(build / "source.patch"),
                 graph_compiler_objects_unchanged=True, hcl_unchanged=True,
                 replaced_runtime_unit="stream_compute_scal.cpp.o",
                 compilation_sha256=parent_sha)
    baseline.unlink()
    (build / "RESULT.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
