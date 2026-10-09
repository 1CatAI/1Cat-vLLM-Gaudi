# SPDX-License-Identifier: Apache-2.0
"""Private stock-HCCL C6 capacity guard; no new transport or shared writes."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--baseline-abi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = json.loads(args.commands.read_text())
    abi = json.loads(args.baseline_abi.read_text())
    baseline = abi["native_dependencies"]["hcl"]
    if digest(Path(baseline["path"])) != baseline["sha256"]:
        raise ValueError("Baseline stock HCL changed")
    compile_command = commands["compile"][3:]
    source = Path(compile_command[-1])
    original = source.read_text()
    start = original.index('extern "C" hcclResult_t HCCL_API_CALL hcclTp4NativeGraphCreate(')
    stop = original.index('extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphCapture(', start)
    section = original[start:stop]
    old = "reduceOp != hcclSum || count == 0 || count > 32768 || phase < 0 || phase > 2 ||"
    if section.count(old) != 1:
        raise ValueError("Stock HCL capacity guard changed")
    replacement = ( '{\n    const char* large = std::getenv("VLLM_HPU_DSV41_DSPARK_LARGE_HCCL");\n'
                              '    const size_t maximum = phase == 2 && large && std::strcmp(large, "1") == 0\n'
                              '        ? 387840 : 32768;\n    if (!graphHandle')
    section = section.replace("{\n    if (!graphHandle", replacement, 1)
    updated = original[:start] + section.replace(old, old.replace("count > 32768", "count > maximum")) + original[stop:]
    private = output / "hccl.cpp"
    private.write_text(updated)
    (output / "stock-capacity.patch").write_text("".join(
        difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                             fromfile=str(source), tofile=str(private))))
    old_object = Path(compile_command[compile_command.index("-o") + 1])
    compile_command[compile_command.index("-o") + 1] = str(output / "hccl.cpp.o")
    compile_command[-1] = str(private)
    cwd = Path(commands["cwd"])
    link_command = commands["link"][3:]
    link_command[link_command.index("-o") + 1] = str(output / "libhcl.so")
    linked_inputs = []
    for index, item in enumerate(link_command):
        if item == str(old_object):
            link_command[index] = str(output / "hccl.cpp.o")
        elif item.endswith((".o", ".a", ".so")):
            path = Path(item)
            if not path.is_absolute():
                path = cwd / path
            if path.is_file():
                link_command[index] = str(path)
                linked_inputs.append(dict(path=str(path), sha256=digest(path)))
    proof = dict(shared_source=str(source), shared_source_sha256=digest(source),
                 private_source_sha256=digest(private), baseline=baseline, link_inputs=linked_inputs,
                 commands=dict(compile=compile_command, link=link_command),
                 default_enabled=False, transport="unchanged stock HCL AllGather phase2", credited_ms=0)
    (output / "BUILD_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")
    for name, command in (("compile", compile_command), ("link", link_command)):
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof["binary_sha256"] = digest(output / "libhcl.so")
    if digest(source) != proof["shared_source_sha256"]:
        raise RuntimeError("Shared HCL source changed during private build")
    (output / "BUILD_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
