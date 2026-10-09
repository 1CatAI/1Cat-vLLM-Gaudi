# SPDX-License-Identifier: Apache-2.0
"""Reuse cold recipes only across the proven non-compiled repair-state fix.

The new launch keeps its own source/runtime cache identity. Synapse still
checks each recipe key. This copies unchanged compiled artifacts; it neither
replaces live graphs nor relaxes native-library validation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prove_source(before, after):
    replay = "vllm_gaudi/ops/deepseek_v41_draft_replay.py"
    repair = "vllm_gaudi/ops/deepseek_v41_round_repair.py"
    original = (before / replay).read_text()
    old = ('            self.states += tuple(value for name, value in self.repair_frame.named_buffers()\n'
           '                                 if self.full_main or not self.full or not name.startswith("payload_"))')
    new = '            self.states += self.repair_frame.protocol_states(exact_repair=self.full and not self.full_main)'
    if original.count(old) != 1 or original.replace(old, new) != (after / replay).read_text():
        raise ValueError("Replay changes exceed the proven Python-only state ownership correction")
    left, right = ast.parse((before / repair).read_text()), ast.parse((after / repair).read_text())
    frames = [node for node in right.body if isinstance(node, ast.ClassDef) and node.name == "SampledRoundRepairFrame"]
    if len(frames) != 1:
        raise ValueError("Repair frame definition changed")
    methods = [node for node in frames[0].body if isinstance(node, ast.FunctionDef) and node.name == "protocol_states"]
    if len(methods) != 1:
        raise ValueError("Expected the non-compiled state ownership method")
    frames[0].body.remove(methods[0])
    if ast.dump(left) != ast.dump(right):
        raise ValueError("Journal/protocol tensor operations changed; existing recipes cannot be seeded")
    return {replay, repair}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before, after = args.before.resolve(), args.after.resolve()
    old_record = json.loads((before / "process.json").read_text())
    new_record = json.loads((after / "process.json").read_text())
    allowed = prove_source(before / "source", after / "source")
    for name in ("engine_commit", "engine_patch_sha256", "engine_sources_sha256", "command", "modules"):
        if old_record[name] != new_record[name]:
            raise ValueError(f"Changed execution dependency: {name}")
    for name, value in old_record["source_hashes"].items():
        if (name.startswith(("vllm_gaudi/", "flashinfer_gaudi/")) and name not in allowed
                and new_record["source_hashes"].get(name) != value):
            raise ValueError(f"Changed compiled source: {name}")
    old_profile = json.loads((before / "runtime-profile.json").read_text())
    new_profile = json.loads((after / "runtime-profile.json").read_text())
    for group in ("additional_libraries", "configuration_files"):
        if old_profile[group] != new_profile[group]:
            raise ValueError(f"Changed native dependency: {group}")
        for artifact in new_profile[group]:
            if digest(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("Native artifact differs from its frozen profile")
    old_env, new_env = old_profile["environment"].copy(), new_profile["environment"].copy()
    old_env.pop("TMPDIR", None)
    new_env.pop("TMPDIR", None)
    if old_env != new_env:
        raise ValueError("Feature, precision or graph configuration changed")
    source, destination = (before / "recipe_cache").resolve(), (after / "recipe_cache").resolve()
    if source == destination:
        raise ValueError("New source must retain its independent cache namespace")
    copied = identical = bytes_copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix != ".recipe" and not any(p.name.endswith(".recipe_debug_files") for p in path.parents):
            continue
        target = destination / path.relative_to(source)
        if target.exists():
            if digest(target) != digest(path):
                raise ValueError(f"Different existing cache entry; no overwrite: {target.name}")
            identical += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
        bytes_copied += path.stat().st_size
    report = dict(before=str(before), after=str(after), copied_files=copied, identical_files=identical,
                  copied_bytes=bytes_copied, independent_namespace=True, source_proof=sorted(allowed),
                  math_and_native_dependencies_unchanged=True, performance_credit_ms=0)
    (after / "startup-recipe-seed-proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
