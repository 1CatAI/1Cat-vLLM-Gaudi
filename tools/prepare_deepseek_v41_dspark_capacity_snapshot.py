# SPDX-License-Identifier: Apache-2.0
"""Freeze the private large-C6 capability experiment without editing C1 core."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    source = args.output.resolve() / "source"
    source.mkdir(parents=True, exist_ok=False)
    for pattern in ("vllm_gaudi/**/*.py", "vllm_gaudi/**/*.txt", "flashinfer_gaudi/**/*.py",
                    "flashinfer_gaudi/**/*.json", "tools/*deepseek_v41*.py"):
        for path in workspace.glob(pattern):
            target = source / path.relative_to(workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    name = "vllm_gaudi/distributed/tp2_fused_ar_norm.py"
    shared = workspace / name
    path = source / name
    original = shared.read_text()
    start = original.index("def _tp_peer_allgather_impl(")
    stop = original.index("\ndef _tp_peer_allgather_scheduled_impl", start)
    section = original[start:stop]
    old = "    bridge, backend, _ = _resolve_runtime()\n"
    new = old + ('    maximum = (387840 if tp_size == 4 and\n'
                 '               os.environ.get("VLLM_HPU_DSV41_DSPARK_LARGE_HCCL") == "1" else 32768)\n')
    if new[len(old):] in section and "partial.numel() <= maximum" in section:
        # The serving contract now lives in maintained source. Preserve it
        # rather than applying the historical capability patch twice.
        updated = original
    else:
        if section.count(old) != 1 or section.count("partial.numel() <= 32768") != 1:
            raise ValueError("Shared gather guard changed; review before expanding private capacity")
        updated = original[:start] + section.replace(old, new).replace("partial.numel() <= 32768",
                                                                     "partial.numel() <= maximum") + original[stop:]
    path.write_text(updated)
    patches = list(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                       fromfile=f"shared/{name}", tofile=f"private/{name}"))
    files = [dict(shared_source=str(shared), shared_sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),
                  private_sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    name = "vllm_gaudi/ops/tp2_prepared_plan.py"
    shared, path = workspace / name, source / name
    original = shared.read_text()
    old = "    if count < (1 if tp4 else 2):\n"
    new = ('    probe_context = getattr(_local, "native_context", None) or {}\n'
           '    capacity_probe = (os.environ.get("VLLM_HPU_DSV41_DSPARK_LARGE_HCCL") == "1"\n'
           '                      and getattr(probe_context.get("adapter"), "name", None)\n'
           '                      == "deepseek_v41_dspark_large_vocab")\n'
           '    if count < (1 if tp4 or capacity_probe else 2):\n')
    if original.count(old) != 1:
        raise ValueError("Private single-collective capability eligibility changed")
    updated = original.replace(old, new)
    path.write_text(updated)
    patches.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                       fromfile=f"shared/{name}", tofile=f"private/{name}"))
    files.append(dict(shared_source=str(shared), shared_sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),
                      private_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (args.output / "source.patch").write_text("".join(patches))
    proof = dict(files=files, default_enabled=False,
                 scope="OFF large-HCCL flag extends stock gather capacity and enables the one-collective probe")
    (args.output / "CAPACITY_SNAPSHOT_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(source)


if __name__ == "__main__":
    main()
