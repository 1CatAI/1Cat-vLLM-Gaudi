# SPDX-License-Identifier: Apache-2.0
"""Build the in-tree DeepSeek V4 Gaudi2 kernels and Bridge registrations."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def check_pytorch_source_coverage(directory):
    """Reject source fingerprints that include an unbuilt registration unit."""
    tree = ast.parse((directory / "setup.py").read_text())
    declared = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".cpp")
    }
    missing = sorted(path.name for path in directory.glob("hpu_*.cpp") if path.name not in declared)
    if missing:
        raise RuntimeError("Custom operator sources missing from setup.py: " + ", ".join(missing))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--build-root", type=Path, default=root / "build/deepseek_v4")
    parser.add_argument("--output-dir", type=Path, default=root / "vllm_gaudi/lib")
    parser.add_argument("--kernel-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = root / "csrc/deepseek_v4"
    if not args.kernel_only:
        check_pytorch_source_coverage(source / "pytorch")
    build, output = args.build_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    def fingerprint_sources():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source.rglob("*"))
                if path.is_file() and path.suffix in (".py", ".cpp", ".hpp", ".h", ".c", ".txt")}
    before = fingerprint_sources()
    subprocess.run(["cmake", "-S", str(source), "-B", str(build / "kernels"),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build / "kernels"), "--parallel", str(args.jobs)], check=True)
    if __package__:
        from .audit_deepseek_v41_tpc_loops import audit_object
    else:
        from audit_deepseek_v41_tpc_loops import audit_object
    objects = [build / "kernels" / (path.stem + ".o") for path in sorted((source / "kernels").glob("*.c"))]
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(os.sched_getaffinity(0)))) as pool:
        loop_audit = list(pool.map(audit_object, objects))
    (output / "tpc_loop_audit.json").write_text(json.dumps(loop_audit, indent=2) + "\n")
    if any(item["empty_self_loops"] for item in loop_audit):
        (output / "INVALID_BUILD.json").write_text(json.dumps(dict(empty_self_loop=True), indent=2) + "\n")
        raise RuntimeError("Compiled TPC empty self-loop rejected before native artifact publication")
    kernel = output / "libdeepseek_v4_gaudi2_kernels.so"
    shutil.copy2(build / "kernels" / kernel.name, kernel)
    if not args.kernel_only:
        env = dict(os.environ, MAX_JOBS=str(args.jobs))
        subprocess.run([sys.executable, "setup.py", "build_ext", "--build-lib", str(output),
                        "--build-temp", str(build / "pytorch")], cwd=source / "pytorch", env=env, check=True)
    sources = fingerprint_sources()
    if sources != before:
        changed = sorted(key for key in set(before) | set(sources) if before.get(key) != sources.get(key))
        # A compiler can read old contents then give its object a newer mtime.
        # Force the changed units newer than completed objects before retrying.
        for key in changed:
            if (root/key).exists():
                (root/key).touch()
        (output/"INVALID_BUILD.json").write_text(json.dumps(dict(changed_during_build=changed), indent=2)+"\n")
        raise RuntimeError("Native source changed during compilation; no valid build manifest published")
    binaries = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.glob("*.so") if path.name.startswith(("hpu_dsv4_", "libdeepseek_v4_"))}
    (output / "INVALID_BUILD.json").unlink(missing_ok=True)
    manifest = {"sources": sources, "binaries": binaries}
    (output / "deepseek_v4_build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
