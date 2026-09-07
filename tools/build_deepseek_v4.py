# SPDX-License-Identifier: Apache-2.0
"""Build the in-tree DeepSeek V4 Gaudi2 kernels and Bridge registrations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


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
    build, output = args.build_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-S", str(source), "-B", str(build / "kernels"),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build / "kernels"), "--parallel", str(args.jobs)], check=True)
    kernel = output / "libdeepseek_v4_gaudi2_kernels.so"
    shutil.copy2(build / "kernels" / kernel.name, kernel)
    if not args.kernel_only:
        env = dict(os.environ, MAX_JOBS=str(args.jobs))
        subprocess.run([sys.executable, "setup.py", "build_ext", "--build-lib", str(output),
                        "--build-temp", str(build / "pytorch")], cwd=source / "pytorch", env=env, check=True)
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(source.rglob("*"))
               if path.is_file() and path.suffix in (".py", ".cpp", ".hpp", ".h", ".c", ".txt")}
    binaries = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in output.glob("*.so") if path.name.startswith(("hpu_dsv4_", "libdeepseek_v4_"))}
    manifest = {"sources": sources, "binaries": binaries}
    (output / "deepseek_v4_build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
