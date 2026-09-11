# SPDX-License-Identifier: Apache-2.0
"""Build V4.1's reused Gaudi kernels and native host Engram gather together."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=root / "build/deepseek_v41")
    parser.add_argument("--output-dir", type=Path, default=root / "vllm_gaudi/lib")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    build, output = args.build_root.resolve(), args.output_dir.resolve()
    subprocess.run([sys.executable, str(root / "tools/build_deepseek_v4.py"), "--build-root", str(build / "native"),
                    "--output-dir", str(output), "--jobs", str(args.jobs)], check=True)
    subprocess.run([sys.executable, "setup.py", "build_ext", "--build-lib", str(output),
                    "--build-temp", str(build / "host")], cwd=root / "csrc/deepseek_v41", check=True)
    files = list((root / "csrc/deepseek_v41").glob("*.cpp")) + list((root / "csrc/deepseek_v41").glob("*.py"))
    native = json.loads((output / "deepseek_v4_build.json").read_text())
    for path in files:
        native["sources"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in output.glob("dsv41_host_gather*.so"):
        native["binaries"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    native.update(prepared_layout_version=2, host_gather_abi_version=1)
    (output / "deepseek_v41_build.json").write_text(json.dumps(native, indent=2) + "\n")


if __name__ == "__main__":
    main()
