# SPDX-License-Identifier: Apache-2.0
"""Build the standalone exact CPU layout helper; no HPU runtime is linked."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "csrc/deepseek_v41_startup_pack.cpp"
    args.output.mkdir(parents=True, exist_ok=True)
    library = args.output / "libdeepseek_v41_startup_pack.so"
    if library.exists():
        raise ValueError("Build requires an independent empty library output")
    command = ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", str(source), "-o", str(library)]
    subprocess.run(command, check=True)
    record = dict(schema=1,
                  abi=1,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  command=command,
                  cpu_dispatch="AVX512BW when supported, otherwise scalar",
                  hpu_dependencies=False)
    library.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(str(library))


if __name__ == "__main__":
    main()
