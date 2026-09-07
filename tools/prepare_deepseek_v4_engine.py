# SPDX-License-Identifier: Apache-2.0
"""Apply the version-pinned engine source patch before installation, never at runtime."""

import argparse
from pathlib import Path
import subprocess


VLLM_COMMIT = "fe755c88995ad468882517b6c4bdd60138d46a3a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path, help="Clean vLLM checkout at the documented commit")
    args = parser.parse_args()
    engine = args.engine.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
    if head != VLLM_COMMIT:
        parser.error(f"Expected vLLM {VLLM_COMMIT}; refusing to patch a different revision")
    patches = sorted((Path(__file__).resolve().parents[1] / "patches/deepseek_v4").glob("*.patch"))
    command = ["git", "apply", *(str(path) for path in patches)]
    if subprocess.run([*command, "--reverse", "--check"], cwd=engine, capture_output=True).returncode == 0:
        print("DeepSeek V4 engine source patches are already applied")
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=engine, text=True).strip():
        parser.error("Expected a clean engine checkout; existing changes are left untouched")
    subprocess.run([*command, "--check"], cwd=engine, check=True)
    subprocess.run(command, cwd=engine, check=True)


if __name__ == "__main__":
    main()
