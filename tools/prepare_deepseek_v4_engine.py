# SPDX-License-Identifier: Apache-2.0
"""Apply the version-pinned engine source patch before installation, never at runtime."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


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
    if not patches:
        parser.error("Engine patch series is missing from this source distribution")
    # Later patches also modify files introduced earlier in the series. Build
    # the expected tree in a temporary index so validation and idempotence do
    # not depend on reversing overlapping patches against the working tree.
    with tempfile.TemporaryDirectory(prefix="deepseek-v4-patch-check-") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        subprocess.run(["git", "read-tree", "HEAD"], cwd=engine, env=env, check=True)
        for patch in patches:
            subprocess.run(["git", "apply", "--cached", str(patch)], cwd=engine, env=env, check=True)
        affected = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "-z", "HEAD"], cwd=engine, env=env).decode().split("\0")
        already_applied = subprocess.run(
            ["git", "diff", "--quiet", "--", *filter(None, affected)], cwd=engine, env=env).returncode == 0
    if already_applied:
        print("DeepSeek V4 engine source patches are already applied")
        return
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=engine, text=True).strip():
        parser.error("Expected a clean engine checkout; existing changes are left untouched")
    for patch in patches:
        subprocess.run(["git", "apply", str(patch)], cwd=engine, check=True)


if __name__ == "__main__":
    main()
