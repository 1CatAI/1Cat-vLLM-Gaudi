# SPDX-License-Identifier: Apache-2.0
"""Lease four free cards for a fixed native prologue/consumer/peer A/B gate."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-layer-count', type=int, choices=(4, 16), default=4)
    parser.add_argument('--modules')
    parser.add_argument('--independent-prologue', action='store_true')
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = workspace.parent
    command = [sys.executable, str(workspace/'tools/run_deepseek_v41.py'), '--devices', '4',
               '--preferred-cpus', '8-47', '--cpu-conflict-policy', 'relocate-or-measure',
               '--lock-dir', str(root/'locks'), '--secondary-lock-dir',
               '/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks', '--runtime-profile', str(args.runtime_profile),
               '--engine-source', str(root/'builds/dsv41-tp4-dspark-main-v1/engine')]
    if args.modules:
        command += ['--modules', args.modules]
    command += [str(args.output), '--', sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', 'tools/check_deepseek_v41_request_c6_batch.py',
                '--prepared', str(args.prepared), '--fixtures', str(args.fixtures),
                '--candidate', 'fp8_prologue', '--group', '5', '--real-layer-count', str(args.real_layer_count),
                '--samples', '3']
    if args.independent_prologue:
        command += ['--independent-prologue']
    subprocess.run(command, cwd=workspace, check=True)


if __name__ == '__main__':
    main()
