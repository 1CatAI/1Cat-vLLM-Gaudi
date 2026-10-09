# SPDX-License-Identifier: Apache-2.0
"""Fixed leased DSpark official request/trace batch, with automatic retirement."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def check_draft_coverage_policy(environment, *, trace_only=False):
    """Do not readmit the unchanged draft path rejected by whole requests."""
    def enabled(name):
        return str(environment.get('VLLM_HPU_DSV41_DSPARK_' + name, '0')).lower() in ('1', 'true')

    if (not trace_only and enabled('GLOBAL_BOUNDED_SAMPLING')
            and not enabled('WEIGHTED_DRAFT_NUCLEUS') and not enabled('EXACT_DRAFT_SAMPLING')):
        raise ValueError('Unchanged K64 draft coverage failed complete official requests270/446; '
                         'retain the qualified weighted draft probability path')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--modules', default='2,6,7,3')
    parser.add_argument('--trace-only', action='store_true',
                        help='Recover a companion trace without repeating formal timing')
    parser.add_argument('--source-snapshot', type=Path)
    parser.add_argument('--recipe-cache-dir', type=Path)
    parser.add_argument('--baseline-output', type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    storage = Path('/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving')
    notes = workspace / 'evidence/20260930_tp4_dspark'
    profile = args.profile or notes / 'runtime-official-tp4-dspark-current.json'
    run = args.run.resolve()
    ledger = json.loads((notes / 'DSPARK_PENDING_E2E_LEDGER.json').read_text())
    baseline_decision = json.loads((notes / ledger['active_baseline']['decision']).read_text())
    baseline_output = (args.baseline_output or Path(baseline_decision['source']) /
                       ledger['active_baseline']['name'] / 'token_ids.json')
    if not baseline_output.is_file():
        raise FileNotFoundError(f'Active baseline output is unavailable: {baseline_output}')
    candidate_environment = json.loads(profile.read_text())['environment']
    check_draft_coverage_policy(candidate_environment, trace_only=args.trace_only)
    launch = [sys.executable, 'tools/run_deepseek_v41.py', '--devices', '4', '--modules', args.modules,
              '--cpu-conflict-policy', 'relocate-or-measure', '--preferred-cpus', '10-19,38-47',
              '--min-host-available-gib', '350', '--lock-dir', str(workspace.parent / 'locks'),
              '--secondary-lock-dir', '/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks',
              '--runtime-profile', str(profile.resolve()), '--engine-source',
              str(workspace.parent / 'builds/dsv41-tp4-dspark-main-v1/engine'),
              '--raw-profiler',
              '--recipe-cache-dir', str(args.recipe_cache_dir or storage / (run.name + '-cache'))]
    if args.source_snapshot:
        launch += ['--source-snapshot', str(args.source_snapshot.resolve())]
    launch += [str(run), '--',
              sys.executable, '-m', 'vllm_gaudi.entrypoints.deepseek_v41',
              '/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2',
              '--tensor-parallel-size', '4', '--pipeline-parallel-size', '1', '--host', '127.0.0.1',
              '--port', str(args.port), '--max-model-len', '1048576', '--max-num-batched-tokens', '8192',
              '--max-num-seqs', '32', '--block-size', '128', '--gpu-memory-utilization', '1.0',
              '--num-gpu-blocks-override', '8193', '--additional-config',
              json.dumps({'dsv41_native_warmup_searches': [512, 1024, 32768, 65536]})]
    run.parent.mkdir(parents=True, exist_ok=True)
    if run.exists():
        raise FileExistsError(run)
    with run.with_name(run.name + '-launcher.log').open('xb') as log:
        launcher = subprocess.Popen(launch, cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
        guard, guard_log = None, None
        if not args.trace_only:
            unused = sorted(set(range(8)) - set(map(int, args.modules.split(','))))
            guard_log = run.with_name(run.name + '-machine-lease.log').open('xb')
            guard = subprocess.Popen(
                [sys.executable, 'tools/hold_deepseek_v41_formal_machine.py', '--run', str(run),
                 '--modules', ','.join(map(str, unused))], cwd=workspace,
                stdout=guard_log, stderr=subprocess.STDOUT)
        client = [sys.executable, 'tools/qualify_deepseek_v41_dspark_batch.py', '--run', str(run),
                  '--launcher-pid', str(launcher.pid), '--request', str(notes / 'official-seed42-16k-request.json'),
                  '--baseline-output', str(baseline_output.resolve()),
                  '--url', f'http://127.0.0.1:{args.port}', '--modules', args.modules]
        if args.trace_only:
            client += ['--trace-only']
        else:
            client += ['--require-machine-lease']
        try:
            subprocess.run(client, cwd=workspace, check=True)
        finally:
            # The client verifies the owned process manifest before retiring
            # the service group; the launcher releases card locks afterward.
            launcher.wait()
            if guard is not None:
                try:
                    guard.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    guard.terminate()
                    guard.wait(timeout=30)
                guard_log.close()


if __name__ == '__main__':
    main()
