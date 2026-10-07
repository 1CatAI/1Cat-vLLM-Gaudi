# SPDX-License-Identifier: Apache-2.0
"""TP4 resident native-chain launcher; owned device leases, diagnostics off by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import signal
import time

from deepseek_v41_owned_devices import lease_free_modules, wait_for_host_memory, retire_process_group, wait_for_owned_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installation', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--dense-sidecar', type=Path, required=True)
    parser.add_argument('--dense-config', type=Path, required=True)
    parser.add_argument('--physical-audit', action='store_true', help='Save final compiler symbol graphs per rank; no runtime profiler or DUMP settings')
    parser.add_argument('--resume', action='store_true', help='Reuse an unchanged execution snapshot after waiting was cancelled; no worktree is created')
    parser.add_argument('--min-host-free-gib',type=float,default=256,help='Observed host headroom before loading; no reservation')
    parser.add_argument('--recipe-cache-dir', type=Path)
    parser.add_argument('--master-port', type=int, default=29689)
    parser.add_argument('--worker-cpus', default='10,15,38,43', help='One main CPU for each TP rank')
    parser.add_argument('--worker-helper-cpus', default='11-14;16-19;39-42;44-47',
                        help='Four semicolon-separated helper CPU ranges')
    parser.add_argument('--modules', help='Optional four comma-separated module IDs; otherwise use any free four')
    parser.add_argument('--lock-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'locks')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    installation, evidence = args.installation.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    runtime = json.loads(args.runtime_profile.read_text())
    binary = Path(runtime['environment']['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE'])
    manifest = json.loads(binary.with_suffix('.abi.json').read_text())
    content = binary.read_bytes()
    if (hashlib.sha256(content).hexdigest() != manifest['binary_sha256']
            or b'configure_dependency_policy' not in content):
        raise RuntimeError('Resident candidates require the fingerprinted dependency-policy bridge before loading')
    # A fixed execution snapshot permits offline development in the existing
    # workspace while a resident plan is preparing or timing. This is an
    # ordinary source copy, without a Git checkout or runtime code injection.
    frozen = evidence / 'execution-source'
    if frozen.exists():
        if not args.resume:
            raise RuntimeError('Execution snapshot already exists; preserve it and use --resume only for unchanged source')
    else:
        frozen.mkdir()
        for package in ('tools', 'vllm_gaudi'):
            shutil.copytree(root / package, frozen / package,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {str(path.relative_to(frozen)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for package in ('tools', 'vllm_gaudi') for path in (frozen / package).rglob('*.py')}
    if args.resume and json.loads((evidence/'execution-sources.json').read_text()) != source_hashes:
        raise RuntimeError('Execution snapshot changed; resumption is unsafe')
    (evidence / 'execution-sources.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    for request_path in (evidence / 'control').glob('*.request.json'):
        request = json.loads(request_path.read_text())
        for key in ('factory', 'baseline_factory'):
            if not request.get(key):
                continue
            factory = Path(request[key]).resolve()
            try:
                relative = factory.relative_to(frozen) if args.resume and factory.is_relative_to(frozen) else factory.relative_to(root)
            except ValueError as error:
                raise RuntimeError('Queued factory must belong to the maintained workspace') from error
            request[key] = str(frozen / relative)
            request[key+'_sha256'] = source_hashes[str(relative)]
        request_path.write_text(json.dumps(request, indent=2)+'\n')
    environment = dict(os.environ)
    environment.update(runtime['environment'])
    environment = {k: v for k, v in environment.items() if 'DUMP' not in k}
    for key in ('PYTHONPATH', 'DUMP_POST_GRAPHS', 'GRAPH_VISUALIZATION', 'GRAPH_VISUALIZATION_DIR',
                'DSV41_RESIDENT_POST_GRAPH', 'HABANA_PROF_CONFIG', 'HABANA_PROFILE', 'HABANA_PROFILE_WRITE_HLTV',
                'VLLM_TORCH_PROFILER_DIR', 'VLLM_HPU_DSV41_RAW_TRACE', 'VLLM_HPU_DSV41_PHASE_TRACE',
                'VLLM_HPU_DSV41_PREFILL_EVENT_TRACE', 'DSV41_OWNED_CAMPAIGN_LEASE_FDS',
                'DSV41_OWNED_CAMPAIGN_LEASE_PID', 'HLS_MODULE_ID'):
        environment.pop(key, None)
    # Keep paths short enough for runtime IPC and create compiler scratch before
    # Bridge starts. Both prevent previously diagnosed cold-start failures.
    scratch = evidence.parent / 'micro-tmp'
    scratch.mkdir(exist_ok=True)
    recipe_dir = args.recipe_cache_dir.resolve() if args.recipe_cache_dir else evidence / 'recipes'
    recipe_dir.mkdir(parents=True, exist_ok=True)
    worker_cpus = [int(cpu) for cpu in args.worker_cpus.split(',')]
    helper_ranges = args.worker_helper_cpus.split(';')
    if len(worker_cpus) != 4 or len(set(worker_cpus)) != 4 or len(helper_ranges) != 4:
        raise ValueError('Four distinct main CPUs and four helper ranges are required')
    affinity = set(worker_cpus)
    for group in helper_ranges:
        for item in group.split(','):
            bounds = [int(value) for value in item.split('-')]
            affinity.update(range(bounds[0], bounds[-1] + 1))
    if not affinity.issubset(os.sched_getaffinity(0)):
        raise ValueError('Worker CPUs are outside the available CPU affinity')
    environment.update(TMPDIR=str(scratch),
                       VLLM_HPU_DSV4_WORKER_CPUS=args.worker_cpus,
                       VLLM_HPU_DSV4_WORKER_HELPER_CPUS=args.worker_helper_cpus,
                       GLOO_SOCKET_IFNAME='lo', OMP_NUM_THREADS='1', DSV41_RUN_EVIDENCE=str(evidence),
                       DSV41_RUNTIME_PROFILE=str(args.runtime_profile.resolve()),
                       HABANA_LOGS=str(evidence/'habana_logs'),
                       PT_HPU_RECIPE_CACHE_CONFIG=f'{recipe_dir},false,8192')
    if args.physical_audit:
        environment['GRAPH_VISUALIZATION'] = '1'
    bindings = evidence / 'bindings.json'
    if not bindings.exists():
        bindings.write_text('{}\n')
    model = json.loads((installation / 'settings.json').read_text())['model']
    command = [str(installation/'venv/bin/python'), '-m', 'torch.distributed.run', '--nproc_per_node=4',
               f'--master_port={args.master_port}', '--module', 'tools.check_deepseek_v41_tp4_continuation', model,
               '--bindings', str(bindings), '--speed-probe', '--production-visible-prefix', '--shared-stage-replay',
               '--resident-ab', '--resident-control-dir', str(evidence/'control'), '--context-tokens', '16384',
               '--max-model-len', '524288', '--ab-dense-sidecar', str(args.dense_sidecar.resolve()),
               '--ab-dense-config', str(args.dense_config.resolve())]
    # Reject incomplete native bundles before any worker opens a device.
    preflight = [str(installation/'venv/bin/python'), '-c',
                 "import runpy, os; from pathlib import Path; import torch; "
                 "import habana_frameworks.torch; "
                 "runpy.run_path('vllm_gaudi/entrypoints/deepseek_v41.py')['prepare_native_libraries'](); "
                 "from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _verify_prepared_runtime; "
                 "p=Path(os.environ['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE']).resolve(); "
                 "_load_bridge(p); _verify_prepared_runtime(p)"]
    subprocess.run(preflight, cwd=frozen, env=environment, check=True)
    os.sched_setaffinity(0, affinity)
    modules = None if args.modules is None else tuple(int(value) for value in args.modules.split(','))
    wait_for_host_memory(evidence/'host-memory-availability.json',minimum_gib=args.min_host_free_gib)
    environment = {k: v for k, v in environment.items() if 'DUMP' not in k}
    with lease_free_modules(evidence/'device-availability.json', lock_dir=args.lock_dir, modules=modules) as (selected, load):
        environment['HABANA_VISIBLE_MODULES'] = ','.join(map(str, selected))
        with (evidence/'component.log').open('w') as log:
            process = subprocess.Popen(command, cwd=frozen, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            (evidence/'process.json').write_text(json.dumps(dict(
                pid=process.pid, pgid=process.pid, command=command, started=time.time(), cards=load,
                modules=selected, resource_policy='lease idle modules through all existing lock aliases',
                execution_source=str(frozen), maintained_workspace=str(root),
                cpu_pressure=Path('/proc/pressure/cpu').read_text(), environment=environment,
                protocol='same-process native ABABAB; three consistent device rounds; no IQR veto'), indent=2)+'\n')
            previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            owned = {}
            try:
                result = wait_for_owned_process(process, owned, record=evidence/'owned-processes.json')
            finally:
                signal.signal(signal.SIGTERM, previous)
                retire_process_group(process, owned=owned)
            raise SystemExit(result)


if __name__ == '__main__':
    main()
