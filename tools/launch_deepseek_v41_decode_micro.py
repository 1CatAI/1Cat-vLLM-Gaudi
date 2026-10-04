# SPDX-License-Identifier: Apache-2.0
"""TP4 resident native-chain launcher; no device locks, diagnostics off by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from deepseek_v41_owned_devices import wait_for_free_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installation', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--dense-sidecar', type=Path, required=True)
    parser.add_argument('--dense-config', type=Path, required=True)
    parser.add_argument('--master-port', type=int, default=29689)
    parser.add_argument('--modules', help='Optional four comma-separated module IDs; otherwise use any free four')
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
        raise RuntimeError('Execution snapshot already exists; preserve it and use a fresh measurement case')
    frozen.mkdir()
    for package in ('tools', 'vllm_gaudi'):
        shutil.copytree(root / package, frozen / package,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {str(path.relative_to(frozen)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for package in ('tools', 'vllm_gaudi') for path in (frozen / package).rglob('*.py')}
    (evidence / 'execution-sources.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    for request_path in (evidence / 'control').glob('*.request.json'):
        request = json.loads(request_path.read_text())
        for key in ('factory', 'baseline_factory'):
            if not request.get(key):
                continue
            factory = Path(request[key]).resolve()
            try:
                relative = factory.relative_to(root)
            except ValueError as error:
                raise RuntimeError('Queued factory must belong to the maintained workspace') from error
            request[key] = str(frozen / relative)
            request[key+'_sha256'] = source_hashes[str(relative)]
        request_path.write_text(json.dumps(request, indent=2)+'\n')
    environment = dict(os.environ)
    environment.update(runtime['environment'])
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
    (evidence / 'recipes').mkdir(exist_ok=True)
    environment.update(TMPDIR=str(scratch),
                       VLLM_HPU_DSV4_WORKER_CPUS='10,15,38,43',
                       VLLM_HPU_DSV4_WORKER_HELPER_CPUS='11-14;16-19;39-42;44-47',
                       GLOO_SOCKET_IFNAME='lo', OMP_NUM_THREADS='1', DSV41_RUN_EVIDENCE=str(evidence),
                       DSV41_RUNTIME_PROFILE=str(args.runtime_profile.resolve()),
                       HABANA_LOGS=str(evidence/'habana_logs'),
                       PT_HPU_RECIPE_CACHE_CONFIG=f'{evidence}/recipes,false,8192')
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
    os.sched_setaffinity(0, set(range(10, 20)) | set(range(38, 48)))
    modules = None if args.modules is None else tuple(int(value) for value in args.modules.split(','))
    selected, load = wait_for_free_modules(evidence/'device-availability.json', modules=modules)
    environment['HABANA_VISIBLE_MODULES'] = ','.join(map(str, selected))
    with (evidence/'component.log').open('w') as log:
        process = subprocess.Popen(command, cwd=frozen, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        (evidence/'process.json').write_text(json.dumps(dict(
            pid=process.pid, pgid=process.pid, command=command, started=time.time(), cards=load,
            modules=selected, resource_policy='use observed free modules without locks; wait if unavailable',
            execution_source=str(frozen), maintained_workspace=str(root),
            cpu_pressure=Path('/proc/pressure/cpu').read_text(), environment=environment,
            protocol='same-process native ABABAB; three consistent device rounds; no IQR veto'), indent=2)+'\n')
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
