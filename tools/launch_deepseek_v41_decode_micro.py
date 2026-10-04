# SPDX-License-Identifier: Apache-2.0
"""Fixed owned TP4 resident native-chain launcher; diagnostics off by default."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installation', type=Path, required=True)
    parser.add_argument('--runtime-profile', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--dense-sidecar', type=Path, required=True)
    parser.add_argument('--dense-config', type=Path, required=True)
    parser.add_argument('--master-port', type=int, default=29689)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    installation, evidence = args.installation.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    inherited = json.loads(os.environ.get('DSV41_OWNED_CAMPAIGN_LEASE_FDS', '[]'))
    if not inherited or os.getppid() != int(os.environ['DSV41_OWNED_CAMPAIGN_LEASE_PID']):
        raise RuntimeError('Use the campaign module lease; do not launch unreserved workers')
    for fd in inherited:
        path = Path(os.readlink(f'/proc/self/fd/{fd}')).resolve()
        if os.fstat(fd).st_ino != path.stat().st_ino:
            raise RuntimeError('Inherited module reservation changed')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    load = subprocess.check_output(
        ['hl-smi', '-Q', 'module_id,memory.used,utilization.aip', '-f', 'csv,noheader'], text=True)
    for module in (0, 1, 4, 5):
        row = next(line for line in load.splitlines() if line.startswith(f'{module},'))
        if '768 MiB' not in row or '0 %' not in row:
            raise RuntimeError(f'Owned module is not free: {row}')
    runtime = json.loads(args.runtime_profile.read_text())
    binary = Path(runtime['environment']['VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE'])
    manifest = json.loads(binary.with_suffix('.abi.json').read_text())
    content = binary.read_bytes()
    if (hashlib.sha256(content).hexdigest() != manifest['binary_sha256']
            or b'configure_dependency_policy' not in content):
        raise RuntimeError('Resident candidates require the fingerprinted dependency-policy bridge before loading')
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
    environment.update(HABANA_VISIBLE_MODULES='0,1,4,5', TMPDIR=str(scratch),
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
    with (evidence/'component.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        (evidence/'process.json').write_text(json.dumps(dict(
            pid=process.pid, pgid=process.pid, command=command, started=time.time(), cards=load,
            cpu_pressure=Path('/proc/pressure/cpu').read_text(), environment=environment,
            protocol='same-process native ABABAB; three consistent device rounds; no IQR veto'), indent=2)+'\n')
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
