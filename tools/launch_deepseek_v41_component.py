# SPDX-License-Identifier: Apache-2.0
"""Launch a native component from a maintained profile, with an owned device lease."""
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
from deepseek_v41_component_paths import component_scratch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-root', type=Path, help='Separate immutable source storage; logs and graphs remain in output')
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-dir', type=Path, required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--min-host-free-gib',type=float,default=64,help='Observed host headroom before loading; no reservation')
    parser.add_argument('--recipe-cache-dir',type=Path,help='Machine-local rebuildable cache, separate from SSD evidence')
    parser.add_argument('--ipc-tmp-root', type=Path, help='Short SSD directory for Unix-domain IPC sockets')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--modules', help='Optional comma-separated modules to lease')
    parser.add_argument('--device-poll-seconds', type=float, default=5,
                        help='Recheck free ownership/locks this often while queued; no reservation of occupied cards')
    parser.add_argument('--lock-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'locks')
    args = parser.parse_args()
    if not 0.25 <= args.device_poll_seconds <= 60:
        parser.error('--device-poll-seconds must be between0.25 and60')
    root = Path(__file__).resolve().parents[1]
    case = args.output.resolve()
    case.mkdir(parents=True, exist_ok=True)
    if os.statvfs(case).f_favail < 4096:
        raise RuntimeError('Component output filesystem needs at least 4096 free inodes before launch')
    frozen=(args.snapshot_root.resolve()/case.name) if args.snapshot_root else case/'execution-source'
    if frozen.exists():
        raise RuntimeError('Preserve an existing execution snapshot; use a fresh component case')
    frozen.mkdir(parents=True)
    for package in ('tools','vllm_gaudi','flashinfer_gaudi'):
        shutil.copytree(root/package,frozen/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source_hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in frozen.rglob('*.py')}
    (case/'execution-sources.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    profile = json.loads(args.template.read_text())
    # Artifact paths are immutable, even when the old run built them inside
    # its output directory. Rewrite only our explicitly assigned writable
    # paths below; a blanket replacement silently relocates the native bridge.
    environment = dict(profile['environment'])
    native = args.native_dir.resolve()
    ops = native / 'hpu_dsv4_sparse_attn_pt2.cpython-312-x86_64-linux-gnu.so'
    kernels = native / 'libdeepseek_v4_gaudi2_kernels.so'
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ops, kernels)}
    bridge = environment.get('VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE')
    if bridge:
        bridge = Path(bridge)
        if not bridge.is_file():
            raise RuntimeError(f'Pinned native bridge absent before device lease: {bridge}')
        hashes[str(bridge)] = hashlib.sha256(bridge.read_bytes()).hexdigest()
    scratch = component_scratch(case, profile['environment'].get('TMPDIR'), args.ipc_tmp_root)
    scratch.mkdir(parents=True,exist_ok=True)
    recipe_dir=args.recipe_cache_dir.resolve() if args.recipe_cache_dir else case/'recipes'
    recipe_dir.mkdir(parents=True,exist_ok=True)
    environment.update(PYTHONPATH=str(frozen),GC_KERNEL_PATH=str(kernels), VLLM_HPU_DSV4_TPC_OP_LIBRARY=str(ops),
                       VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=str(native),
                       TMPDIR=str(scratch), HABANA_LOGS=str(case/'habana_logs'),
                       GRAPH_VISUALIZATION_DIR=str(case/'graphs'),
                       PT_HPU_RECIPE_CACHE_CONFIG=f'{recipe_dir},false,8192')
    module_file = root.joinpath(*args.module.split('.')).with_suffix('.py')
    hashes[str(module_file)] = hashlib.sha256(module_file.read_bytes()).hexdigest()
    if not (args.sidecar / 'manifest.json').is_file():
        raise RuntimeError('Dense sidecar manifest is absent; fail before opening devices')
    command = profile['command'][:]
    for flag, value in (('--module', args.module), ('--output', str(case)), ('--sidecar', str(args.sidecar.resolve()))):
        command[command.index(flag) + 1] = value
    command = [f'--master_port={args.port}' if v.startswith('--master_port=') else v for v in command]
    for key in ('TMPDIR', 'HABANA_LOGS', 'GRAPH_VISUALIZATION_DIR'):
        if environment.get(key):
            Path(environment[key]).mkdir(parents=True, exist_ok=True)
    modules = None if args.modules is None else tuple(int(v) for v in args.modules.split(','))
    wait_for_host_memory(case/'host-memory-availability.json',minimum_gib=args.min_host_free_gib)
    # DUMP_* values are paths in the bridge, including the string "0".
    environment = {k: v for k, v in environment.items() if 'DUMP' not in k}
    child_environment = {k: v for k, v in os.environ.items() if 'DUMP' not in k}
    with lease_free_modules(case / 'devices.json', lock_dir=args.lock_dir, modules=modules,
                            poll_s=args.device_poll_seconds) as (selected, load):
        environment['HABANA_VISIBLE_MODULES'] = ','.join(map(str, selected))
        (case / 'launch.json').write_text(json.dumps(dict(command=command, environment=environment,
                                                        working_directory=str(frozen)), indent=2) + '\n')
        child_environment.update(environment)
        with (case / 'component.log').open('w') as log:
            process = subprocess.Popen(command, cwd=frozen, env=child_environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (case / 'process.json').write_text(json.dumps(dict(pid=process.pid, pgid=process.pid,
                modules=selected, start=time.time(), competing_load=load,
                cpu_pressure=Path('/proc/pressure/cpu').read_text(), fingerprints=hashes,
                execution_source=str(frozen), lock_directory=str(args.lock_dir)), indent=2) + '\n')
            previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            owned = {}
            try:
                result = wait_for_owned_process(process, owned, record=case/'owned-processes.json')
            finally:
                signal.signal(signal.SIGTERM, previous)
                retire_process_group(process, owned=owned)
            raise SystemExit(result)



if __name__ == '__main__':
    main()
