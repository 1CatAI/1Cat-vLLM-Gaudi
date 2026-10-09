# SPDX-License-Identifier: Apache-2.0
"""Reserve idle non-serving modules only for an owned formal request."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from run_deepseek_v41 import acquire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--modules', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = args.run / 'process.json'
    deadline = time.monotonic() + 7200
    while not manifest.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Owned launcher did not publish a process manifest')
        time.sleep(1)
    owner = json.loads(manifest.read_text())
    pid, group = owner['pid'], owner['pgid']
    if os.getpgid(pid) != group or owner.get('exit_code') is not None:
        raise RuntimeError('Owned service must be running')
    serving = {item['module'] for item in owner['modules']}
    modules = [int(item) for item in args.modules.split(',')]
    if serving.intersection(modules) or len(set(modules)) != len(modules):
        raise ValueError('Reserve distinct modules outside the serving group')
    selected, locks = None, []
    evidence = args.run / 'formal-machine-lease.json'
    record = dict(owner_pid=pid, owner_pgid=group, guard_pid=os.getpid(), requested=modules,
                  status='acquiring unused-card locks', opens_HPU_context=False)

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        client_path = args.run.with_name(args.run.name + '-client.json')
        while True:
            if not Path(f'/proc/{pid}').exists():
                raise RuntimeError('Owned service exited during warmup')
            client = json.loads(client_path.read_text()) if client_path.exists() else {}
            if client.get('status') == 'waiting for formal machine lease':
                break
            if client.get('status') == 'failed':
                raise RuntimeError('Owned request client failed before timing')
            if time.monotonic() > deadline:
                raise TimeoutError('Owned service did not finish full warmup')
            time.sleep(15)
        while selected is None:
            if not Path(f'/proc/{pid}').exists():
                raise RuntimeError('Owned service exited before reservation')
            selected, locks = acquire(root / 'locks', len(modules), modules,
                                      (Path('/opt/ssd960/ymzx-dsv41-exact-full-stack-v1/locks'),))
            if selected is None:
                time.sleep(15)
        record.update(status='held', acquired_time_ns=time.time_ns(), modules=selected)
        evidence.write_text(json.dumps(record, indent=2) + '\n')
        print(f'Reserved unused modules {modules} for {args.run.name}', flush=True)
        while Path(f'/proc/{pid}').exists():
            result_path = args.run / 'official-16k-eos-formal/result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if (result.get('status') == 'passed' and result.get('profile') is None
                    and result.get('qualification') == 'natural_eos'):
                record['release_reason'] = 'Exclusive formal request complete; companion trace permits other cards'
                break
            current = json.loads(manifest.read_text())
            if (current['pid'], current['pgid']) != (pid, group):
                raise RuntimeError('Owned process identity changed')
            if current.get('exit_code') is not None:
                break
            try:
                state = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state in ('Z', 'X'):
                break
            time.sleep(15)
    finally:
        for stream in locks:
            stream.close()
        record.update(status='released', released_time_ns=time.time_ns())
        evidence.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
