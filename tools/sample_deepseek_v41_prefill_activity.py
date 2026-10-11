# SPDX-License-Identifier: Apache-2.0
"""Observe an existing API prefill without attaching a profiler to its workers.

CPU clocks and sampled utilization diagnose host pressure; they do not replace
kernel durations or establish a device idle-time partition.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--no-device-sampling', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    body = json.loads(args.request.read_text())
    body['cache_salt'] = 'prefill-host-activity-' + str(time.time_ns())
    body['max_tokens'] = 1
    request = args.output / 'request.json'
    request.write_text(json.dumps(body, ensure_ascii=False) + '\n')
    workers = [int(value) for value in args.workers.split(',')]
    ticks = os.sysconf('SC_CLK_TCK')

    def snapshot(pid):
        rows = {}
        for task in Path(f'/proc/{pid}/task').iterdir():
            try:
                raw = (task / 'stat').read_text()
                after = raw[raw.rindex(')') + 2:].split()
                rows[int(task.name)] = {
                    'name': raw[raw.index('(') + 1:raw.rindex(')')],
                    'cpu_ticks': int(after[11]) + int(after[12])
                }
            except (FileNotFoundError, ProcessLookupError):
                pass
        return rows

    before = {pid: snapshot(pid) for pid in workers}
    began = time.perf_counter()
    stop = threading.Event()
    loads = []

    def observe():
        while not stop.is_set():
            started = time.perf_counter()
            text = subprocess.check_output(['hl-smi', '-Q', 'module_id,utilization.aip', '-f', 'csv,noheader'],
                                           text=True)
            loads.append(
                dict(start_s=started - began,
                     end_s=time.perf_counter() - began,
                     modules={
                         int(line.split(',')[0]): float(line.split(',')[1].strip().rstrip('%'))
                         for line in text.splitlines()
                     }))
            stop.wait(.25)

    thread = threading.Thread(target=observe)
    if not args.no_device_sampling:
        thread.start()
    waiting = {pid: Counter() for pid in workers}
    command = [
        args.python,
        str(Path(__file__).with_name('qualify_deepseek_v41_request.py')),
        str(request),
        str(args.output / 'request'), '--url', 'http://127.0.0.1:18552', '--api-key-file',
        str(args.key_file), '--prefill-only'
    ]
    env = dict(os.environ)
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(name, None)
    with (args.output / 'client.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            for pid in workers:
                waiting[pid][Path(f'/proc/{pid}/wchan').read_text().strip()] += 1
            time.sleep(.05)
    stop.set()
    if not args.no_device_sampling:
        thread.join()
    after = {pid: snapshot(pid) for pid in workers}
    report = dict(scope=__doc__,
                  elapsed_s=time.perf_counter() - began,
                  client_exit=process.returncode,
                  utilization_samples=loads,
                  workers={})
    for pid in workers:
        rows = []
        for tid, value in after[pid].items():
            delta = value['cpu_ticks'] - before[pid].get(tid, {'cpu_ticks': 0})['cpu_ticks']
            rows.append(dict(tid=tid, name=value['name'], cpu_ms=delta / ticks * 1000))
        report['workers'][pid] = dict(tasks=sorted(rows, key=lambda value: -value['cpu_ms']),
                                      main_wait_samples=dict(waiting[pid]))
    (args.output / 'activity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(exit=process.returncode, elapsed_s=report['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
