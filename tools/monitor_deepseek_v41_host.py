# SPDX-License-Identifier: Apache-2.0
"""Record one-second host scheduling evidence without importing the model runtime."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def process_stat(text):
    # comm can contain spaces and parentheses; fields after its final ')' are stable.
    fields = text[text.rfind(')') + 2:].split()
    return dict(ppid=int(fields[1]), ticks=int(fields[11]) + int(fields[12]),
                start_ticks=int(fields[19]), processor=int(fields[36]), state=fields[0])


def process_table(proc):
    table = {}
    for directory in proc.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            item = process_stat((directory / 'stat').read_text())
            item['command'] = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            table[int(directory.name)] = item
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendants(table, roots):
    result = set(roots) & table.keys()
    while True:
        children = {pid for pid, item in table.items() if item['ppid'] in result}
        enlarged = result | children
        if enlarged == result:
            return result
        result = enlarged


def snapshot(proc, cpus, roots):
    stamp = dict(wall_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())
    cpu = {}
    system_switches = None
    for line in (proc / 'stat').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'ctxt':
            system_switches = int(fields[1])
        if fields[0].startswith('cpu') and fields[0][3:].isdecimal() and int(fields[0][3:]) in cpus:
            # guest/guest_nice are already included in user/nice; don't double-count.
            cpu[fields[0][3:]] = list(map(int, fields[1:9]))
    table = process_table(proc)
    threads = {}
    for pid in descendants(table, roots):
        try:
            tasks = list((proc / str(pid) / 'task').iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                item = process_stat((task / 'stat').read_text())
                status = dict(line.split(':', 1) for line in (task / 'status').read_text().splitlines() if ':' in line)
                item.update(pid=pid, tid=int(task.name),
                            allowed_cpus=status.get('Cpus_allowed_list', '').strip(),
                            voluntary=int(status['voluntary_ctxt_switches']),
                            involuntary=int(status['nonvoluntary_ctxt_switches']))
                scheduled = list(map(int, (task / 'schedstat').read_text().split()))
                item.update(runtime_ns=scheduled[0], runqueue_wait_ns=scheduled[1], timeslices=scheduled[2])
                threads[f'{pid}/{task.name}'] = item
            except (OSError, KeyError, ValueError, IndexError):
                continue
    pressure = (proc / 'pressure/cpu').read_text().strip()
    return dict(**stamp, cpu_pressure=pressure, load=list(map(float, (proc / 'loadavg').read_text().split()[:3])),
                cpu_ticks=cpu, system_context_switches=system_switches, threads=threads, processes=table)


def interval(previous, current, hz):
    seconds = (current['monotonic_ns'] - previous['monotonic_ns']) / 1e9
    result = dict(wall_ns=current['wall_ns'], monotonic_ns=current['monotonic_ns'], interval_s=seconds,
                  load1=current['load'][0], load5=current['load'][1], load15=current['load'][2],
                  cpu_pressure=current['cpu_pressure'], cpus={}, threads={})
    for cpu, ticks in current['cpu_ticks'].items():
        if cpu not in previous['cpu_ticks']:
            continue
        delta = [a - b for a, b in zip(ticks, previous['cpu_ticks'][cpu])]
        total = sum(delta)
        if total > 0:
            result['cpus'][cpu] = dict(busy_percent=100 * (total - delta[3] - delta[4]) / total,
                                       iowait_percent=100 * delta[4] / total, ticks_delta=delta)
    result['system_context_switches_delta'] = current['system_context_switches'] - previous['system_context_switches']
    for key, item in current['threads'].items():
        old = previous['threads'].get(key)
        if old is None or old['start_ticks'] != item['start_ticks']:
            continue
        result['threads'][key] = dict(pid=item['pid'], tid=item['tid'], processor=item['processor'],
            allowed_cpus=item['allowed_cpus'], voluntary_delta=item['voluntary'] - old['voluntary'],
            involuntary_delta=item['involuntary'] - old['involuntary'],
            runtime_ns_delta=item['runtime_ns'] - old['runtime_ns'],
            runqueue_wait_ns_delta=item['runqueue_wait_ns'] - old['runqueue_wait_ns'],
            timeslices_delta=item['timeslices'] - old['timeslices'])
    active = []
    for pid, item in current['processes'].items():
        old = previous['processes'].get(pid)
        if old is None or old['start_ticks'] != item['start_ticks']:
            continue
        percent = 100 * (item['ticks'] - old['ticks']) / hz / seconds
        if percent >= 10 or item['state'] == 'D' or any(name in item['command'] for name in ('gnome-shell', 'ToDesk')):
            active.append(dict(pid=pid, cpu_percent=percent, processor=item['processor'], state=item['state'],
                               command=item['command']))
    result['active_processes'] = sorted(active, key=lambda p: -p['cpu_percent'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cpus', default='10,15,38,43')
    parser.add_argument('--root-pids', default='', help='Service launcher PID(s); include all descendant threads')
    parser.add_argument('--duration', type=float, default=60)
    args = parser.parse_args()
    cpus = set(map(int, args.cpus.split(',')))
    roots = set(map(int, args.root_pids.split(','))) if args.root_pids else set()
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    proc = Path('/proc')
    hz = os.sysconf('SC_CLK_TCK')
    previous = snapshot(proc, cpus, roots)
    deadline = time.monotonic() + args.duration
    next_sample = time.monotonic() + 1
    with args.output.open('x') as stream:
        stream.write(json.dumps(dict(type='metadata', cpus=sorted(cpus), root_pids=sorted(roots), hz=hz,
            wall_ns=previous['wall_ns'], monotonic_ns=previous['monotonic_ns'],
            context_switch_scope='per service thread and whole system; per-core switches unavailable',
            schedstat_scope='per-thread scheduler counters; may be zero if kernel schedstats disabled')) + '\n')
        stream.flush()
        while running and time.monotonic() < deadline:
            time.sleep(max(0, next_sample - time.monotonic()))
            current = snapshot(proc, cpus, roots)
            stream.write(json.dumps(interval(previous, current, hz)) + '\n')
            stream.flush()
            previous = current
            next_sample = max(next_sample + 1, time.monotonic())


if __name__ == '__main__':
    main()
