# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[3] / 'tools' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_positive_tail_excess_is_not_mean_minus_median():
    result = load('report_deepseek_v41_host_jitter').distribution([1, 10, 10, 10, 20])
    assert result['mean_ms'] == 10.2
    assert result['median_ms'] == 10
    assert result['positive_excess_above_median_ms_per_token'] == 2
    assert result['over_15ms_percent'] == 20


def test_spike_association_uses_absolute_request_clock():
    module = load('report_deepseek_v41_host_jitter')
    result = dict(request_start_ns=10_000_000_000, client_inter_token_latency=dict(valid=True, samples_ms=[20]),
                  sse_events=[dict(arrival_s=1), dict(arrival_s=1.02)])
    host = [dict(wall_ns=11_010_000_000, interval_s=1, load1=50,
                 cpu_pressure='some avg10=0.00 avg60=0.00 avg300=0.00 total=10')]
    report = module.report(result, host)
    assert report['spikes'][0]['host_sample_indices'] == [0]
    assert report['host']['all_recorded_cpu_some_avg10_below_1']
    assert report['host']['load1_is_not_an_acceptance_gate']


def test_coalesced_arrivals_rejected():
    with pytest.raises(ValueError, match='Coalesced'):
        load('report_deepseek_v41_host_jitter').report(dict(client_inter_token_latency=dict(valid=False)), [])


def test_proc_stat_comm_with_spaces_and_parentheses():
    fields = ['0'] * 50
    fields[0], fields[1], fields[11], fields[12], fields[19], fields[36] = 'S', '2', '3', '4', '5', '43'
    result = load('monitor_deepseek_v41_host').process_stat('1 (worker (rank 0)) ' + ' '.join(fields))
    assert result == dict(ppid=2, ticks=7, start_ticks=5, processor=43, state='S')


def test_cpu_accounting_excludes_iowait_from_busy():
    module = load('monitor_deepseek_v41_host')
    a = dict(monotonic_ns=0, cpu_ticks={'10': [0] * 8}, system_context_switches=10, threads={}, processes={})
    b = dict(monotonic_ns=1_000_000_000, wall_ns=2_000_000_000, load=[50, 40, 30],
             cpu_pressure='some avg10=0.00', cpu_ticks={'10': [20, 0, 10, 60, 10, 0, 0, 0]},
             system_context_switches=30, threads={}, processes={})
    result = module.interval(a, b, 100)
    assert result['cpus']['10']['busy_percent'] == 30
    assert result['cpus']['10']['iowait_percent'] == 10
    assert result['system_context_switches_delta'] == 20
