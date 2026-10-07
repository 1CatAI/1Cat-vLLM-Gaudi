# SPDX-License-Identifier: Apache-2.0
"""CPU accounting and legacy explicit-scope export, without an HPU capture."""
import gzip
import json
from types import SimpleNamespace

import pytest

from vllm_gaudi.ops import deepseek_v41_native_trace as trace


def test_scope_cpu_accounting_survives_failure(monkeypatch):
    recorder = SimpleNamespace(scope_events=[])
    monkeypatch.setattr(trace, '_active', True)
    monkeypatch.setattr(trace, '_scope_recorder', recorder)
    with pytest.raises(ValueError, match='test'), trace.scope('entry'):
        sum(range(10000))
        raise ValueError('test')
    start, end, tid, name, detail = recorder.scope_events[0]
    assert end >= start and tid > 0 and name == 'entry'
    assert detail['thread_cpu_ns'] > 0
    assert detail['thread_nonrunning_ns'] >= 0
    assert detail['voluntary_switches'] >= 0 and detail['involuntary_switches'] >= 0


def test_export_accepts_legacy_and_accounted_scopes(tmp_path):
    recorder = trace.NativeTrace(tmp_path, scope_only=True)
    recorder.clock_samples = [dict(monotonic_raw_before_ns=10**9, monotonic_raw_after_ns=10**9),
                              dict(monotonic_raw_before_ns=2*10**9, monotonic_raw_after_ns=2*10**9)]
    recorder.scope_events = [(10**9, 10**9+1000, 7, 'legacy'),
                             (10**9, 10**9+2000, 7, 'accounted', {'thread_cpu_ns': 1000})]
    recorder._export_scopes()
    with gzip.open(next(tmp_path.glob('*.json.gz')), 'rt') as stream:
        events = json.load(stream)['traceEvents']
    assert events[0]['name'] == 'legacy' and 'args' not in events[0]
    assert events[1]['args']['thread_cpu_ns'] == 1000
