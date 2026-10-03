# SPDX-License-Identifier: Apache-2.0
"""Synthetic clocks prove alignment and prevent interpreting missing CPU data."""
import importlib.util
from pathlib import Path

import pytest

p = Path(__file__).resolve().parents[3] / 'tools/analyze_deepseek_v41_host_entry.py'
spec = importlib.util.spec_from_file_location('host_entry', p)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def document(rank):
    return dict(clockDomain='CLOCK_MONOTONIC_RAW', baseTimeNanoseconds=10**9,
                traceEvents=[dict(ph='X', name='v41::worker_commit::PP0::decode::P8::C1::emit1',
                                  tid=7, ts=0, dur=100),
                             dict(ph='X', name='phase=StageReplay.begin_segmented_from_input_ids', tid=7,
                                  ts=10+rank*10, dur=5, args={'thread_cpu_ns': 1000}),
                             dict(ph='X', name='unrelated helper', tid=11, ts=12, dur=3)])


def test_four_worker_host_alignment_keeps_cpu_distinct_from_device():
    ranks = [entry.host_entries(document(rank)) for rank in range(4)]
    result = entry.compare(ranks)
    assert result['per_token'] == [dict(position=9, prefix_host_call_skew_us=30.)]
    assert result['device_stage_skew_not_measured'] is True
    phases = ranks[0][0]['phases']
    assert len(phases) == 2 and phases[0]['thread_cpu_us'] is None
    assert phases[1]['thread_cpu_us'] == 1.
    assert ranks[0][0]['prefix_call_start_raw_us'] == 1_000_010.


def test_wrong_clock_and_incomplete_card_set_are_rejected():
    data = document(0)
    data['clockDomain'] = 'wall'
    with pytest.raises(ValueError, match='monotonic'):
        entry.host_entries(data)
    with pytest.raises(ValueError, match='four'):
        entry.compare([[]])
