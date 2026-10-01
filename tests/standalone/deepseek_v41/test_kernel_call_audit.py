# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[3] / 'tools/audit_deepseek_v41_kernel_switches.py'
spec = importlib.util.spec_from_file_location('kernel_call_audit', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_cached_recipe_occurrences_keep_distinct_runtime_slots():
    slots = [(0, dict(name='first')), (0, dict(name='third'))]
    frames = [[[10., 'cached', 0], [20., 'cached', 1]], [[30., 'cached', 2], [40., 'cached', 3]]]
    entries = audit.recipe_occurrences(frames, slots)['cached']
    starts = [entry['start_us'] for entry in entries]
    first = audit.packet_occurrence(11., 12., entries, starts)
    third = audit.packet_occurrence(21., 22., entries, starts)
    assert first['slot'] == 0 and third['slot'] == 1
    assert audit.packet_occurrence(31., 32., entries, starts)['token'] == 1
    assert audit.packet_occurrence(9., 11., entries, starts) is None
    assert audit.packet_occurrence(19., 21., entries, starts) is None


def test_duplicate_or_missing_stage_anchors_are_rejected():
    with pytest.raises(ValueError, match='count'):
        audit.recipe_occurrences([[[1., 'cached']]], [])
    with pytest.raises(ValueError, match='unique'):
        audit.recipe_occurrences([[[1., 'cached'], [1., 'cached']]], [(0, dict(name='a')), (0, dict(name='b'))])
