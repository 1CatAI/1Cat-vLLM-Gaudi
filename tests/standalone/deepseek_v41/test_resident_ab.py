# SPDX-License-Identifier: Apache-2.0
"""A/B decision and immutable allocation ownership contracts, independent of HPU noise."""
import importlib.util
from pathlib import Path
from types import MethodType

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / 'tools/deepseek_v41_resident_ab.py'
_spec = importlib.util.spec_from_file_location('resident_ab', _path)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def periods(a, b):
    return [dict(arm=arm, token_intervals_ms=list(a if arm == 'A' else b)) for arm in 'ABABAB']


def test_strict_noise_threshold_and_order():
    a = [4.9, 5.1] * 100
    weak = ab.compare_periods(periods(a, [4.7, 4.9] * 100))
    assert not weak['effective']
    assert weak['validity_threshold_ms'] == pytest.approx(0.4)
    strong = ab.compare_periods(periods(a, [4.3, 4.5] * 100))
    assert strong['effective'] and not strong['formal_gain_credit']
    negative = ab.compare_periods(periods(a, [5.5, 5.7] * 100))
    assert negative['slower'] and not negative['effective']
    with pytest.raises(ValueError, match='ABABAB'):
        ab.compare_periods(periods(a, a)[:-1])
    with pytest.raises(ValueError, match='200'):
        ab.summarize([1.] * 199)


def test_clones_own_modules_but_share_immutable_storage():
    root = torch.nn.Module()
    root.child = torch.nn.Module()
    root.child.register_buffer('weight', torch.arange(8.))
    root.forward = MethodType(lambda self: self.child.weight, root)
    clone = ab.clone_module(root, recursive=True)
    assert clone is not root and clone.child is not root.child
    assert clone.child.weight is root.child.weight
    assert clone.forward.__self__ is clone
    clone.child.weight = torch.ones(8)
    assert torch.equal(root.child.weight, torch.arange(8.))
    assert torch.equal(clone(), torch.ones(8))


def test_only_monotonic_foreign_weight_growth_blocks_timing():
    def sample(own, foreign):
        return dict(modules=[dict(module=0, memory_mib=own), dict(module=2, memory_mib=foreign)])
    assert ab.loading_modules([sample(100, 100), sample(200, 180), sample(300, 260)]) == [2]
    assert ab.loading_modules([sample(100, 100), sample(200, 260), sample(300, 260)]) == [2]
    assert ab.loading_modules([sample(100, 100), sample(200, 80), sample(300, 260)]) == []
    assert ab.loading_modules([sample(100, 100), sample(300, 100), sample(600, 100)]) == []


def test_loading_pool_reset_must_settle_before_measurement():
    def sample(memory):
        return dict(modules=[dict(module=2, memory_mib=memory)])
    assert ab.settling_modules([sample(768), sample(98304), sample(773)]) == [2]
    assert ab.settling_modules([sample(35000), sample(35005), sample(35009)]) == []


def test_cold_compiler_settings_restore_on_success_and_failure():
    class Library:
        def __init__(self):
            self.values = {b'SRAM_SLICER_MAX_CAPACITY_BYTES': b'18446744073709551615',
                           b'ENABLE_PIPELINE_MANAGEMENT': b'true'}

        def synConfigurationGet(self, key, value, size):
            value.value = self.values[key]
            return 0

        def synConfigurationSet(self, key, value):
            self.values[key] = value
            return 0

    lib = Library()
    before = dict(lib.values)
    with ab.compiler_settings({'SRAM_SLICER_MAX_CAPACITY_BYTES': '0'}, lib):
        assert lib.values[b'SRAM_SLICER_MAX_CAPACITY_BYTES'] == b'0'
    assert lib.values == before
    with pytest.raises(ValueError, match='compile failure'):
        with ab.compiler_settings({'ENABLE_PIPELINE_MANAGEMENT': 'false'}, lib):
            assert lib.values[b'ENABLE_PIPELINE_MANAGEMENT'] == b'false'
            raise ValueError('compile failure')
    assert lib.values == before
