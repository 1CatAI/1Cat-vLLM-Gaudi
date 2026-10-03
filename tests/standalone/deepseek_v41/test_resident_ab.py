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


def test_swa_candidate_never_changes_reference_dispatch_or_allocations():
    stage = torch.nn.Module()
    stage.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for block, ratio in zip(stage.layers, (0, 2)):
        block.attention = torch.nn.Module()
        block.attention.ratio = ratio
        block.attention.decode_swa_packed = False
        block.attention.register_buffer('swa', torch.zeros(256, 528, dtype=torch.uint8))
    candidate = ab.make_swa_stage(stage)
    assert candidate.layers[0].attention.decode_swa_packed
    assert not candidate.layers[1].attention.decode_swa_packed
    for original, cloned in zip(stage.layers, candidate.layers):
        assert not original.attention.decode_swa_packed
        assert original.attention is not cloned.attention
        assert original.attention.swa is cloned.attention.swa


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
            self.values = {
                b'SRAM_SLICER_MAX_CAPACITY_BYTES': b'18446744073709551615',
                b'ENABLE_PIPELINE_MANAGEMENT': b'true'
            }

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
    with pytest.raises(ValueError, match='compile failure'), ab.compiler_settings(
            {'ENABLE_PIPELINE_MANAGEMENT': 'false'}, lib):
        assert lib.values[b'ENABLE_PIPELINE_MANAGEMENT'] == b'false'
        raise ValueError('compile failure')
    assert lib.values == before


def test_norm_candidate_does_not_enable_the_reference_block():
    attention = torch.nn.Module()
    attention.ratio = 0
    attention.decode_swa_packed = False
    block = torch.nn.Module()
    block.attention = attention
    block.decode_attention_norm_quant = False
    stage = torch.nn.Module()
    stage.layers = torch.nn.ModuleList([block])
    candidate = ab.make_attention_norm_stage(stage)
    assert candidate.layers[0].decode_attention_norm_quant
    assert candidate.layers[0].attention.decode_swa_packed
    assert not stage.layers[0].decode_attention_norm_quant
    assert not stage.layers[0].attention.decode_swa_packed


def test_handoff_candidate_keeps_all_reference_dispatch_independent():
    block = torch.nn.Module()
    block.attention = torch.nn.Module()
    block.attention.ratio = 0
    block.attention.decode_swa_packed = False
    block.decode_attention_norm_quant = False
    block.decode_engram_update = False
    block.mhc_interlayer_collapse = False
    block.mhc_interlayer_bf16 = False
    block.register_buffer('residual', torch.zeros(1, 4, 5120))
    stage = torch.nn.Module()
    stage.layers = torch.nn.ModuleList([block])
    candidate = ab.make_handoff_stage(stage)
    cloned = candidate.layers[0]
    for name in ('decode_attention_norm_quant', 'decode_engram_update', 'mhc_interlayer_collapse',
                 'mhc_interlayer_bf16'):
        assert getattr(cloned, name)
        assert not getattr(block, name)
    assert cloned.residual is block.residual
    assert cloned.attention is not block.attention
    assert cloned.attention.decode_swa_packed
    assert not block.attention.decode_swa_packed


def test_mirror_observer_records_keys_without_mutation_and_masks_incomplete_pair():
    from types import SimpleNamespace
    from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4

    packed = pack_fp4(torch.randn(256, 128).bfloat16(), 32)
    mirror = unpack_fp4(packed, 128, 32)
    mirror[128] = 99  # The incomplete pair is invisible to scoring.
    cache = SimpleNamespace(ratio=2, index=packed, index_mirror=mirror)
    selection = SimpleNamespace(indices=torch.tensor([[3, 7, 11]], dtype=torch.int32))
    block = SimpleNamespace(layer=2, attention=SimpleNamespace(owns_index=True, selection=selection))
    shared = SimpleNamespace(sources={'2': cache}, physical_rows=lambda rows, ratio: rows)
    program = SimpleNamespace(shared=shared, stop=16, layers=[block])
    hidden = torch.randn(1, 5120).bfloat16()
    record = ab.observe_mirror_step(program, hidden, 256)
    keys = record['keys']['2']
    assert not keys['visible'][-1] and keys['rows'][-1] == 128
    assert torch.equal(keys['mirror'][keys['visible']], keys['canonical'][keys['visible']])
    assert torch.equal(record['hidden'], hidden) and torch.equal(record['indices']['2'], selection.indices)
    record['hidden'].zero_()
    assert hidden.count_nonzero() > 0
