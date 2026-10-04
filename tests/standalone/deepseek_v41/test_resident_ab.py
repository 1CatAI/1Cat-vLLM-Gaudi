# SPDX-License-Identifier: Apache-2.0
"""A/B decision and immutable allocation ownership contracts, independent of HPU noise."""
import importlib.util
from pathlib import Path
from types import MethodType
import hashlib
import json

import pytest
import torch

_path = Path(__file__).resolve().parents[3] / 'tools/deepseek_v41_resident_ab.py'
_spec = importlib.util.spec_from_file_location('resident_ab', _path)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def test_physical_audit_cannot_overwrite_the_template_namespace(tmp_path):
    original = dict(GeneralSettings=dict(values={
        'outdir': dict(value='/old/formal/evidence'), 'session': dict(value='formal'),
        'addPid': dict(value=False)}), Plugins=[dict(name='HwTrace', enable=True,
        values=dict(parseOptions=dict(skipParse=dict(value=False))))])
    template = tmp_path / 'template.json'
    template.write_text(json.dumps(original))
    changed = ab.isolated_trace_config(template, tmp_path/'audit')
    assert json.loads(template.read_text()) == original
    assert changed['GeneralSettings']['values']['outdir']['value'] == str(tmp_path/'audit')
    assert changed['GeneralSettings']['values']['addPid']['value']
    assert changed['Plugins'][0]['values']['parseOptions']['skipParse']['value']
    original['Plugins'].append(original['Plugins'][0].copy())
    template.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='exactly one'):
        ab.isolated_trace_config(template, tmp_path/'audit')


def periods(a, b, device_a=5., device_b=4.9):
    return [dict(arm=arm, token_intervals_ms=list(a if arm == 'A' else b),
                 ranks=[dict(device_ms=device_a if arm == 'A' else device_b)]) for arm in 'ABABAB']


def test_three_device_rounds_without_host_iqr_veto():
    a = [4.9, 5.1] * 100
    weak = ab.compare_periods(periods(a, [4.7, 4.9] * 100))
    assert weak['effective']
    assert weak['saving_ms'] == pytest.approx(.1)
    assert 'validity_threshold_ms' not in weak
    strong = ab.compare_periods(periods(a, [4.3, 4.5] * 100))
    assert strong['effective'] and not strong['formal_gain_credit']
    negative = ab.compare_periods(periods(a, [5.5, 5.7] * 100, device_b=5.1))
    assert negative['slower'] and not negative['effective']
    with pytest.raises(ValueError, match='ABABAB'):
        ab.compare_periods(periods(a, a)[:-1])
    with pytest.raises(ValueError, match='200'):
        ab.summarize([1.] * 199)


def test_device_rounds_must_all_agree_and_use_slowest_rank():
    values = periods([5.] * 200, [4.] * 200)
    values[-1]['ranks'].append(dict(device_ms=5.2))
    comparison = ab.compare_periods(values)
    assert not comparison['effective'] and not comparison['slower']
    assert comparison['round_savings_ms'] == pytest.approx([.1, .1, -.2])


def test_repartition_gate_rejects_equal_tokens_with_different_mutable_state():
    before = [dict(shape=[1, 512], dtype='torch.bfloat16', sha256='canonical')]
    changed = [dict(shape=[1, 512], dtype='torch.bfloat16', sha256='changed')]
    assert ab.check_repartition_state([31, 42], before, [31, 42], before)['mutable_state_exact']
    with pytest.raises(RuntimeError, match='mutable state'):
        ab.check_repartition_state([31, 42], before, [31, 42], changed)
    with pytest.raises(RuntimeError, match='tokens'):
        ab.check_repartition_state([31, 42], before, [31, 43], before)


def test_candidate_factory_rejects_external_files_and_changed_source(tmp_path):
    external = tmp_path / 'candidate.py'
    external.write_text('raise AssertionError("untrusted code must not execute")\n')
    with pytest.raises(ValueError, match='repository Python'):
        ab.load_candidate_factory(external, hashlib.sha256(external.read_bytes()).hexdigest())
    factory = _path.parent / 'deepseek_v41_candidates/all_route_slots.py'
    with pytest.raises(RuntimeError, match='changed after submission'):
        ab.load_candidate_factory(factory, 'not-the-submitted-hash')


def test_all_route_factory_preserves_reference_flags_and_weight_ownership(monkeypatch):
    import importlib
    from types import SimpleNamespace

    stage = torch.nn.Module()
    stage.decode_static_int32 = stage.decode_static_factories = True
    block = torch.nn.Module()
    block.moe = torch.nn.Module()
    block.moe.all_route_slots = False
    block.moe.register_buffer('weight', torch.ones(4))
    stage.layers = torch.nn.ModuleList([block])
    implementation = lambda self, *args, **kwargs: self.weight
    from vllm_gaudi.models import deepseek_v41_program
    monkeypatch.setattr(importlib, 'reload', lambda module: SimpleNamespace(
        PreparedMoE=SimpleNamespace(_forward_n256_fp8=implementation, forward=implementation),
        CompiledStage=deepseek_v41_program.CompiledStage)
        if module is deepseek_v41_program else module)
    path = _path.parent / 'deepseek_v41_candidates/all_route_slots.py'
    factory, _ = ab.load_candidate_factory(path, hashlib.sha256(path.read_bytes()).hexdigest())
    candidate = factory(stage, ab.clone_module)
    assert candidate.decode_static_int32 and candidate.decode_static_factories
    assert candidate.layers[0].moe.all_route_slots and not block.moe.all_route_slots
    assert candidate.layers[0] is not block and candidate.layers[0].moe is not block.moe
    assert candidate.layers[0].moe.weight is block.moe.weight
    assert candidate.layers[0].moe._forward_n256_fp8() is block.moe.weight


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


def test_cpu_queue_gate_uses_pressure_not_driver_d_state_load(tmp_path):
    pressure = tmp_path / 'cpu'
    pressure.write_text('some avg10=0.00 avg60=0.25 avg300=0.65 total=1000\n'
                        'full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n')
    assert ab.cpu_pressure_avg10(pressure) == 0
    pressure.write_text('some avg10=42.98 avg60=13.72 avg300=3.70 total=2000\n')
    assert ab.cpu_pressure_avg10(pressure) == 42.98


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
    program = SimpleNamespace(shared=shared, layers=[block])
    hidden = torch.randn(1, 5120).bfloat16()
    record = ab.observe_mirror_step(program, hidden, 256)
    keys = record['keys']['2']
    assert not keys['visible'][-1] and keys['rows'][-1] == 128
    assert torch.equal(keys['mirror'][keys['visible']], keys['canonical'][keys['visible']])
    assert torch.equal(record['hidden'], hidden) and torch.equal(record['indices']['2'], selection.indices)
    record['hidden'].zero_()
    assert hidden.count_nonzero() > 0
# A Python compilation count alone misses per-storage-offset GC recipes.
def test_recipe_count_includes_deferred_position_copy_variants(tmp_path):
    from tools.deepseek_v41_resident_ab import recipe_cache_count

    root = tmp_path / 'rank2'
    root.mkdir()
    environment = {'PT_HPU_RECIPE_CACHE_CONFIG': str(tmp_path / 'rank{rank}') + ',false,8192',
                   'LOCAL_RANK': '2'}
    assert recipe_cache_count(environment) == 0
    (root / 'first.recipe').write_bytes(b'first position')
    (root / 'first.metadata').write_bytes(b'ignored')
    assert recipe_cache_count(environment) == 1
    (root / 'second.recipe').write_bytes(b'next position')
    assert recipe_cache_count(environment) == 2
    assert recipe_cache_count({}) == 0
