# SPDX-License-Identifier: Apache-2.0
"""C6 native input uses bounded seeds without entering C1 segmentation."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_replay import StageReplay, _native_input_count_compatible


def owner(dspark=True, legacy=False):
    program = torch.nn.Module()
    program.pp_rank, program.dspark = 0, dspark
    program.fp8_decode, program.expert_n256 = True, not legacy
    return program


@pytest.mark.parametrize('count', [1, 2, 5, 6])
def test_c6_seed_ownership_and_complete_stage_dispatch(monkeypatch, count):
    monkeypatch.setenv('VLLM_HPU_DSV41_DSPARK_NATIVE_TARGET_INPUT', '1')
    monkeypatch.setenv('VLLM_HPU_DSV41_V2_SEGMENTED_PREFIX', '0')
    monkeypatch.setenv('VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT', '0')

    class Capture(StageReplay):
        def __call__(self, *args, **kwargs):
            return args, kwargs

    program = owner()
    replay = Capture(program)
    ids = torch.arange(count, dtype=torch.int32)
    positions = ids + 16384
    first, flags = replay.from_input_ids(positions, ids, ())
    second, _ = replay.from_input_ids(positions, ids + 1, ())
    assert first[0] is second[0] and first[1] is second[1]
    assert first[0].shape == (count, 4, 5120) and first[1].shape == (count, 4)
    assert first[3] is ids and flags == dict(native_input=True)
    assert replay.latest_tail is None


def test_disabled_candidate_preserves_dspark_exclusion(monkeypatch):
    monkeypatch.delenv('VLLM_HPU_DSV41_DSPARK_NATIVE_TARGET_INPUT', raising=False)
    assert not _native_input_count_compatible(owner(), 6)


def test_native_input_limits_and_legacy_precision(monkeypatch):
    monkeypatch.setenv('VLLM_HPU_DSV41_DSPARK_NATIVE_TARGET_INPUT', '1')
    assert not _native_input_count_compatible(owner(), 7)
    assert not _native_input_count_compatible(owner(), 0)
    assert not _native_input_count_compatible(owner(legacy=True), 6)
    assert _native_input_count_compatible(owner(dspark=False), 1)
    assert not _native_input_count_compatible(owner(dspark=False), 6)
