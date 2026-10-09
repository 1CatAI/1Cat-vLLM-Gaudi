# SPDX-License-Identifier: Apache-2.0
"""C6 MME weight ownership and exclusion of C1/prompt arithmetic."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.deepseek_v41_program import PreparedDecoderLayer
from vllm_gaudi.ops.deepseek_v41_math import hc_pre


@pytest.mark.parametrize("enabled", [False, True])
def test_prepared_control_lifetime_preserves_checkpoint(enabled):
    layer = PreparedDecoderLayer.__new__(PreparedDecoderLayer)
    torch.nn.Module.__init__(layer)
    generator = torch.Generator().manual_seed(41)
    weight = torch.randn(24, 20480, generator=generator) / 100
    original = weight.clone()
    layer.weights = SimpleNamespace(hc_attn_fn=weight, hc_ffn_fn=weight)
    layer.mhc_control_rrms, layer.batch_control_reuse = True, False
    layer.mhc_control_mme = enabled
    layer.prepare_mhc_control_weights()
    assert layer.hc_attn_fn_packed.data_ptr() == weight.data_ptr()
    if enabled:
        packed = layer.hc_attn_fn_mme
        assert packed.shape == (48, 20480) and packed.dtype == torch.bfloat16
        restored = packed[:24].float() + packed[24:].float()
        assert (restored - weight).abs().max() < 5e-7
    else:
        assert layer.hc_attn_fn_mme is None
    layer.release_mhc_control_weights()
    assert layer.hc_attn_fn_mme is layer.hc_ffn_fn_mme is None
    assert torch.equal(weight, original)


@pytest.mark.parametrize("count,decode,expected_calls", [(1, True, 0), (6, False, 0), (2, True, 1), (6, True, 1)])
def test_control_dispatch_keeps_c1_and_prompt_reference(monkeypatch, count, decode, expected_calls):
    generator = torch.Generator().manual_seed(42)
    residual = torch.randn(count, 4, 5120, generator=generator).bfloat16()
    weight = torch.randn(24, 20480, generator=generator) / 1000
    high = weight.bfloat16()
    packed = torch.cat((high, (weight - high.float()).bfloat16()))
    calls = []

    def mme(x, w):
        calls.append((tuple(x.shape), tuple(w.shape)))
        return torch.nn.functional.linear(x.float(), w.float())

    monkeypatch.setattr(torch.ops.custom_op, "custom_deepseek_v41_control_mme_f32_gaudi2", mme, raising=False)
    arguments = (residual, torch.ones(count, 4), weight, torch.ones(3), torch.zeros(24))
    reference = hc_pre(*arguments, decode=decode)
    result = hc_pre(*arguments, decode=decode, control_mme_weight=packed)
    assert len(calls) == expected_calls
    if not expected_calls:
        assert all(torch.equal(a, b) for a, b in zip(reference, result, strict=True))
    else:
        assert calls == [((count, 20480), (48, 20480))]
        assert torch.equal(reference[0], result[0])
        assert all(torch.isfinite(x).all() for x in result)
