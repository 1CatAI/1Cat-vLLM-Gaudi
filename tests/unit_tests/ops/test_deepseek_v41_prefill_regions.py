# SPDX-License-Identifier: Apache-2.0
"""Explicit layer bindings and BF16 boundaries of the prompt input region."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import hc_pre, prefill_hc_input, rms_norm
from vllm_gaudi.ops.deepseek_v41_prefill_regions import _signature, validate_prefill_region_config


@pytest.mark.parametrize("tokens", [1, 7, 31])
def test_input_region_uses_current_weights_and_previous_sublayer_mix(monkeypatch, tokens):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REGIONS", "0")
    torch.manual_seed(5187)
    residual = torch.randn(tokens, 4, 64).bfloat16()
    previous = torch.rand(tokens, 4)
    scale, base = torch.rand(3), torch.randn(24)
    old = None
    for change in range(2):
        weight = torch.randn(24, 256) * 0.01
        norm = torch.rand(64).bfloat16()
        mixed = previous if change == 0 else previous.flip(-1)
        result = prefill_hc_input(residual, mixed, weight, scale, base, norm, 1e-6, 1e-6, 20, None)
        expected = hc_pre(residual, mixed, weight, scale, base, 1e-6, 1e-6, 20)
        for observed, reference in zip(result[:4], expected, strict=True):
            assert torch.equal(observed, reference)
        assert torch.equal(result[4], rms_norm(expected[0], norm, 1e-6))
        assert result[0].dtype == result[4].dtype == torch.bfloat16
        if old is not None:
            assert not torch.equal(result[0], old[0])
            assert not torch.equal(result[1], old[1])
        old = result


def test_region_signature_ignores_runtime_values_but_keeps_layout():
    first = torch.zeros(2, 4)
    changed = torch.ones(2, 4)
    assert _signature((first, )) == _signature((changed, ))
    transposed = torch.zeros(4, 2).t()
    assert _signature((first, )) != _signature((transposed, ))
    offset_view = torch.zeros(3, 4)[1:]
    assert _signature((first, )) != _signature((offset_view, ))


@pytest.mark.parametrize("missing", ["VLLM_HPU_DSV41_PREFILL_REGIONS", "VLLM_HPU_DSV41_PREFILL_MHC_POST"])
def test_input_region_cannot_silently_skip_compilation_or_bf16_boundary(monkeypatch, missing):
    for name in ("VLLM_HPU_DSV41_PREFILL_MHC_INPUT", "VLLM_HPU_DSV41_PREFILL_REGIONS",
                 "VLLM_HPU_DSV41_PREFILL_MHC_POST"):
        monkeypatch.setenv(name, "1")
    validate_prefill_region_config()
    monkeypatch.setenv(missing, "0")
    with pytest.raises(ValueError, match="Compiled prefill mHC input requires"):
        validate_prefill_region_config()
