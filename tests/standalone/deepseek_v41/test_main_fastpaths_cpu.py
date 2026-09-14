# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for main fast paths reused by the C1-C6 paged program."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, restore_expert
from vllm_gaudi.ops.deepseek_v41_math import quantize_activation
from vllm_gaudi.ops.deepseek_v41_qkv import FusedQKVInput


@pytest.mark.parametrize("n,k", [(256, 128), (2304, 5120), (5120, 1152)])
def test_prepared_n256_keeps_all_original_codes(n, k):
    rng = np.random.default_rng(102)
    q = rng.integers(-32768, 32768, (n // 128, k * 32), dtype=np.int16)
    s = rng.integers(110, 115, (n // 128, k * 4), dtype=np.uint16) << 7
    nq, ns, channel, record = prepare_expert(q, s)
    restored_q, restored_s = restore_expert(nq, ns)
    assert np.array_equal(q, restored_q)
    assert np.array_equal(s, restored_s)
    assert nq.nbytes == q.nbytes
    assert channel.shape == (n // 256, 256)
    assert record["temporary_upper_bound_bytes"] <= 2 * 2**30


class InputProjection(FusedQKVInput, nn.Module):

    def __init__(self, quantized):
        super().__init__()
        self.weights = nn.Module()
        for name, rows in (("wq_a", 64), ("wkv", 32)):
            projection = nn.Module()
            projection.register_buffer("weight", torch.randn(rows, 128).bfloat16())
            self.weights.add_module(name, projection)
        if quantized:
            self.weights.wq_a.register_buffer("scale", torch.ones(1))
            self.weights.wkv.register_buffer("scale", torch.ones(1))
        self.qkv_fused_input = True
        self._fused_qkv_weight = None
        self._fused_qkv_quantized = False
        self.linear = lambda value, w: F.linear(quantize_activation(value) if hasattr(w, "scale") else value, w.weight)


@pytest.mark.parametrize("tokens", range(1, 7))
@pytest.mark.parametrize("quantized", [False, True])
def test_qkv_c6_rows_and_reload(tokens, quantized):
    torch.manual_seed(81)
    layer = InputProjection(quantized)
    with torch.inference_mode():
        x = torch.randn(tokens, 128).bfloat16()
        expected = layer._project_qkv_input(x)
        layer.prepare_qkv_input_weight()
        actual = layer._project_qkv_input(x)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        assert layer.weights.wq_a.weight.untyped_storage().data_ptr() == layer.fused_wqa_wkv.untyped_storage().data_ptr(
        )
        assert layer.weights.wkv.weight.untyped_storage().data_ptr() == layer.fused_wqa_wkv.untyped_storage().data_ptr()
        layer.invalidate_qkv_input_weight()
        layer.weights.wq_a.weight = torch.randn_like(layer.weights.wq_a.weight)
        layer.weights.wkv.weight = torch.randn_like(layer.weights.wkv.weight)
        expected = layer._project_qkv_input(-x)
        layer.prepare_qkv_input_weight()
        actual = layer._project_qkv_input(-x)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
