# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


@pytest.mark.parametrize('operator', ['dual_quant', 'dual_bf16_quant'])
@pytest.mark.parametrize('batch', [1, 2, 6])
def test_shared_and_routed_quantizers_have_separate_outputs(batch, operator):
    value = torch.empty((batch, 5120), dtype=torch.bfloat16, device='meta')
    weight = torch.empty((5120,), dtype=torch.bfloat16, device='meta')
    op = getattr(torch.ops.custom_op, f"custom_deepseek_v41_ffn_norm_{operator}_gaudi2")
    out = op(value, weight, 1e-20)
    assert [tuple(t.shape) for t in out] == [(batch, 5120), (batch, 5120), (batch, 1),
                                           (batch, 5120), (batch, 1)]
    assert [t.dtype for t in out] == [torch.bfloat16, torch.float8_e4m3fn, torch.float32,
                                    torch.float8_e4m3fn, torch.float32]


@pytest.mark.parametrize('operator', ['dual_quant', 'dual_bf16_quant'])
def test_invalid_input_is_rejected(operator):
    value = torch.empty((1, 5120), dtype=torch.float32, device='meta')
    weight = torch.empty((5120,), dtype=torch.bfloat16, device='meta')
    with pytest.raises(RuntimeError, match='BF16'):
        op = getattr(torch.ops.custom_op, f"custom_deepseek_v41_ffn_norm_{operator}_gaudi2")
        op(value, weight, 1e-20)
