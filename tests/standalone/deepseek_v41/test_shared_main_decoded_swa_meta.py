# SPDX-License-Identifier: Apache-2.0
"""The mirror reader uses the publisher's full layer slot, without mutation."""
import os

import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])


def operands(heads=16, *, rows=512, dtype=torch.bfloat16):
    def tensor(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device='meta')
    return (
        tensor((1, heads, 512), torch.bfloat16), tensor((rows, 512), dtype),
        tensor((1, 640, 512), torch.bfloat16), tensor((1, 640)), tensor((1,), torch.int32),
        tensor((heads,)), tensor((1,)), tensor((1,), torch.int32),
    )


@pytest.mark.parametrize('heads', [8, 16, 32, 64])
def test_reader_preserves_heads_and_does_not_mutate_state(heads):
    op = torch.ops.custom_op.custom_deepseek_v41_main_reuse_decoded_swa_mla_gaudi2
    inputs = operands(heads)
    result = op(*inputs)
    assert result.shape == inputs[0].shape and result.dtype == torch.bfloat16
    assert all(argument.alias_info is None for argument in op.default._schema.arguments)


@pytest.mark.parametrize('rows,dtype', [(256, torch.bfloat16), (512, torch.uint8), (512, torch.float32)])
def test_reader_rejects_incompatible_publisher_slot(rows, dtype):
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_main_reuse_decoded_swa_mla_gaudi2(
            *operands(rows=rows, dtype=dtype))
