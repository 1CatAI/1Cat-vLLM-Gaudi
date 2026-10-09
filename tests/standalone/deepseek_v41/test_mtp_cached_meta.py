# SPDX-License-Identifier: Apache-2.0
"""Draft banks retain the original top-three BF16 producer/consumer geometry."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_MTP_CACHED_LIBRARY")
    if not path:
        pytest.skip("Set the independent draft cache registration")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(path)


def operands(rows, intermediate):
    def t(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")
    return (t((rows, 5120), torch.bfloat16), t((rows, 3), torch.int32),
            t((rows, 3), torch.float32), t((128, intermediate * 2 // 128, 163840), torch.int16),
            t((128, 40, intermediate * 32), torch.int16),
            t((128, intermediate * 2 // 128, 20480), torch.bfloat16),
            t((128, 40, intermediate * 4), torch.bfloat16), t((128,), torch.bfloat16), True,
            t((128, 5120, intermediate * 2), torch.bfloat16),
            t((128, intermediate, 5120), torch.bfloat16))


@pytest.mark.parametrize("rows,width", ((1, 640), (5, 640), (6, 640), (5, 1280)))
def test_draft_shapes(rows, width):
    result = torch.ops.custom_op.custom_deepseek_v41_mtp_cached_moe_bf16_gaudi2(*operands(rows, width))
    assert result.shape == (rows, 5120) and result.dtype == torch.bfloat16


def test_wrong_bank_layout():
    values = list(operands(5, 640))
    values[-1] = values[-1].transpose(1, 2).contiguous()
    with pytest.raises(RuntimeError, match="Cached draft weight bank"):
        torch.ops.custom_op.custom_deepseek_v41_mtp_cached_moe_bf16_gaudi2(*values)
