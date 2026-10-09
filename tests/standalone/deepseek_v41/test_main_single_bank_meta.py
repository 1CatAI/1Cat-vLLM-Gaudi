# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.getenv("DSV41_MAIN_SINGLE_BANK_LIBRARY")
    if not path:
        pytest.skip("Set the single-bank registration")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(path)


def arguments(rows, heads):
    def t(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")
    return (t((rows, heads, 512), torch.bfloat16), t((256, 528), torch.uint8),
            t((4096, 288), torch.uint8), t((rows, 512), torch.int32),
            t((rows,), torch.int32), t((8,), torch.int32),
            t((heads,), torch.float32), t((1,), torch.float32), t((rows,), torch.int32), 1)


@pytest.mark.parametrize("rows,heads", [(2, 8), (5, 16), (6, 8)])
def test_publisher_keeps_only_one_bf16_bank_and_mask(rows, heads):
    args = arguments(rows, heads)
    output, bank, mask = torch.ops.custom_op.custom_deepseek_v41_main_single_bank_publish_mla_gaudi2(*args)
    assert output.shape == (rows, heads, 512)
    assert bank.shape == (rows, 640, 512) and bank.dtype == torch.bfloat16
    assert mask.shape == (rows, 640) and mask.dtype == torch.float32
    reused = torch.ops.custom_op.custom_deepseek_v41_main_single_bank_reuse_mla_gaudi2(
        args[0], args[1], bank, mask, args[4], args[6], args[7], args[8])
    assert reused.shape == output.shape and reused.dtype == output.dtype


def test_wrong_selection_rows_rejected():
    args = list(arguments(6, 8))
    args[3] = torch.empty((1, 512), dtype=torch.int32, device="meta")
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_main_single_bank_publish_mla_gaudi2(*args)


def test_seven_queries_rejected():
    with pytest.raises(RuntimeError):
        torch.ops.custom_op.custom_deepseek_v41_main_single_bank_publish_mla_gaudi2(*arguments(7, 8))
