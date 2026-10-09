# SPDX-License-Identifier: Apache-2.0
"""C1 post/collapse accepts batch rows and carried RRMS without new TP code."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_MHC_DEFERRED_LIBRARY")
    if not path:
        pytest.skip("Set the additive deferred mHC registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


@pytest.mark.parametrize("rows,peers", ((2, 1), (6, 1), (6, 4)))
def test_prepared_rows(rows, peers):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    values = tensor((rows, 5120) if peers == 1 else (peers, rows, 5120), torch.bfloat16)
    residual = tensor((rows, 4, 5120), torch.bfloat16)
    raw = tensor((rows, 25), torch.float32)
    outputs = torch.ops.custom_op.custom_deepseek_v41_mhc_mme_post_collapse_gaudi2(
        values, residual, raw, tensor((3,), torch.float32), tensor((24,), torch.float32), 1e-20)
    assert [result.shape for result in outputs] == [(rows, 4, 5120), (rows, 5120), (rows, 24)]
    assert [result.dtype for result in outputs] == [torch.bfloat16, torch.bfloat16, torch.float32]
