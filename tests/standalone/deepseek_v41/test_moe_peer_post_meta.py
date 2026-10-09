# SPDX-License-Identifier: Apache-2.0
"""Peer/post preserves the shared TP-parametric consumer contract."""
import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def register():
    path = os.environ.get("DSV41_MOE_PEER_POST_LIBRARY")
    if not path:
        pytest.skip("Set the additive MoE peer/post registration")
    import habana_frameworks.torch.core  # noqa: F401

    torch.ops.load_library(path)


def inputs(ranks, rows):
    def tensor(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    return (tensor((ranks, rows, 5120), torch.bfloat16), tensor((rows, 4, 5120), torch.bfloat16),
            tensor((rows, 4), torch.float32), tensor((rows, 4, 4), torch.float32),
            tensor((rows, 4), torch.float32))


@pytest.mark.parametrize("ranks,rows", ((2, 2), (2, 6), (4, 6)))
def test_shared_tp_contract(ranks, rows):
    residual, collapse = torch.ops.custom_op.custom_deepseek_v41_dspark_moe_peer_post_gaudi2(*inputs(ranks, rows))
    assert residual.shape == (rows, 4, 5120) and residual.dtype == torch.bfloat16
    assert collapse.shape == (rows, 5120) and collapse.dtype == torch.bfloat16


@pytest.mark.parametrize("ranks,rows", ((1, 6), (4, 1), (4, 7)))
def test_reject_other_dispatch(ranks, rows):
    with pytest.raises(RuntimeError, match="C2-C6"):
        torch.ops.custom_op.custom_deepseek_v41_dspark_moe_peer_post_gaudi2(*inputs(ranks, rows))
