# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_direct_state_matches_indexed_decode(batch: int, state_dtype: torch.dtype):
    torch.manual_seed(7)
    dim = 64
    width = 4
    state_start = 2

    x = torch.randn(batch, dim, dtype=torch.bfloat16)
    weight = torch.randn(dim, width, dtype=torch.bfloat16)
    bias = torch.randn(dim, dtype=torch.bfloat16)
    indexed_pool = torch.randn(batch + state_start + 1, width - 1, dim, dtype=state_dtype)
    direct_pool = indexed_pool[state_start:state_start + batch].clone()
    indices = torch.arange(state_start, state_start + batch, dtype=torch.int32)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32)

    indexed_out = hpu_causal_conv1d_update(
        x,
        indexed_pool,
        weight,
        bias,
        activation="silu",
        conv_state_indices=indices,
        query_start_loc=query_start_loc,
    )
    direct_out = hpu_causal_conv1d_update(
        x,
        direct_pool,
        weight,
        bias,
        activation="silu",
        conv_state_indices=None,
        query_start_loc=query_start_loc,
        direct_state_layout=True,
    )
    torch.testing.assert_close(direct_out, indexed_out, rtol=0, atol=0)
    torch.testing.assert_close(
        direct_pool,
        indexed_pool[state_start:state_start + batch],
        rtol=0,
        atol=0,
    )


def test_direct_state_rejects_index_tensor():
    with pytest.raises(ValueError, match="cache_indices"):
        hpu_causal_conv1d_update(
            torch.zeros(1, 8),
            torch.zeros(1, 3, 8),
            torch.zeros(8, 4),
            conv_state_indices=torch.zeros(1, dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            direct_state_layout=True,
        )
