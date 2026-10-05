# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
op = torch.ops.custom_op.custom_deepseek_v41_candidate_coordinates_gaudi2


@pytest.mark.parametrize('tokens', [1, 2, 6])
@pytest.mark.parametrize('blocks', [1, 7, 256, 2048])
def test_full_owned_logical_and_gather_coordinates(tokens, blocks):
    values = op(torch.empty(tokens, blocks, dtype=torch.int32, device='meta'), 524287)
    assert all(tuple(t.shape) == (tokens, blocks * 8) and t.dtype == torch.int32
               and t.storage_offset() == 0 and t.is_contiguous() for t in values)


@pytest.mark.parametrize('shape,dtype,maximum', [((0, 8), torch.int32, 7), ((7, 8), torch.int32, 7),
                                                ((1, 2049), torch.int32, 7), ((1, 8), torch.int64, 7),
                                                ((1, 8), torch.int32, -1), ((1, 8), torch.int32, 1 << 31)])
def test_invalid_contracts(shape, dtype, maximum):
    with pytest.raises(RuntimeError):
        op(torch.empty(shape, dtype=dtype, device='meta'), maximum)
