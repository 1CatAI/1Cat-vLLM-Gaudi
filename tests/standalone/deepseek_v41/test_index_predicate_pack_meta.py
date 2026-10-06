# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch
import habana_frameworks.torch.core  # noqa: F401

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
OP = torch.ops.custom_op.custom_deepseek_v41_index_threshold_gaudi2


@pytest.mark.parametrize('batch,variant', [(1, 0), (1, 1), (2, 1), (6, 1)])
def test_threshold_variants_keep_metadata_contract(batch, variant):
    scores = torch.empty((batch, 16384), device='meta')
    position = torch.empty(batch, dtype=torch.int32, device='meta')
    result = OP(scores, position, 2, 1, 0, variant)
    assert result.shape == (batch, 1074) and result.dtype == torch.int32


def test_previous_call_uses_reference_by_default():
    scores = torch.empty(8192, device='meta')
    position = torch.empty(1, dtype=torch.int32, device='meta')
    assert OP(scores, position, 2, 0, 0).shape == (562, )


@pytest.mark.parametrize('variant', [-1, 2])
def test_unknown_variant_rejected(variant):
    with pytest.raises(RuntimeError, match='variant'):
        OP(torch.empty((1, 16384), device='meta'), torch.empty(1, dtype=torch.int32, device='meta'), 2, 1, 0, variant)
