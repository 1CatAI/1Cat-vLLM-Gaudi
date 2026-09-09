# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.gqa_compact import single_batch_to_blocks


@pytest.mark.parametrize("shape", [(1, 1, 3072), (1, 2, 6, 1, 1), (1, 12)])
@pytest.mark.parametrize("blocks", [1, 17, 32])
def test_mapping_exact_for_changing_masks(shape, blocks):
    generator = torch.Generator().manual_seed(192)
    x = torch.randn(shape, generator=generator).bfloat16()
    for active in range(blocks + 1):
        mapping = (torch.arange(blocks).reshape(-1, 1) < active).bfloat16()
        expected = torch.matmul(mapping, x.reshape(1, -1)).reshape(blocks, *shape[1:])
        assert torch.equal(single_batch_to_blocks(x, mapping), expected)


def test_all_finite_bf16_values_match_single_product_matmul():
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    values = values[torch.isfinite(values)].reshape(1, -1)
    mapping = torch.tensor([[0], [1]], dtype=torch.bfloat16)
    assert torch.equal(single_batch_to_blocks(values, mapping), mapping @ values)


@pytest.mark.parametrize("x,mapping", [
    (torch.empty(2, 16, dtype=torch.bfloat16), torch.empty(8, 2, dtype=torch.bfloat16)),
    (torch.empty(1, 16), torch.empty(8, 1, dtype=torch.bfloat16)),
    (torch.empty(1, 16, dtype=torch.bfloat16), torch.empty(8, dtype=torch.bfloat16)),
])
def test_mapping_rejects_unsupported_inputs(x, mapping):
    with pytest.raises(RuntimeError, match="Single-batch mapping"):
        single_batch_to_blocks(x, mapping)
