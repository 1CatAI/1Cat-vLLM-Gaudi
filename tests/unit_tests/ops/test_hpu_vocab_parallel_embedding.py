# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.ops.hpu_vocab_parallel_embedding as embedding_module
from vllm_gaudi.ops.hpu_vocab_parallel_embedding import HPUVocabParallelEmbedding


@pytest.mark.parametrize("defer_reduce", [False, True])
def test_vocab_parallel_embedding_can_defer_tp2_reduce(monkeypatch, defer_reduce):
    input_ids = torch.tensor([3, 7])
    input_mask = torch.tensor([False, True])
    masked_input = torch.tensor([3, 0])
    reduced = []

    monkeypatch.setattr(
        embedding_module,
        "get_masked_input_and_mask",
        lambda *_args: (masked_input, input_mask),
    )
    monkeypatch.setattr(
        embedding_module,
        "tensor_model_parallel_all_reduce",
        lambda output: reduced.append(output.clone()) or output + 10,
    )

    embedding = SimpleNamespace(
        tp_size=2,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=8,
            num_org_vocab_padding=0,
            added_vocab_start_index=8,
            added_vocab_end_index=8,
        ),
        quant_method=SimpleNamespace(
            embedding=lambda _layer, indices: indices.unsqueeze(-1).expand(-1, 2).to(torch.float32),
        ),
        _hpu_defer_tp2_reduce=defer_reduce,
    )

    output = HPUVocabParallelEmbedding.forward(embedding, input_ids)

    local_output = torch.tensor([[3.0, 3.0], [0.0, 0.0]])
    if defer_reduce:
        assert reduced == []
        assert torch.equal(output, local_output)
    else:
        assert len(reduced) == 1
        assert torch.equal(reduced[0], local_output)
        assert torch.equal(output, local_output + 10)
