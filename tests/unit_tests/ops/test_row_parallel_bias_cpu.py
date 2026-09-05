# SPDX-License-Identifier: Apache-2.0
"""Post-reduction bias semantics must be the same on every rank."""

from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops import hpu_row_parallel_linear as row_parallel


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("skip_bias_add", [False, True])
def test_chunked_bias_matches_on_both_ranks(monkeypatch, rank, skip_bias_add):
    layer = row_parallel.HPURowParallelLinear.__new__(row_parallel.HPURowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.input_is_parallel = True
    layer.num_chunks = 2
    layer.chunk_threshold = 2
    layer.reduce_results = True
    layer.tp_size = 2
    layer.tp_rank = rank
    layer.output_size_per_partition = 4
    layer.bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    layer.skip_bias_add = skip_bias_add
    layer.return_bias = True
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    layer.quant_method = SimpleNamespace(apply=lambda layer, x, bias: torch.nn.functional.linear(x, weight, bias))
    monkeypatch.setattr(row_parallel, "get_tp_group", lambda: SimpleNamespace(device_group="group"))

    def all_reduce(tensor, *, group, async_op):
        assert group == "group" and async_op
        return SimpleNamespace(wait=lambda: tensor.mul_(2))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    output, output_bias = layer(x)
    reference = torch.nn.functional.linear(x, weight) * 2
    if not skip_bias_add:
        reference = reference + layer.bias
        assert output_bias is None
    else:
        assert output_bias is layer.bias
    torch.testing.assert_close(output, reference)
