# SPDX-License-Identifier: Apache-2.0
"""The decode input producer reads only the requested committed token."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_batch_input import fill_request_metadata
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState


class NoPrefixCopy(list):

    def __add__(self, other):
        raise AssertionError("Decode copied the complete request prefix")


def test_position_reads_survive_append_reconcile_and_recompute():
    request = RequestState("a", NoPrefixCopy([3, 5, 7]), [], None, ([1], ), output=[11, 13])
    for output_count, new_tokens, all_tokens, expected in (
        (4, [17, 19], None, [3, 5, 7, 11, 13, 17, 19]),
        (1, [], None, [3, 5, 7, 11]),
        (2, [], [3, 5, 7, 23, 29], [3, 5, 7, 23, 29]),
    ):
        request.reconcile(output_count, new_tokens, all_tokens)
        request.recompute_until = request.token_count
        assert request.token_count == len(expected)
        assert [request.token_at(i) for i in range(request.token_count)] == expected
        assert [request.token_at(i - len(expected)) for i in range(len(expected))] == expected
        assert request.decode_start == len(expected)
        for bad in (-request.token_count - 1, request.token_count):
            with pytest.raises(IndexError):
                request.token_at(bad)


@pytest.mark.parametrize("count,capacity", [(1, 1), (3, 4), (16, 16), (32, 32)])
def test_shared_input_ids_do_not_copy_history(count, capacity):
    host = torch.full((3, capacity), 99, dtype=torch.int32)
    requests = [
        RequestState(str(i), NoPrefixCopy([7] * 2048), [], None, ([1], ), 2048, [129264 if i == 0 else 31 + i])
        for i in reversed(range(count))
    ]
    owners = [SimpleNamespace(index=i) for i in reversed(range(count))]
    inputs = [request.token_at(request.num_computed_tokens) for request in requests]
    fill_request_metadata(host, requests, owners, input_ids=iter(inputs))
    expected = torch.full_like(host, -1)
    expected[0].zero_()
    for row, (request, owner, token) in enumerate(zip(requests, owners, inputs, strict=True)):
        expected[:, row] = torch.tensor([token, request.num_computed_tokens, owner.index])
    assert torch.equal(host, expected)
