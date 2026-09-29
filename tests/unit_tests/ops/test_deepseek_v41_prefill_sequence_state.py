# SPDX-License-Identifier: Apache-2.0
"""Token ownership keeps residual arithmetic and packed Engram row order."""
import pytest
import torch

from vllm_gaudi.ops import deepseek_v41_prefill_sequence_state as sequence
from vllm_gaudi.ops.deepseek_v41_math import hc_post


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_engram_exchange_keeps_all_heads_for_the_owned_rows(monkeypatch, rank, dtype):
    packets = [(torch.arange(8 * 2 * 8).reshape(8, 2, 8) + source * 17).to(dtype) for source in range(4)]
    group = object()
    monkeypatch.setattr(sequence.dist, "get_world_size", lambda actual: 4 if actual is group else 0)
    monkeypatch.setattr(sequence.dist, "get_rank", lambda actual: rank)

    def exchange(received, sent, *, group):
        wire_dtype = torch.int32 if dtype == torch.uint8 else dtype
        assert sent.dtype == received.dtype == wire_dtype
        assert torch.equal(sent.view(dtype), packets[rank])
        incoming = torch.cat([packet[rank * 2:(rank + 1) * 2] for packet in packets], 0)
        received.copy_(incoming.contiguous().view(wire_dtype))

    monkeypatch.setattr(sequence.dist, "all_to_all_single", exchange)
    actual = sequence.exchange_engram_tokens(packets[rank], group=group)
    expected = torch.cat([packet[rank * 2:(rank + 1) * 2] for packet in packets], 1)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("rank", range(4))
def test_owned_reduction_keeps_the_original_full_allreduce(monkeypatch, rank):
    from types import SimpleNamespace
    from vllm import distributed
    monkeypatch.setattr(distributed, "get_tp_group", lambda: SimpleNamespace(world_size=4, rank_in_group=rank))
    partial = torch.arange(32).reshape(8, 4).bfloat16()
    original = partial.clone()
    reduced = partial.flip(0).clone()
    seen = []

    def reduce(value):
        seen.append(value)
        return reduced

    actual = sequence.reduce_owned_tokens(partial, reduce)
    assert seen == [partial] and torch.equal(partial, original)
    assert torch.equal(actual, reduced[rank * 2:(rank + 1) * 2])


def test_sequence_owned_mhc_preserves_controls_and_next_input(monkeypatch):
    from vllm_gaudi.ops.deepseek_v41_math import prefill_hc_input
    from vllm_gaudi.ops.deepseek_v41_prefill_sequence_state import token_owner
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REGIONS", "0")
    torch.manual_seed(986)
    residual = torch.randn(32, 4, 32).bfloat16()
    previous = torch.rand(32, 4)
    fn, scale, base = torch.randn(24, 128) * .01, torch.ones(3), torch.randn(24) * .02
    norm = torch.rand(32).bfloat16()
    arguments = fn, scale, base, norm, 1e-20, 1e-6, 20, fn
    expected = prefill_hc_input(residual, previous, *arguments)
    pieces = [
        prefill_hc_input(residual[token_owner(32, rank)],
                         previous[token_owner(32, rank)],
                         *arguments,
                         logical_tokens=32) for rank in range(4)
    ]
    for position, wanted in enumerate(expected):
        torch.testing.assert_close(torch.cat([part[position] for part in pieces]), wanted, rtol=1e-6, atol=1e-7)
    updated = hc_post(expected[-1], residual, expected[2], expected[3])
    actual = torch.cat([
        hc_post(pieces[rank][-1], residual[token_owner(32, rank)], pieces[rank][2], pieces[rank][3])
        for rank in range(4)
    ])
    assert torch.equal(actual, updated)
    with pytest.raises(ValueError, match="four equal"):
        token_owner(31, 0)
