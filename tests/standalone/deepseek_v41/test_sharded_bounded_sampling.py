# SPDX-License-Identifier: Apache-2.0
"""Partitioned exact p/q preserves full sorting and uncovered fallback."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_sampling import local_nucleus_packet
from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
    sample_bounded_or_full_sharded, sample_full_distribution)


@pytest.mark.parametrize("tp_size,kind", [(2, "peaked"), (4, "peaked"), (4, "flat"), (4, "ties")])
def test_partitioned_probability(tp_size, kind, monkeypatch):
    # This gate validates probability algebra on CPU. Dynamo's registered HPU
    # device interface may acquire a device even for a CPU cond compilation;
    # hardware conditionals/collectives have a separate leased backend gate.
    def cpu_cond(predicate, yes, no, operands):
        assert predicate.device.type == "cpu"
        return (yes if bool(predicate) else no)(*operands)

    monkeypatch.setattr(torch, "cond", cpu_cond)
    local_vocab, width = 512, 64
    generator = torch.Generator().manual_seed(42)
    full = torch.randn(2, local_vocab * tp_size, generator=generator) * .1
    for rank in range(tp_size):
        full[:, rank * local_vocab:rank * local_vocab + 8] += 12 + torch.arange(8).float() * .3
    if kind == "flat":
        full.zero_()
    elif kind == "ties":
        full[:, :8] = 20
    controls = torch.tensor([[1., .95, .13, -1.], [1., .95, .77, -1.]])
    packet = torch.cat([local_nucleus_packet(full[:, r * local_vocab:(r + 1) * local_vocab], controls, r, width)
                        for r in range(tp_size)], -1)
    expected, probabilities = sample_full_distribution(full, controls)

    def gather(value, dim):
        assert dim == -1
        return full if value.shape[-1] == local_vocab else packet

    pieces = []
    for rank in range(tp_size):
        actual, probability = sample_bounded_or_full_sharded(
            full[:, rank * local_vocab:(rank + 1) * local_vocab], controls,
            tp_rank=rank, tp_size=tp_size, all_gather=gather, width=width)
        assert torch.equal(actual, expected)
        pieces.append(probability)
    actual = torch.cat(pieces, -1)
    torch.testing.assert_close(actual, probabilities, rtol=2e-6, atol=2e-8)
    if kind in ("flat", "ties"):
        assert torch.equal(actual, probabilities)
