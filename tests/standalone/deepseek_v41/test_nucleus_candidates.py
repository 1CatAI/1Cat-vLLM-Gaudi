# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_sampling import (
    local_greedy_candidate, local_nucleus_candidates, sample_nucleus_candidates, sample_replay_candidates,
)


def packet(logits, controls, size, candidates, sorted_candidates=True):
    shards = logits.chunk(size, -1)
    return torch.cat([
        local_nucleus_candidates(shard, controls, rank, candidates=candidates, sorted_candidates=sorted_candidates)
        for rank, shard in enumerate(shards)
    ], -1)


@pytest.mark.parametrize('size', [2, 4])
@pytest.mark.parametrize('candidates', [64, 128, 256])
@pytest.mark.parametrize('sorted_candidates', [False, True])
def test_exact_draws_with_full_vocabulary_mass(size, candidates, sorted_candidates):
    generator = torch.Generator().manual_seed(914)
    logits = torch.randn(1, 129280, generator=generator) * 8
    controls = torch.tensor([[1., .95, .5, -1.]])
    gathered = packet(logits, controls, size, candidates, sorted_candidates)
    draws = torch.rand(4096, generator=generator).clamp(1e-6, 1 - 1e-6)
    controls = controls.expand(draws.numel(), -1).clone()
    controls[:, 2] = draws
    actual, complete = sample_nucleus_candidates(
        gathered.expand(draws.numel(), -1), controls, candidates=candidates
    )
    values, ids = logits.sort(-1, descending=True)
    probabilities = values.softmax(-1)
    before = probabilities.cumsum(-1) - probabilities
    cumulative = torch.where(before < .95, probabilities, 0.).cumsum(-1)[0]
    expected = ids[0, torch.searchsorted(cumulative, draws * cumulative[-1])]
    assert complete.all()
    assert torch.equal(actual[:, 0].long(), expected)


def test_incomplete_nucleus_is_never_renormalized_as_complete():
    logits = torch.linspace(-.1, .1, 400).reshape(1, -1)
    controls = torch.tensor([[1., .95, .4, -1.]])
    _, complete = sample_nucleus_candidates(packet(logits, controls, 4, 8), controls, candidates=8)
    assert not complete.item()


def test_ties_and_unfiltered_draws_require_separate_paths():
    logits = torch.zeros(1, 400)
    logits[:, :2] = 30
    controls = torch.tensor([[1., .95, .4, -1.]])
    _, complete = sample_nucleus_candidates(packet(logits, controls, 4, 8), controls, candidates=8)
    assert not complete.item()
    controls[:, 1] = 1
    _, complete = sample_nucleus_candidates(packet(logits, controls, 4, 8), controls, candidates=8)
    assert not complete.item()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_mass_requires_reference(value):
    logits = torch.full((1, 400), value)
    controls = torch.tensor([[1., .95, .4, -1.]])
    _, complete = sample_nucleus_candidates(packet(logits, controls, 4, 8), controls, candidates=8)
    assert not complete.item()


@pytest.mark.parametrize('size', [2, 4])
def test_replay_packet_preserves_greedy_ties_and_marks_fallback(size):
    logits = torch.zeros(2, 400)
    logits[0, 3] = logits[0, 203] = 10
    controls = torch.tensor([[0., 1., .5, -1.], [1., .95, .5, -1.]])
    local = logits.chunk(size, -1)
    packets = [torch.cat((local_nucleus_candidates(value, controls, rank, candidates=8),
                          local_greedy_candidate(value, rank)), -1) for rank, value in enumerate(local)]
    selected = sample_replay_candidates(local[0], controls, 0,
                                        lambda value, dim: torch.cat(packets, dim), candidates=8)
    assert selected[0, 0] == 3
    assert selected[1, 0] >= logits.shape[-1]
