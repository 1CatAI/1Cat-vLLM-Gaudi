# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_gaudi.ops.deepseek_v41_sampling import (
    local_nucleus_packet, request_uniform, sample_nucleus_packet, sample_probabilities)


def bounded(logits, controls, width):
    packet = torch.cat([local_nucleus_packet(shard, controls, rank, width)
                        for rank, shard in enumerate(logits.chunk(4, -1))], -1)
    return sample_nucleus_packet(packet, controls, tp_size=4, width=width)


def test_covered_nucleus_fixed_draws_match_full():
    torch.manual_seed(17)
    logits = torch.randn(1, 1024) * 5
    draws = torch.tensor([request_uniform('sample', 42, i) for i in range(4096)])
    controls = torch.stack((torch.ones_like(draws), torch.full_like(draws, .95), draws,
                            -torch.ones_like(draws)), -1)
    logits = logits.expand(len(draws), -1)
    selected, covered = bounded(logits, controls, 64)
    assert covered.all()
    reference = sample_probabilities(logits, controls, filtered=True)
    assert torch.equal(reference, selected)


def test_diffuse_nucleus_requires_full_fallback():
    torch.manual_seed(18)
    controls = torch.tensor([[1., .95, .5, -1.]])
    _, covered = bounded(torch.randn(1, 1024) * .01, controls, 16)
    assert not covered.item()


def test_boundary_ties_require_fallback():
    controls = torch.tensor([[1., .5, .5, -1.]])
    _, covered = bounded(torch.zeros(1, 64), controls, 4)
    assert not covered.item()


def test_greedy_keeps_lowest_rank_unique_maximum():
    controls = torch.tensor([[0., .95, .5, -1.]])
    logits = torch.zeros(1, 64)
    logits[0, 33] = 1
    selected, covered = bounded(logits, controls, 4)
    assert covered.item() and selected.item() == 33


def test_device_draw_counter_and_distribution():
    from vllm_gaudi.ops.deepseek_v41_sampling import device_sampling_draw

    count = 100000
    parameters = torch.tensor([[1., .95, -1.]]).expand(count, -1)
    seed = torch.full((count,), 42, dtype=torch.int32)
    counter = torch.arange(count, dtype=torch.int32)
    original = counter.clone()
    controls = device_sampling_draw(parameters, seed, counter)
    assert torch.equal(counter, original + 1)
    assert torch.all((controls[:, 2] > 0) & (controls[:, 2] < 1))
    expected = device_sampling_draw(parameters, seed, original)
    assert torch.equal(controls, expected)
    histogram = torch.histc(controls[:, 2], bins=10, min=0, max=1) / count
    torch.testing.assert_close(histogram, torch.full((10,), .1), atol=.004, rtol=0)


def test_greedy_ties_use_global_lowest_id_even_outside_local_topk():
    controls = torch.tensor([[0., .95, .5, -1.]])
    selected, covered = bounded(torch.zeros(1, 64), controls, 4)
    assert covered.item() and selected.item() == 0


def test_scalar_completion_wire_preserves_vocabulary_boundary_and_coverage():
    from vllm_gaudi.ops.deepseek_v41_sampling import pack_sample_status, unpack_sample_status

    for token in (0, 31, 129279):
        for covered in (False, True):
            wire = pack_sample_status(torch.tensor([[token]], dtype=torch.int32), torch.tensor([[covered]]))
            assert wire.dtype == torch.int32 and wire.shape == (1, 1)
            assert unpack_sample_status(wire[0].tolist()) == (token, covered)
