# SPDX-License-Identifier: Apache-2.0
"""A repaired proposal always belongs to its actual probability support."""
import torch

from vllm_gaudi.ops.deepseek_v41_speculative_sampling import full_vocabulary_cdf_draw


def test_zero_and_one_uniform_skip_zero_probability_words():
    q = torch.zeros(5, 129280)
    q[:, [2046, 2047, 4096, 126999]] = torch.tensor([.125, .125, .25, .5])
    uniforms = torch.tensor([[0.], [.25], [.25000003], [.99999994], [1.]])
    before = q.clone()
    actual = full_vocabulary_cdf_draw(q, uniforms)
    assert actual.tolist() == [2046, 2047, 4096, 126999, 126999]
    assert (q.gather(1, actual.long()[:, None]) > 0).all()
    assert torch.equal(q, before)


def test_repair_total_comes_from_the_same_scan():
    generator = torch.Generator().manual_seed(683)
    q = torch.zeros(1, 129280)
    prefix = torch.randn(2048, generator=generator).mul(2).exp()
    prefix = prefix / prefix.double().sum() * .25
    q[0, :2048] = prefix
    q[0, 4095] = .75
    selected = full_vocabulary_cdf_draw(q, torch.tensor([[.25]]))
    assert 0 <= int(selected) < q.shape[-1]
    assert q[0, selected] > 0
    # The selected word crosses this scan's own threshold. Independent
    # tiled sums cannot substitute a different total here.
    cdf = q.cumsum(-1)
    threshold = .25 * cdf[0, -1]
    assert cdf[0, selected] >= threshold
    if int(selected) > 0:
        assert cdf[0, selected - 1] < threshold
