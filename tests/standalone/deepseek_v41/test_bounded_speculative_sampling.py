# SPDX-License-Identifier: Apache-2.0
"""Bounded p/q retains the official distribution and an explicit fallback."""
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_speculative_sampling import (
    bounded_sampling_parts, sample_bounded_or_full_distribution, sample_full_distribution)


@pytest.mark.parametrize("kind", ["peaked", "flat", "ties", "unfiltered"])
def test_coverage_and_fallback(kind):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(2, 512, generator=generator) * 0.2
    if kind == "peaked":
        logits[:, :8] += torch.arange(8).float() * .3 + 10
    elif kind == "ties":
        logits[:, :8] = 10
    elif kind == "flat":
        logits.zero_()
    controls = torch.tensor([[1., 1. if kind == "unfiltered" else .95, .13, -1.],
                             [1., 1. if kind == "unfiltered" else .95, .77, -1.]])
    _, _, covered = bounded_sampling_parts(logits, controls, 32)
    assert bool(covered.all()) == (kind == "peaked")
    a, p = sample_full_distribution(logits, controls)
    b, q = sample_bounded_or_full_distribution(logits, controls, 32)
    assert torch.equal(a, b)
    torch.testing.assert_close(q, p, rtol=2e-6, atol=2e-8)
    if not bool(covered.all()):
        assert torch.equal(p, q)
