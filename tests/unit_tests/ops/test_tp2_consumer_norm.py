# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_gaudi.ops.tp2_consumer_norm import tp2_consumer_norm


def test_consumer_preserves_both_bf16_rounding_boundaries():
    partial = torch.full((1, 5120), 4096, dtype=torch.bfloat16)
    peer = -partial
    residual = torch.ones_like(partial)
    weight = torch.ones(5120, dtype=torch.bfloat16)
    calls = []

    def norm(value, actual_weight, epsilon):
        assert value.shape == (1, 1, 5120) and value.dtype == torch.bfloat16
        assert actual_weight is weight and epsilon == 1e-6
        calls.append(value.clone())
        return value * actual_weight

    actual, state = tp2_consumer_norm(partial, peer, residual, weight, 1e-6, norm)
    assert len(calls) == 1
    assert torch.equal(actual, residual) and torch.equal(state, residual)
    assert not torch.equal(state, partial + (peer + residual))


def test_consumer_continuous_residual_is_exact_and_does_not_mutate_inputs():
    generator = torch.Generator().manual_seed(516)
    residual = torch.zeros(1, 5120, dtype=torch.bfloat16)
    reference = residual.clone()
    weight = torch.randn(5120, generator=generator).to(torch.bfloat16)

    def norm(value, weight, epsilon):
        x = value.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight.float()).bfloat16()

    for _ in range(64):
        left = torch.randn(1, 5120, generator=generator).bfloat16()
        right = torch.randn(1, 5120, generator=generator).bfloat16()
        retained = residual.clone()
        actual, updated = tp2_consumer_norm(left, right, residual, weight, 1e-6, norm)
        reference = ((left.float() + right.float()).bfloat16().float() + reference.float()).bfloat16()
        assert torch.equal(updated, reference)
        assert torch.equal(actual, norm(reference.unsqueeze(0), weight, 1e-6).squeeze(0))
        assert torch.equal(residual, retained)
        residual = updated
