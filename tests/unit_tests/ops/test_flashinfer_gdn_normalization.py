# SPDX-License-Identifier: Apache-2.0
"""Independent additive-epsilon oracles for complete GDN state updates."""

import os

import pytest
import torch

from flashinfer_gaudi._reference import _l2_normalize_rsqrt, packed_recurrent_decode, recurrent_decode_from_qkv


def _oracle(q, k, v, state, log_decay, beta, normalize=True):
    # Deliberately use FP64 division/sqrt, not the implementation's rsqrt or
    # F.normalize (whose clamped-denominator contract is different).
    q, k, v, state = (value.double() for value in (q, k, v, state))
    if normalize:
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    repeat = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(repeat, dim=2)
    k = k.repeat_interleave(repeat, dim=2)
    outputs, checkpoints = [], []
    for token in range(q.shape[1]):
        state = state * log_decay[:, token].double().exp()[..., None, None]
        delta = (v[:, token] - (state * k[:, token, :, None]).sum(-1)) * beta[:, token].double()[..., None]
        state = state + delta[..., None] * k[:, token, :, None]
        outputs.append((state * q[:, token, :, None]).sum(-1) * (q.shape[-1]**-0.5))
        checkpoints.append(state.clone())
    return torch.stack(outputs, 1), torch.stack(checkpoints, 1)


@pytest.mark.parametrize("amplitude", [0.0, 1e-8, 1e-3, 1.0])
def test_l2norm_uses_additive_epsilon(amplitude):
    x = torch.linspace(-1, 1, 128).reshape(2, 64) * amplitude
    expected = x.double() / torch.sqrt(x.double().square().sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(_l2_normalize_rsqrt(x), expected.float(), atol=1e-8, rtol=2e-6)


@pytest.mark.parametrize("amplitude", [0.0, 1e-8, 1e-3, 1.0])
def test_fused_step_benchmark_uses_the_same_normalization_contract(amplitude):
    from tools.benchmark_flashinfer_gaudi_gdn_fused_decode import _normalize_packed_qk

    generator = torch.Generator().manual_seed(521)
    qk = (torch.randn(2, 32, 128, generator=generator) * amplitude).bfloat16()
    values = torch.randn(2, 16, 3, 128, generator=generator).bfloat16()
    packed = torch.cat((qk.flatten(1), values.flatten(1)), dim=-1)
    original = packed.clone()
    expected = qk.double() / torch.sqrt(qk.double().square().sum(-1, keepdim=True) + 1e-6)
    q, k, value = _normalize_packed_qk(packed)
    torch.testing.assert_close(torch.cat((q, k), dim=1), expected.float(), atol=1e-8, rtol=2e-6)
    torch.testing.assert_close(value, values.float(), atol=0, rtol=0)
    torch.testing.assert_close(packed, original, atol=0, rtol=0)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("geometry", [(2, 4, 8), (16, 48, 128), (8, 24, 128)])
@pytest.mark.parametrize("amplitude", [0.0, 1e-8, 1e-3, 0.1])
def test_packed_decode_matches_independent_small_qk_oracle(direct, geometry, amplitude):
    q_heads, value_heads, dim = geometry
    generator = torch.Generator().manual_seed(521)
    q = (torch.randn(1, 1, q_heads, dim, generator=generator) * amplitude).bfloat16()
    k = (torch.randn(q.shape, generator=generator) * amplitude).bfloat16()
    v = torch.randn(1, 1, value_heads, dim, generator=generator).bfloat16()
    state = torch.randn(1, value_heads, dim, dim, generator=generator) * 0.01
    decay = -torch.rand(1, 1, value_heads, generator=generator)
    beta = torch.sigmoid(torch.randn(decay.shape, generator=generator)).bfloat16()
    expected, checkpoints = _oracle(q, k, v, state, decay, beta)
    packed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
    output, returned = packed_recurrent_decode(packed, decay, beta, state, None, None, None, True, direct)
    assert returned is state
    torch.testing.assert_close(output, expected[:, 0].bfloat16(), atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(state, checkpoints[:, -1].float(), atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("update_final", [False, True])
def test_mtp_normalization_preserves_all_checkpoints_padding_and_final_state(normalize, update_final):
    generator = torch.Generator().manual_seed(719)
    q = torch.randn(2, 8, 2, 8, generator=generator) * 1e-3
    k = torch.randn(q.shape, generator=generator) * 1e-3
    v = torch.randn(2, 8, 4, 8, generator=generator)
    pool = torch.randn(5, 4, 8, 8, generator=generator)
    initial_pool = pool.clone()
    decay = -torch.rand(2, 8, 4, generator=generator)
    beta = torch.sigmoid(torch.randn(decay.shape, generator=generator))
    expected, checkpoints = _oracle(q[:1], k[:1], v[:1], pool[2:3], decay[:1], beta[:1], normalize)
    intermediate = torch.full((2, 8, 4, 8, 8), 17.0)
    output, returned = recurrent_decode_from_qkv(q,
                                                 k,
                                                 v,
                                                 decay,
                                                 beta,
                                                 pool,
                                                 torch.tensor([2, -1]),
                                                 torch.tensor([4, -1]),
                                                 use_qk_l2norm=normalize,
                                                 intermediate_states_buffer=intermediate,
                                                 update_final_state=update_final)
    assert returned is pool
    torch.testing.assert_close(output[:1], expected.float(), atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), atol=0, rtol=0)
    torch.testing.assert_close(intermediate[:1], checkpoints.float(), atol=2e-6, rtol=2e-4)
    torch.testing.assert_close(intermediate[1], torch.full_like(intermediate[1], 17.0), atol=0, rtol=0)
    if update_final:
        initial_pool[4] = checkpoints[0, -1].float()
    torch.testing.assert_close(pool, initial_pool, atol=2e-6, rtol=2e-4)


@pytest.mark.skipif(os.environ.get("FLASHINFER_GAUDI_RUN_HARDWARE_TESTS") != "1",
                    reason="explicit Gaudi2 hardware test opt-in required")
@pytest.mark.parametrize("tp_size", [1, 2])
def test_compiled_qwen_decode_small_qk_multistep_matches_independent_oracle(tp_size):
    import habana_frameworks.torch  # noqa: F401

    def run(packed, decay, beta, state):
        return packed_recurrent_decode(packed, decay, beta, state, None, None, None, True, True)[0]

    compiled = torch.compile(run, backend="hpu_backend", fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(829)
    key_heads, value_heads = 16 // tp_size, 48 // tp_size
    state_cpu = torch.randn(1, value_heads, 128, 128, generator=generator) * 0.01
    state = state_cpu.to("hpu")
    for step in range(32):
        amplitude = (0.0, 1e-8, 1e-3, 0.1)[step % 4]
        q = (torch.randn(1, 1, key_heads, 128, generator=generator) * amplitude).bfloat16()
        k = (torch.randn(q.shape, generator=generator) * amplitude).bfloat16()
        v = (torch.randn(1, 1, value_heads, 128, generator=generator) * 0.1).bfloat16()
        decay = -torch.rand(1, 1, value_heads, generator=generator) * 0.1
        beta = torch.sigmoid(torch.randn(decay.shape, generator=generator)).bfloat16()
        expected, checkpoints = _oracle(q, k, v, state_cpu, decay, beta)
        state_cpu = checkpoints[:, -1].float()
        packed = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), -1)
        output = compiled(packed.to("hpu"), decay.to("hpu"), beta.to("hpu"), state)
        torch.hpu.synchronize()
        torch.testing.assert_close(output.cpu(), expected[:, 0].bfloat16(), atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state.cpu(), state_cpu, atol=2e-5, rtol=2e-4)
