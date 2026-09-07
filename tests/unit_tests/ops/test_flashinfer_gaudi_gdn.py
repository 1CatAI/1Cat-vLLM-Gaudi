# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for the FlashInfer-compatible Gaudi GDN API."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from flashinfer_gaudi import clear_backend_policy_override, set_backend_policy
from flashinfer_gaudi import _native
from flashinfer_gaudi._reference import (
    _l2_normalize_rsqrt,
    packed_recurrent_decode,
    qwen38_fused_decode_step_direct,
    recurrent_decode_from_qkv,
)
from flashinfer_gaudi.gdn_decode import (
    BackendUnavailableError,
    _call_native_packed,
    gated_delta_rule_decode,
    gated_delta_rule_decode_packed,
    gated_delta_rule_decode_pretranspose,
    gated_delta_rule_mtp,
    gated_delta_rule_mtp_packed,
    gated_delta_rule_mtp_rollback,
)
from flashinfer_gaudi.gdn_fused_decode import gdn_fused_decode_step, gdn_fused_decode_step_supported
from flashinfer_gaudi.gdn_prefill import chunk_gated_delta_rule
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.flashinfer_gaudi_adapter import (
    maybe_run_gdn_decode_packed,
    maybe_run_gdn_fused_decode_step,
    maybe_run_gdn_prefill,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_fused_gdn_gating


@pytest.fixture(autouse=True)
def _reset_backend_policy():
    clear_backend_policy_override()
    yield
    clear_backend_policy_override()


def _inputs(batch=2, q_heads=2, value_heads=4, dim=8, slots=6):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(batch, 1, q_heads, dim, generator=generator)
    k = torch.randn(batch, 1, q_heads, dim, generator=generator)
    v = torch.randn(batch, 1, value_heads, dim, generator=generator)
    state = torch.randn(slots, value_heads, dim, dim, generator=generator)
    log_decay = -torch.rand(batch, 1, value_heads, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, 1, value_heads, generator=generator))
    return q, k, v, state, log_decay, beta


def _qwen38_mtp_native_inputs(batch=1, value_heads=48):
    tokens = 8
    packed = torch.zeros(batch, tokens, 10240, dtype=torch.bfloat16)
    log_decay = torch.zeros(batch, tokens, value_heads, dtype=torch.float32)
    beta = torch.zeros(batch, tokens, value_heads, dtype=torch.bfloat16)
    state = torch.zeros(1, value_heads, 128, 128, dtype=torch.float32)
    state_indices = torch.zeros(batch, tokens, dtype=torch.int32)
    accepted = torch.ones(batch, dtype=torch.int32)
    query_lengths = torch.full((batch, ), tokens, dtype=torch.int32)
    return packed, log_decay, beta, state, state_indices, accepted, query_lengths


def _naive_step(q, k, v, state, log_decay, beta):
    repeat = v.shape[2] // q.shape[2]
    q, k = q.float(), k.float()
    q = (q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)).repeat_interleave(repeat, dim=2)
    k = (k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)).repeat_interleave(repeat, dim=2)
    scale = q.shape[-1]**-0.5
    decayed = state * torch.exp(log_decay[:, 0]).unsqueeze(-1).unsqueeze(-1)
    projection = torch.matmul(decayed, k[:, 0].unsqueeze(-1)).squeeze(-1)
    delta = (v[:, 0].float() - projection) * beta[:, 0].unsqueeze(-1)
    updated = decayed + delta.unsqueeze(-1) * k[:, 0].unsqueeze(-2)
    output = torch.matmul(updated, (q[:, 0] * scale).unsqueeze(-1)).squeeze(-1)
    return output, updated


@pytest.mark.parametrize("magnitude", [0.0, 1e-6, 1e-4, 0.1, 1.0])
def test_gdn_normalization_adds_epsilon_to_squared_norm(magnitude):
    value = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32) * magnitude
    expected = value / torch.sqrt(torch.sum(value.square(), dim=-1, keepdim=True) + 1e-6)
    actual = _l2_normalize_rsqrt(value)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-8)
    if 0 < magnitude <= 1e-4:
        assert not torch.allclose(actual, F.normalize(value, dim=-1, eps=1e-6))


@pytest.mark.parametrize("magnitude", [0.0, 1e-4, 0.1])
def test_dflash_verify_small_qk_preserves_every_accepted_state(magnitude):
    generator = torch.Generator().manual_seed(517)
    q = torch.randn(1, 8, 2, 8, generator=generator) * magnitude
    k = torch.randn(1, 8, 2, 8, generator=generator) * magnitude
    v = torch.randn(1, 8, 4, 8, generator=generator) * 0.1
    g = -torch.rand(1, 8, 4, generator=generator) * 0.1
    beta = torch.rand(1, 8, 4, generator=generator)
    pool = torch.randn(10, 4, 8, 8, generator=generator) * 0.01
    expected_pool = pool.clone()
    packed = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
    indices = torch.tensor([[8, 2, 4, 6, 1, 7, 3, 5]], dtype=torch.int32)
    lengths = torch.tensor([8], dtype=torch.int32)
    for accepted in range(1, 9):
        state = expected_pool[indices[0, accepted - 1]:indices[0, accepted - 1] + 1].clone()
        outputs = []
        for token in range(8):
            output, state = _naive_step(q[:, token:token + 1], k[:, token:token + 1], v[:, token:token + 1], state,
                                        g[:, token:token + 1], beta[:, token:token + 1])
            expected_pool[indices[0, token]] = state[0]
            outputs.append(output)
        actual, _ = gated_delta_rule_mtp_packed(packed,
                                                g,
                                                beta,
                                                pool,
                                                indices,
                                                torch.tensor([accepted], dtype=torch.int32),
                                                lengths,
                                                assume_full_query=True)
        torch.testing.assert_close(actual, torch.stack(outputs, dim=1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pool, expected_pool, rtol=1e-5, atol=1e-6)


def test_prefill_signature_tracks_flashinfer_contract():
    parameters = inspect.signature(chunk_gated_delta_rule).parameters
    assert tuple(parameters) == (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "use_qk_l2norm_in_kernel",
        "output",
        "output_state",
        "state_checkpoints",
        "checkpoint_cu_starts",
        "checkpoint_every_n_tokens",
        "use_cp",
        "state_indices",
        "_cp_chunk_len",
    )


def test_prefill_public_alpha_contract_matches_log_gate_reference():
    generator = torch.Generator().manual_seed(11)
    tokens, q_heads, value_heads, dim = 16, 2, 4, 8
    q = torch.randn(tokens, q_heads, dim, generator=generator) * 0.1
    k = torch.randn(tokens, q_heads, dim, generator=generator) * 0.1
    v = torch.randn(tokens, value_heads, dim, generator=generator) * 0.1
    log_decay = -torch.rand(tokens, value_heads, generator=generator) * 0.02
    beta = torch.sigmoid(torch.randn(tokens, value_heads, generator=generator))
    initial_state = torch.randn(1, value_heads, dim, dim, generator=generator) * 0.01
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32)

    output, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=torch.exp(log_decay),
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    assert output.shape == (tokens, value_heads, dim)
    assert final_state.shape == initial_state.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()


def test_prefill_public_default_state_remains_fp32():
    tokens, q_heads, value_heads, dim = 16, 2, 4, 8
    q = torch.zeros(tokens, q_heads, dim, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros(tokens, value_heads, dim, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32)

    output, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32


def test_prefill_rejects_unimplemented_context_parallel_state():
    q = torch.zeros(8, 2, 8)
    k = torch.zeros_like(q)
    v = torch.zeros(8, 4, 8)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="Context-parallel"):
        chunk_gated_delta_rule(q, k, v, cu_seqlens=cu_seqlens, use_cp=True)


def test_pretranspose_signature_tracks_flashinfer_contract():
    parameters = inspect.signature(gated_delta_rule_decode_pretranspose).parameters
    assert tuple(parameters) == (
        "q",
        "k",
        "v",
        "state",
        "A_log",
        "a",
        "dt_bias",
        "b",
        "scale",
        "output",
        "use_qk_l2norm",
        "initial_state",
        "initial_state_indices",
        "output_state_indices",
    )


def test_k_major_signature_tracks_flashinfer_contract():
    parameters = inspect.signature(gated_delta_rule_decode).parameters
    assert tuple(parameters) == (
        "q",
        "k",
        "v",
        "state",
        "A_log",
        "a",
        "dt_bias",
        "b",
        "scale",
        "output",
        "use_qk_l2norm",
    )


def test_k_major_decode_matches_vk_reference():
    generator = torch.Generator().manual_seed(29)
    batch, q_heads, value_heads, key_dim, value_dim = 2, 2, 4, 8, 6
    q = torch.randn(batch, 1, q_heads, key_dim, generator=generator)
    k = torch.randn(batch, 1, q_heads, key_dim, generator=generator)
    v = torch.randn(batch, 1, value_heads, value_dim, generator=generator)
    state_kv = torch.randn(batch, value_heads, key_dim, value_dim, generator=generator)
    expected_state_vk = state_kv.transpose(-1, -2).contiguous()
    A_log = torch.randn(value_heads, generator=generator)
    a = torch.randn(batch, 1, value_heads, generator=generator)
    dt_bias = torch.randn(value_heads, generator=generator)
    b = torch.randn(batch, 1, value_heads, generator=generator)

    expected_output, _ = gated_delta_rule_decode_pretranspose(
        q,
        k,
        v,
        expected_state_vk,
        A_log,
        a,
        dt_bias,
        b,
    )
    output, returned_state = gated_delta_rule_decode(
        q,
        k,
        v,
        state_kv,
        A_log,
        a,
        dt_bias,
        b,
    )

    assert returned_state is state_kv
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(state_kv, expected_state_vk.transpose(-1, -2))


def test_fused_decode_signature_tracks_flashinfer_contract():
    parameters = inspect.signature(gdn_fused_decode_step).parameters
    assert tuple(parameters) == (
        "hidden_states",
        "w_ba",
        "mixed_qkv",
        "conv_weight",
        "conv_bias",
        "conv_state",
        "A_log",
        "dt_bias",
        "scale",
        "ssm_state",
        "state_indices",
        "use_qk_l2norm",
        "out",
    )


def test_fused_decode_support_probe_does_not_promote_reference():
    assert not gdn_fused_decode_step_supported(1, device="cpu")
    assert not gdn_fused_decode_step_supported(16, device="cpu")
    assert not gdn_fused_decode_step_supported(1, hidden_size=4096, device="cpu")


def test_fused_decode_updates_live_rows_and_skips_padding():
    generator = torch.Generator().manual_seed(113)
    batch, slots = 3, 4
    hidden_size, qk_heads, value_heads, dim, width = 6, 2, 4, 4, 4
    qkv_dim = (2 * qk_heads + value_heads) * dim
    hidden_states = torch.randn(batch, hidden_size, dtype=torch.bfloat16, generator=generator)
    w_ba = torch.randn(hidden_size, 2 * value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    mixed_qkv = torch.randn(batch, qkv_dim, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_weight = torch.randn(qkv_dim, width, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_bias = torch.randn(qkv_dim, dtype=torch.bfloat16, generator=generator) * 0.1
    # Exercise FlashInfer's logical [P,D,S] view over vLLM's physical SD rows.
    conv_state_sd = torch.randn(slots, width - 1, qkv_dim, dtype=torch.bfloat16, generator=generator) * 0.1
    ssm_state = torch.randn(slots, value_heads, dim, dim, generator=generator) * 0.1
    A_log = torch.randn(value_heads, generator=generator) * 0.1
    dt_bias = torch.randn(value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    state_indices = torch.tensor([2, -1, 0], dtype=torch.int32)

    expected_conv = conv_state_sd.clone()
    expected_ssm = ssm_state.clone()
    expected, _, _ = gdn_fused_decode_step(
        hidden_states[[0, 2]],
        w_ba,
        mixed_qkv[[0, 2]],
        conv_weight,
        conv_bias,
        expected_conv.transpose(-1, -2),
        A_log,
        dt_bias,
        None,
        expected_ssm,
        torch.tensor([2, 0], dtype=torch.int32),
    )

    output_buffer = torch.empty(batch, 1, value_heads, dim, dtype=torch.bfloat16)
    actual, returned_conv, returned_ssm = gdn_fused_decode_step(
        hidden_states,
        w_ba,
        mixed_qkv,
        conv_weight,
        conv_bias,
        conv_state_sd.transpose(-1, -2),
        A_log,
        dt_bias,
        0.0,
        ssm_state,
        state_indices,
        out=output_buffer,
    )

    assert actual is output_buffer
    assert returned_conv.untyped_storage().data_ptr() == conv_state_sd.untyped_storage().data_ptr()
    assert returned_ssm is ssm_state
    torch.testing.assert_close(actual[[0, 2]], expected)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))
    torch.testing.assert_close(conv_state_sd, expected_conv)
    torch.testing.assert_close(ssm_state, expected_ssm)


def test_mtp_signature_tracks_flashinfer_contract():
    parameters = inspect.signature(gated_delta_rule_mtp).parameters
    assert tuple(parameters) == (
        "q",
        "k",
        "v",
        "initial_state",
        "initial_state_indices",
        "A_log",
        "a",
        "dt_bias",
        "b",
        "scale",
        "output",
        "intermediate_states_buffer",
        "ssm_state_indices",
        "disable_state_update",
        "use_qk_l2norm",
        "output_state_indices",
    )


def test_mtp_tracks_intermediate_and_distinct_final_state():
    generator = torch.Generator().manual_seed(41)
    batch, tokens, q_heads, value_heads, dim = 1, 3, 2, 4, 8
    q = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    k = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    v = torch.randn(batch, tokens, value_heads, dim, generator=generator)
    pool = torch.randn(6, value_heads, dim, dim, generator=generator)
    original = pool.clone()
    A_log = torch.randn(value_heads, generator=generator)
    a = torch.randn(batch, tokens, value_heads, generator=generator)
    dt_bias = torch.randn(value_heads, generator=generator)
    b = torch.randn(batch, tokens, value_heads, generator=generator)
    intermediate = torch.zeros(batch, tokens, value_heads, dim, dim)
    load = torch.tensor([1], dtype=torch.int32)
    store = torch.tensor([4], dtype=torch.int32)

    output, returned_pool = gated_delta_rule_mtp(
        q,
        k,
        v,
        pool,
        load,
        A_log,
        a,
        dt_bias,
        b,
        intermediate_states_buffer=intermediate,
        disable_state_update=False,
        output_state_indices=store,
    )

    assert returned_pool is pool
    assert output.shape == (batch, tokens, value_heads, dim)
    torch.testing.assert_close(pool[1], original[1])
    torch.testing.assert_close(pool[4], intermediate[0, -1])
    assert not torch.equal(intermediate[:, 0], intermediate[:, -1])


def test_mtp_rollback_selects_checkpoint_and_masks_bucket_padding():
    generator = torch.Generator().manual_seed(211)
    batch, tokens, q_heads, value_heads, dim = 2, 3, 2, 4, 8
    q = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    k = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    v = torch.randn(batch, tokens, value_heads, dim, generator=generator)
    pool = torch.randn(8, value_heads, dim, dim, generator=generator)
    original = pool.clone()
    log_decay = -torch.rand(batch, tokens, value_heads, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, tokens, value_heads, generator=generator))
    state_indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([1, 2], dtype=torch.int32)
    token_mask = torch.tensor([[True, True, True], [True, False, False]])

    expected_outputs = torch.zeros(batch, tokens, value_heads, dim)
    expected_checkpoints: dict[int, torch.Tensor] = {}
    for row, initial_slot in enumerate((1, 5)):
        state = original[initial_slot:initial_slot + 1]
        for token in range(tokens):
            if not token_mask[row, token]:
                continue
            output, state = _naive_step(
                q[row:row + 1, token:token + 1],
                k[row:row + 1, token:token + 1],
                v[row:row + 1, token:token + 1],
                state,
                log_decay[row:row + 1, token:token + 1],
                beta[row:row + 1, token:token + 1],
            )
            expected_outputs[row, token] = output[0]
            expected_checkpoints[int(state_indices[row, token])] = state[0]

    output, returned_pool = gated_delta_rule_mtp_rollback(
        q,
        k,
        v,
        pool,
        state_indices,
        accepted,
        log_decay,
        beta,
        token_mask,
    )

    assert returned_pool is pool
    torch.testing.assert_close(output, expected_outputs)
    for slot, expected in expected_checkpoints.items():
        torch.testing.assert_close(pool[slot], expected)
    torch.testing.assert_close(pool[5], original[5])
    torch.testing.assert_close(pool[6], original[6])
    torch.testing.assert_close(pool[0], original[0])


def test_mtp_rollback_matches_committed_state_across_multiple_rounds():
    generator = torch.Generator().manual_seed(219)
    batch, tokens, q_heads, value_heads, dim = 1, 4, 2, 4, 8
    pool = torch.zeros(tokens + 1, value_heads, dim, dim)
    pool[1] = torch.randn(value_heads, dim, dim, generator=generator)
    state_indices = torch.arange(1, tokens + 1, dtype=torch.int32).view(1, tokens)
    committed_state = pool[1:2].clone()
    previous_accepted = 1

    for current_accepted in (3, 1, 4, 2):
        q = torch.randn(batch, tokens, q_heads, dim, generator=generator)
        k = torch.randn(batch, tokens, q_heads, dim, generator=generator)
        v = torch.randn(batch, tokens, value_heads, dim, generator=generator)
        log_decay = -torch.rand(batch, tokens, value_heads, generator=generator)
        beta = torch.sigmoid(torch.randn(batch, tokens, value_heads, generator=generator))

        expected_state = committed_state.clone()
        expected_output, _ = recurrent_decode_from_qkv(
            q,
            k,
            v,
            log_decay,
            beta,
            expected_state,
            direct_state=True,
        )
        actual_output, _ = gated_delta_rule_mtp_rollback(
            q,
            k,
            v,
            pool,
            state_indices,
            torch.tensor([previous_accepted], dtype=torch.int32),
            log_decay,
            beta,
        )
        torch.testing.assert_close(actual_output, expected_output)

        committed_state = committed_state.clone()
        recurrent_decode_from_qkv(
            q[:, :current_accepted],
            k[:, :current_accepted],
            v[:, :current_accepted],
            log_decay[:, :current_accepted],
            beta[:, :current_accepted],
            committed_state,
            direct_state=True,
        )
        torch.testing.assert_close(
            pool[state_indices[0, current_accepted - 1]],
            committed_state[0],
        )
        previous_accepted = current_accepted


def test_packed_mtp_fallback_matches_rollback_api():
    generator = torch.Generator().manual_seed(223)
    batch, tokens, q_heads, value_heads, dim = 2, 3, 2, 4, 8
    q = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    k = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    v = torch.randn(batch, tokens, value_heads, dim, generator=generator)
    packed = torch.cat(
        (q.flatten(2), k.flatten(2), v.flatten(2)),
        dim=-1,
    )
    expected_pool = torch.randn(8, value_heads, dim, dim, generator=generator)
    actual_pool = expected_pool.clone()
    state_indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([1, 2], dtype=torch.int32)
    query_lengths = torch.tensor([3, 1], dtype=torch.int32)
    log_decay = -torch.rand(batch, tokens, value_heads, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, tokens, value_heads, generator=generator))
    token_mask = torch.arange(tokens).unsqueeze(0) < query_lengths.unsqueeze(1)

    expected, _ = gated_delta_rule_mtp_rollback(
        q,
        k,
        v,
        expected_pool,
        state_indices,
        accepted,
        log_decay,
        beta,
        token_mask,
    )
    actual, returned_pool = gated_delta_rule_mtp_packed(
        packed,
        log_decay,
        beta,
        actual_pool,
        state_indices,
        accepted,
        query_lengths,
    )

    assert returned_pool is actual_pool
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_pool, expected_pool)


def test_packed_mtp_full_query_fast_path_is_exact():
    generator = torch.Generator().manual_seed(227)
    batch, tokens, q_heads, value_heads, dim = 2, 3, 2, 4, 8
    q = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    k = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    v = torch.randn(batch, tokens, value_heads, dim, generator=generator)
    packed = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
    reference_pool = torch.randn(8, value_heads, dim, dim, generator=generator)
    fast_pool = reference_pool.clone()
    state_indices = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([1, 2], dtype=torch.int32)
    query_lengths = torch.full((batch, ), tokens, dtype=torch.int32)
    log_decay = -torch.rand(batch, tokens, value_heads, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, tokens, value_heads, generator=generator))

    reference, _ = gated_delta_rule_mtp_packed(
        packed,
        log_decay,
        beta,
        reference_pool,
        state_indices,
        accepted,
        query_lengths,
    )
    fast, returned_pool = gated_delta_rule_mtp_packed(
        packed,
        log_decay,
        beta,
        fast_pool,
        state_indices,
        accepted,
        query_lengths,
        assume_full_query=True,
    )

    assert returned_pool is fast_pool
    torch.testing.assert_close(fast, reference, rtol=0, atol=0)
    torch.testing.assert_close(fast_pool, reference_pool, rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("use_qk_l2norm", [False, True])
@pytest.mark.parametrize("scale", [None, 0.25])
def test_full_query_preparation_preserves_every_rollback_checkpoint(batch, use_qk_l2norm, scale):
    generator = torch.Generator().manual_seed(331)
    tokens, q_heads, value_heads, dim = 8, 2, 6, 8
    q = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    k = torch.randn(batch, tokens, q_heads, dim, generator=generator)
    v = torch.randn(batch, tokens, value_heads, dim, generator=generator)
    packed = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
    # Leave guard rows untouched and use non-contiguous slot ownership.
    indices = (torch.randperm(batch * tokens, generator=generator) + 1).to(torch.int32).reshape(batch, tokens)
    reference_pool = torch.randn(batch * tokens + 2, value_heads, dim, dim, generator=generator)
    fast_pool = reference_pool.clone()
    query_lengths = torch.full((batch, ), tokens, dtype=torch.int32)
    log_decay = -torch.rand(batch, tokens, value_heads, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, tokens, value_heads, generator=generator))

    for checkpoint in range(1, tokens + 1):
        accepted = torch.full((batch, ), checkpoint, dtype=torch.int32)
        expected, _ = gated_delta_rule_mtp_packed(
            packed,
            log_decay,
            beta,
            reference_pool,
            indices,
            accepted,
            query_lengths,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
        )
        actual, returned_pool = gated_delta_rule_mtp_packed(
            packed,
            log_decay,
            beta,
            fast_pool,
            indices,
            accepted,
            query_lengths,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            assume_full_query=True,
        )
        assert returned_pool is fast_pool
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(fast_pool, reference_pool, rtol=0, atol=0)


@pytest.mark.parametrize("policy,compiling,batch,full_query,selected", [
    ("auto", True, 1, True, True),
    ("pytorch", True, 1, True, False),
    ("auto", False, 1, True, False),
    ("auto", True, 2, True, False),
    ("auto", True, 1, False, False),
])
def test_prepared_mtp_opt_in_is_scoped(policy, compiling, batch, full_query, selected):
    inputs = _qwen38_mtp_native_inputs(batch=batch)
    sentinel = torch.zeros(batch, 8, 48, 128, dtype=torch.bfloat16)
    prepared = mock.Mock(return_value=(sentinel, inputs[3]))
    reference = mock.Mock(return_value=(sentinel, inputs[3]))
    with (
            mock.patch("flashinfer_gaudi.gdn_decode.get_backend_policy", return_value=policy),
            mock.patch("flashinfer_gaudi.gdn_decode.mtp_prepared_enabled", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode._device_is_hpu", return_value=True),
            mock.patch("torch.compiler.is_compiling", return_value=compiling),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_auto_promoted", return_value=False),
            mock.patch("flashinfer_gaudi.gdn_decode._gated_delta_rule_mtp_prepared", prepared),
            mock.patch("flashinfer_gaudi.gdn_decode._gated_delta_rule_mtp_packed_reference", reference),
    ):
        output, pool = gated_delta_rule_mtp_packed(*inputs, assume_full_query=full_query)
    assert output is sentinel and pool is inputs[3]
    assert prepared.call_count == int(selected)
    assert reference.call_count == int(not selected)


def test_prepared_mtp_missing_extension_does_not_mutate_state():
    from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_prepared
    inputs = _qwen38_mtp_native_inputs()
    before = inputs[3].clone()
    with (
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_prepared_op", return_value=None),
            pytest.raises(BackendUnavailableError, match="prepared MTP"),
    ):
        _gated_delta_rule_mtp_prepared(*inputs[:-1])
    torch.testing.assert_close(inputs[3], before, rtol=0, atol=0)


@pytest.mark.parametrize("distinct", [False, True])
def test_prepared_mtp_preserves_checkpoint_write_ownership(distinct):
    from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_prepared
    inputs = list(_qwen38_mtp_native_inputs())
    inputs[3] = torch.full((10, 48, 128, 128), -1.0)
    inputs[4] = (torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
                 if distinct else torch.ones(1, 8, dtype=torch.int32))
    checkpoints = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8, 1, 1, 1).expand(1, 8, 48, 128, 128)
    native = mock.Mock(return_value=(torch.zeros(1, 8, 48, 128), checkpoints))
    with mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_prepared_op", return_value=native):
        _, returned_pool = _gated_delta_rule_mtp_prepared(*inputs[:-1], assume_distinct_checkpoints=distinct)
    assert returned_pool is inputs[3]
    assert torch.all(returned_pool[0] == -1) and torch.all(returned_pool[-1] == -1)
    if distinct:
        torch.testing.assert_close(returned_pool[1:9], checkpoints[0], rtol=0, atol=0)
    else:
        assert torch.all(returned_pool[1] == 8)
        assert torch.all(returned_pool[2:] == -1)


@pytest.mark.parametrize("start", [1, 9])
def test_prepared_direct_checkpoints_preserve_accepted_load_and_guard_rows(start):
    from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_prepared
    inputs = list(_qwen38_mtp_native_inputs())
    inputs[3] = torch.arange(18, dtype=torch.float32).reshape(18, 1, 1, 1).expand(18, 48, 128, 128).clone()
    before = inputs[3].clone()
    inputs[4] = torch.arange(start, start + 8, dtype=torch.int32).reshape(1, 8)
    inputs[5].fill_(4)
    checkpoints = torch.arange(101, 109, dtype=torch.float32).reshape(1, 8, 1, 1, 1).expand(1, 8, 48, 128, 128)
    native = mock.Mock(return_value=(torch.zeros(1, 8, 48, 128), checkpoints))
    with mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_prepared_op", return_value=native):
        _, pool = _gated_delta_rule_mtp_prepared(*inputs[:-1], assume_distinct_checkpoints=True, checkpoint_start=start)
    assert pool is inputs[3]
    torch.testing.assert_close(native.call_args.args[0], before[start + 3:start + 4], rtol=0, atol=0)
    torch.testing.assert_close(pool[start:start + 8], checkpoints[0], rtol=0, atol=0)
    torch.testing.assert_close(pool[:start], before[:start], rtol=0, atol=0)
    torch.testing.assert_close(pool[start + 8:], before[start + 8:], rtol=0, atol=0)


@pytest.mark.parametrize("start,distinct", [(0, True), (-1, True), (3, True), (1, False)])
def test_prepared_direct_checkpoints_reject_invalid_static_destination_before_native_call(start, distinct):
    from flashinfer_gaudi.gdn_decode import _gated_delta_rule_mtp_prepared
    inputs = list(_qwen38_mtp_native_inputs())
    inputs[3] = torch.zeros(10, 48, 128, 128)
    before = inputs[3].clone()
    native = mock.Mock()
    with (mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_prepared_op",
                     return_value=native), pytest.raises(ValueError, match="caller-proven")):
        _gated_delta_rule_mtp_prepared(*inputs[:-1], assume_distinct_checkpoints=distinct, checkpoint_start=start)
    native.assert_not_called()
    torch.testing.assert_close(inputs[3], before, rtol=0, atol=0)


def test_mtp_auto_does_not_dispatch_unpromoted_native_kernel():
    inputs = _qwen38_mtp_native_inputs()
    sentinel = torch.zeros(1, 8, 48, 128, dtype=torch.bfloat16)
    native = mock.Mock(return_value=sentinel)
    fallback = mock.Mock(return_value=(sentinel, inputs[3]))

    with (
            mock.patch("flashinfer_gaudi.gdn_decode._device_is_hpu", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_auto_promoted", return_value=False),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_op", return_value=native),
            mock.patch("flashinfer_gaudi.gdn_decode.gated_delta_rule_mtp_rollback", fallback),
    ):
        output, returned_pool = gated_delta_rule_mtp_packed(*inputs)

    assert output is sentinel
    assert returned_pool is inputs[3]
    native.assert_not_called()
    fallback.assert_called_once()


def test_mtp_auto_dispatches_only_after_independent_promotion():
    inputs = _qwen38_mtp_native_inputs(batch=2)
    sentinel = torch.zeros(2, 8, 48, 128, dtype=torch.bfloat16)
    native = mock.Mock(return_value=sentinel)

    with (
            mock.patch("flashinfer_gaudi.gdn_decode._device_is_hpu", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_auto_promoted", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_op", return_value=native),
    ):
        output, returned_pool = gated_delta_rule_mtp_packed(*inputs)

    assert output is sentinel
    assert returned_pool is inputs[3]
    native.assert_called_once()
    assert native.call_args.args[2].dtype == torch.float32


def test_mtp_auto_keeps_single_request_on_reference_after_promotion():
    inputs = _qwen38_mtp_native_inputs()
    sentinel = torch.zeros(1, 8, 48, 128, dtype=torch.bfloat16)
    native = mock.Mock(return_value=sentinel)
    fallback = mock.Mock(return_value=(sentinel, inputs[3]))

    with (
            mock.patch("flashinfer_gaudi.gdn_decode._device_is_hpu", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_auto_promoted", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_op", return_value=native),
            mock.patch("flashinfer_gaudi.gdn_decode.gated_delta_rule_mtp_rollback", fallback),
    ):
        output, returned_pool = gated_delta_rule_mtp_packed(*inputs)

    assert output is sentinel
    assert returned_pool is inputs[3]
    native.assert_not_called()
    fallback.assert_called_once()


def test_mtp_auto_keeps_compiled_model_on_reference_after_promotion():
    inputs = _qwen38_mtp_native_inputs(batch=2)
    sentinel = torch.zeros(2, 8, 48, 128, dtype=torch.bfloat16)
    native = mock.Mock(return_value=sentinel)
    fallback = mock.Mock(return_value=(sentinel, inputs[3]))

    with (
            mock.patch("flashinfer_gaudi.gdn_decode._device_is_hpu", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_auto_promoted", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_op", return_value=native),
            mock.patch("flashinfer_gaudi.gdn_decode.gated_delta_rule_mtp_rollback", fallback),
            mock.patch("torch.compiler.is_compiling", return_value=True),
    ):
        output, returned_pool = gated_delta_rule_mtp_packed(*inputs)

    assert output is sentinel
    assert returned_pool is inputs[3]
    native.assert_not_called()
    fallback.assert_called_once()


def test_forced_mtp_native_rejects_non_qwen38_head_shape():
    inputs = _qwen38_mtp_native_inputs(value_heads=47)
    native = mock.Mock()
    set_backend_policy("public")

    with (
            mock.patch("flashinfer_gaudi.gdn_decode._device_is_hpu", return_value=True),
            mock.patch("flashinfer_gaudi.gdn_decode.public_mtp_gdn_op", return_value=native),
            pytest.raises(BackendUnavailableError, match="unavailable for this runtime or shape"),
    ):
        gated_delta_rule_mtp_packed(*inputs)

    native.assert_not_called()


def test_grouped_qk_reference_preserves_update_order():
    q, k, v, pool, log_decay, beta = _inputs()
    indices = torch.tensor([4, 1])
    original = pool.clone()
    expected_output, expected_state = _naive_step(
        q,
        k,
        v,
        original.index_select(0, indices),
        log_decay,
        beta,
    )

    packed = torch.cat((q.reshape(2, -1), k.reshape(2, -1), v.reshape(2, -1)), dim=-1)
    output, returned_pool = gated_delta_rule_decode_packed(
        packed,
        log_decay,
        beta,
        pool,
        indices,
        indices,
    )

    assert returned_pool is pool
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(pool.index_select(0, indices), expected_state)
    untouched = torch.tensor([0, 2, 3, 5])
    torch.testing.assert_close(pool.index_select(0, untouched), original.index_select(0, untouched))


def test_distinct_load_and_store_indices():
    q, k, v, pool, log_decay, beta = _inputs(batch=1)
    original = pool.clone()
    load = torch.tensor([2])
    store = torch.tensor([5])
    packed = torch.cat((q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)), dim=-1)
    _, returned_pool = gated_delta_rule_decode_packed(packed, log_decay, beta, pool, load, store)

    assert returned_pool is pool
    torch.testing.assert_close(pool[2], original[2])
    assert not torch.equal(pool[5], original[5])


def test_negative_padding_does_not_mutate_any_slot():
    q, k, v, pool, log_decay, beta = _inputs()
    original = pool.clone()
    load = torch.tensor([-1, 3])
    store = torch.tensor([-1, 3])
    packed = torch.cat((q.reshape(2, -1), k.reshape(2, -1), v.reshape(2, -1)), dim=-1)
    output, _ = gated_delta_rule_decode_packed(packed, log_decay, beta, pool, load, store)

    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    torch.testing.assert_close(pool[0], original[0])
    unchanged = torch.tensor([0, 1, 2, 4, 5])
    torch.testing.assert_close(pool.index_select(0, unchanged), original.index_select(0, unchanged))


def test_padding_write_cannot_clobber_live_slot_zero():
    q, k, v, pool, log_decay, beta = _inputs()
    original = pool.clone()
    expected_output, expected_state = _naive_step(
        q[:1],
        k[:1],
        v[:1],
        original[:1],
        log_decay[:1],
        beta[:1],
    )
    packed = torch.cat((q.reshape(2, -1), k.reshape(2, -1), v.reshape(2, -1)), dim=-1)
    output, _ = gated_delta_rule_decode_packed(
        packed,
        log_decay,
        beta,
        pool,
        torch.tensor([0, -1]),
        torch.tensor([0, -1]),
    )

    torch.testing.assert_close(output[0], expected_output[0])
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    torch.testing.assert_close(pool[0], expected_state[0])


def test_duplicate_output_indices_are_rejected_before_update():
    q, k, v, pool, log_decay, beta = _inputs()
    original = pool.clone()
    indices = torch.tensor([2, 2])
    packed = torch.cat((q.reshape(2, -1), k.reshape(2, -1), v.reshape(2, -1)), dim=-1)
    with pytest.raises(ValueError, match="must be unique"):
        gated_delta_rule_decode_packed(packed, log_decay, beta, pool, indices, indices)
    torch.testing.assert_close(pool, original)


def test_output_buffer_and_gating_api():
    q, k, v, pool, _, _ = _inputs()
    A_log = torch.zeros(4)
    a = torch.randn(2, 1, 4)
    b = torch.randn(2, 1, 4)
    dt_bias = torch.zeros(4)
    output_buffer = torch.empty_like(v)
    result, returned_pool = gated_delta_rule_decode_pretranspose(
        q,
        k,
        v,
        pool,
        A_log,
        a,
        dt_bias,
        b,
        output=output_buffer,
        initial_state_indices=torch.tensor([1, 3]),
        output_state_indices=torch.tensor([1, 3]),
    )
    assert result is output_buffer
    assert returned_pool is pool
    assert torch.isfinite(result).all()


def test_forced_native_backend_fails_before_cpu_state_update():
    q, k, v, pool, log_decay, beta = _inputs(batch=1)
    original = pool.clone()
    packed = torch.cat((q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)), dim=-1)
    set_backend_policy("public")
    with pytest.raises(BackendUnavailableError, match="decomposition is forbidden"):
        gated_delta_rule_decode_packed(
            packed,
            log_decay,
            beta,
            pool,
            torch.tensor([1]),
            torch.tensor([1]),
        )
    torch.testing.assert_close(pool, original)


def test_legacy_native_contract_receives_fp32_compatibility_inputs():

    class LegacyOp:
        _qualified_op_name = "custom_op::custom_gdn_packed_decode_f32_gaudi2"

        def __call__(self, state, packed, decay, beta, indices):
            assert state.dtype == torch.float32
            assert packed.dtype == torch.float32
            assert decay.dtype == torch.float32
            assert beta.dtype == torch.float32
            assert indices.dtype == torch.int32
            return state[:packed.shape[0]], torch.zeros(packed.shape[0], 4, 8)

    q, k, v, pool, log_decay, beta = _inputs(batch=1)
    packed = torch.cat((q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)), dim=-1).to(torch.bfloat16)
    output = _call_native_packed(
        LegacyOp(),
        packed,
        log_decay,
        beta.to(torch.bfloat16),
        pool,
        torch.tensor([1]),
    )
    assert output.dtype == torch.bfloat16


def test_auto_policy_uses_reference_on_cpu():
    q, k, v, pool, log_decay, beta = _inputs(batch=1)
    packed = torch.cat((q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)), dim=-1)
    with mock.patch.dict(os.environ, {"FLASHINFER_GAUDI_BACKEND": "auto"}):
        output, _ = gated_delta_rule_decode_packed(
            packed,
            log_decay,
            beta,
            pool,
            torch.tensor([1]),
            torch.tensor([1]),
        )
    assert output.shape == (1, 4, 8)


def test_native_loader_adds_kernel_database_file_to_gc_path(tmp_path, monkeypatch):
    package_dir = tmp_path / "flashinfer_gaudi"
    library_dir = package_dir / "lib"
    library_dir.mkdir(parents=True)
    kernel_database = library_dir / "libflashinfer_gaudi_kernels.so"
    kernel_database.touch()

    monkeypatch.setattr(_native, "__file__", str(package_dir / "_native.py"))
    monkeypatch.delenv("GC_KERNEL_PATH", raising=False)
    _native._configure_kernel_database_path()

    configured = os.environ["GC_KERNEL_PATH"].split(os.pathsep)
    assert configured[0] == str(kernel_database)
    system_kernel_database = Path("/usr/lib/habanalabs/libtpc_kernels.so")
    if system_kernel_database.is_file():
        assert configured[-1] == str(system_kernel_database)


def test_recurrent_state_is_continuous_across_many_steps():
    q, k, v, pool, log_decay, beta = _inputs(batch=1)
    reference_pool = pool.clone()
    indices = torch.tensor([2])
    packed = torch.cat((q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)), dim=-1)

    outputs = []
    expected_outputs = []
    for _ in range(32):
        output, _ = gated_delta_rule_decode_packed(
            packed,
            log_decay,
            beta,
            pool,
            indices,
            indices,
        )
        expected_output, expected_state = _naive_step(
            q,
            k,
            v,
            reference_pool.index_select(0, indices),
            log_decay,
            beta,
        )
        reference_pool.index_copy_(0, indices, expected_state)
        outputs.append(output)
        expected_outputs.append(expected_output)

    torch.testing.assert_close(torch.stack(outputs), torch.stack(expected_outputs))
    torch.testing.assert_close(pool, reference_pool)


def test_vllm_adapter_uses_direct_group_state_view():
    q, k, v, _, log_decay, beta = _inputs(batch=2)
    groups, max_requests = 3, 4
    pool = torch.randn(groups * max_requests + 2, 4, 8, 8)
    original = pool.clone()
    expected_pool = pool.clone()
    packed = torch.cat((q.reshape(2, -1), k.reshape(2, -1), v.reshape(2, -1)), dim=-1)
    load_store = torch.tensor([max_requests + 1, max_requests + 2], dtype=torch.int32)
    expected_output, _ = packed_recurrent_decode(
        packed,
        log_decay,
        beta,
        expected_pool,
        load_store,
        load_store,
        None,
        True,
    )

    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN": "1"}):
        result = maybe_run_gdn_decode_packed(
            mixed_qkv=packed,
            log_decay=log_decay,
            beta=beta,
            state_pool=pool,
            load_state_indices=load_store,
            store_state_indices=load_store,
            use_qk_l2norm=True,
            direct_state_layout=True,
            direct_state_group_count=groups,
            direct_state_group_offset=1,
        )

    assert result is not None
    output, _ = result
    assert output.shape == (1, 2, 4, 8)
    torch.testing.assert_close(output.squeeze(0), expected_output)
    torch.testing.assert_close(pool, expected_pool)
    assert not torch.equal(pool[max_requests + 1:max_requests + 3], original[max_requests + 1:max_requests + 3])
    torch.testing.assert_close(pool[:max_requests + 1], original[:max_requests + 1])


def test_qwen38_static_direct_recipe_matches_indexed_reference():
    generator = torch.Generator().manual_seed(53)
    batch, q_heads, value_heads, dim = 1, 16, 48, 128
    packed = torch.randn(batch, (2 * q_heads + value_heads) * dim, dtype=torch.bfloat16, generator=generator)
    log_decay = -torch.rand(batch, value_heads, dtype=torch.float32, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, value_heads, dtype=torch.float32, generator=generator)).to(torch.bfloat16)
    indexed_state = torch.randn(batch, value_heads, dim, dim, dtype=torch.float32, generator=generator)
    direct_state = indexed_state.clone()
    indices = torch.arange(batch, dtype=torch.int32)

    expected_output, _ = packed_recurrent_decode(
        packed,
        log_decay,
        beta,
        indexed_state,
        indices,
        indices,
        None,
        True,
    )
    output, _ = packed_recurrent_decode(
        packed,
        log_decay,
        beta,
        direct_state,
        None,
        None,
        None,
        True,
        True,
    )

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected_output, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(direct_state, indexed_state, atol=2e-5, rtol=2e-4)


def test_qwen38_fused_direct_step_matches_existing_decode_chain():
    generator = torch.Generator().manual_seed(127)
    batch, q_heads, value_heads, dim = 1, 16, 48, 128
    packed_width = (2 * q_heads + value_heads) * dim
    packed = torch.randn(batch, packed_width, dtype=torch.bfloat16, generator=generator) * 0.1
    a = torch.randn(batch, value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    b = torch.randn(batch, value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    A_log = torch.randn(value_heads, dtype=torch.float32, generator=generator) * 0.1 - 2.0
    dt_bias = torch.randn(value_heads, dtype=torch.bfloat16, generator=generator) * 0.1
    conv_weight = torch.randn(packed_width, 4, dtype=torch.bfloat16, generator=generator) * 0.05
    expected_conv_state = torch.randn(batch, 3, packed_width, dtype=torch.bfloat16, generator=generator) * 0.1
    actual_conv_state = expected_conv_state.clone()
    expected_ssm_state = torch.randn(batch, value_heads, dim, dim, generator=generator) * 0.01
    actual_ssm_state = expected_ssm_state.clone()
    scale = dim**-0.5

    log_decay, beta = hpu_fused_gdn_gating(A_log, a, b, dt_bias)
    convolved = hpu_causal_conv1d_update(
        x=packed,
        conv_state=expected_conv_state,
        weight=conv_weight,
        bias=None,
        activation="silu",
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32),
        direct_state_layout=True,
    )
    expected_output, _ = packed_recurrent_decode(
        convolved,
        log_decay,
        beta,
        expected_ssm_state,
        None,
        None,
        scale,
        True,
        True,
    )

    actual_output, returned_conv_state, returned_ssm_state = qwen38_fused_decode_step_direct(
        packed,
        a,
        b,
        A_log,
        dt_bias,
        actual_conv_state,
        conv_weight,
        None,
        actual_ssm_state,
        scale,
    )

    assert returned_conv_state is actual_conv_state
    assert returned_ssm_state is actual_ssm_state
    torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
    torch.testing.assert_close(actual_conv_state, expected_conv_state, atol=0, rtol=0)
    torch.testing.assert_close(actual_ssm_state, expected_ssm_state, atol=0, rtol=0)


def test_vllm_adapter_leaves_mtp_batches_on_general_path():
    _, _, _, pool, log_decay, beta = _inputs(batch=2)
    packed = torch.randn(4, 64)
    indices = torch.tensor([1, 2], dtype=torch.int32)
    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN": "1"}):
        result = maybe_run_gdn_decode_packed(
            mixed_qkv=packed,
            log_decay=log_decay,
            beta=beta,
            state_pool=pool,
            load_state_indices=indices,
            store_state_indices=indices,
            use_qk_l2norm=True,
        )
    assert result is None


@pytest.mark.parametrize("batch", [1, 16, 32])
def test_vllm_adapter_routes_qualified_fused_direct_step_without_conv_bias(batch: int):
    packed = torch.empty(batch, 10240, dtype=torch.bfloat16)
    a = torch.empty(batch, 48, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    A_log = torch.zeros(48, dtype=torch.float32)
    dt_bias = torch.zeros(48, dtype=torch.bfloat16)
    conv_state = torch.empty(batch, 3, 10240, dtype=torch.bfloat16)
    conv_weight = torch.zeros(10240, 4, dtype=torch.bfloat16)
    ssm_state = torch.empty(batch + 2, 48, 128, 128, dtype=torch.float32)
    load_indices = torch.arange(1, batch + 1, dtype=torch.int32)
    expected_output = torch.empty(batch, 48, 128, dtype=torch.bfloat16)
    expected = (expected_output, conv_state, ssm_state.narrow(0, 1, batch))

    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN": "1"}), \
            mock.patch(
                "vllm_gaudi.ops.flashinfer_gaudi_adapter.qwen38_fused_decode_step_direct",
                return_value=expected,
            ) as run:
        result = maybe_run_gdn_fused_decode_step(
            packed,
            a,
            b,
            A_log,
            dt_bias,
            conv_state,
            conv_weight,
            None,
            ssm_state,
            load_indices,
            direct_conv_state=True,
            direct_gdn_state=True,
            direct_state_group_count=1,
            direct_state_group_offset=0,
            scale=128**-0.5,
        )

    assert result is not None
    assert result[0].shape == (1, batch, 48, 128)
    assert result[1].untyped_storage().data_ptr() == ssm_state.untyped_storage().data_ptr()
    assert run.call_count == 1
    assert run.call_args.args[7] is None
    assert run.call_args.args[8].untyped_storage().data_ptr() == ssm_state.untyped_storage().data_ptr()


@pytest.mark.parametrize("batch", [12, 20])
def test_vllm_adapter_keeps_unqualified_exponential_buckets_on_fallback(batch: int):
    load_indices = torch.arange(batch, dtype=torch.int32)
    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN": "1"}), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter.qwen38_fused_decode_step_direct") as run:
        result = maybe_run_gdn_fused_decode_step(
            torch.empty(batch, 1),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            None,
            torch.empty(0),
            load_indices,
            direct_conv_state=True,
            direct_gdn_state=True,
            direct_state_group_count=1,
            direct_state_group_offset=0,
            scale=1.0,
        )

    assert result is None
    run.assert_not_called()


def test_vllm_adapter_auto_skips_indexed_reference():
    q, k, v, pool, log_decay, beta = _inputs(batch=2)
    original = pool.clone()
    packed = torch.cat((q.reshape(2, -1), k.reshape(2, -1), v.reshape(2, -1)), dim=-1)
    indices = torch.tensor([1, 2], dtype=torch.int32)

    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN": "1"}), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter._BACKEND_POLICY", "auto"), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter._PUBLIC_AUTO_PROMOTED", False), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter._BRIDGE_AUTO_ENABLED", False):
        result = maybe_run_gdn_decode_packed(
            mixed_qkv=packed,
            log_decay=log_decay,
            beta=beta,
            state_pool=pool,
            load_state_indices=indices,
            store_state_indices=indices,
            use_qk_l2norm=True,
        )

    assert result is None
    torch.testing.assert_close(pool, original)


def test_vllm_adapter_allows_indexed_reference_for_dflash_transition():
    q, k, v, pool, log_decay, beta = _inputs(batch=1)
    packed = torch.cat((q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)), dim=-1)
    load = torch.tensor([2], dtype=torch.int32)
    store = torch.tensor([5], dtype=torch.int32)
    expected_pool = pool.clone()
    expected, _ = gated_delta_rule_decode_packed(
        packed,
        log_decay,
        beta,
        expected_pool,
        load,
        store,
    )

    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN": "1"}), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter._BACKEND_POLICY", "auto"), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter._PUBLIC_AUTO_PROMOTED", False), \
            mock.patch("vllm_gaudi.ops.flashinfer_gaudi_adapter._BRIDGE_AUTO_ENABLED", False):
        result = maybe_run_gdn_decode_packed(
            mixed_qkv=packed,
            log_decay=log_decay,
            beta=beta,
            state_pool=pool,
            load_state_indices=load,
            store_state_indices=store,
            use_qk_l2norm=True,
            allow_indexed_reference=True,
        )

    assert result is not None
    torch.testing.assert_close(result[0], expected.unsqueeze(0))
    torch.testing.assert_close(pool, expected_pool)


def test_vllm_prefill_adapter_selects_promoted_qwen38_tactic():
    tokens = 64
    q = torch.zeros(1, tokens, 16, 128, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros(1, tokens, 48, 128, dtype=torch.bfloat16)
    log_decay = torch.zeros(1, tokens, 48, dtype=torch.float32)
    beta = torch.ones(1, tokens, 48, dtype=torch.bfloat16)
    initial_state = torch.zeros(1, 48, 128, 128, dtype=torch.float32)
    expected = (torch.empty_like(v), initial_state.clone())

    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN_PREFILL": "1"}), \
            mock.patch(
                "vllm_gaudi.ops.flashinfer_gaudi_adapter._chunk_gated_delta_rule_log_gate",
                return_value=expected,
            ) as run:
        result = maybe_run_gdn_prefill(
            q,
            k,
            v,
            log_decay,
            beta,
            initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            chunk_size=128,
            prefill_num_seqs=1,
            prefill_seq_len=tokens,
        )

    assert result is expected
    kwargs = run.call_args.kwargs
    assert kwargs["flashqla_reformulation"] is True
    assert kwargs["deferred_output_add"] is True
    assert kwargs["fused_state_matmul"] is True
    assert kwargs["recursive_solver_base"] == 16
    assert kwargs["compact_repeated_kkt"] is True
    assert kwargs["solve_in_fp32"] is True
    assert kwargs["state_in_fp32"] is True
    assert kwargs["compute_dtype"] == torch.float32
    assert kwargs["preserve_compact_qk"] is True
    assert kwargs["masked_triangular_decay"] is True


def test_vllm_prefill_adapter_falls_back_for_unpromoted_shape():
    tokens = 64
    q = torch.zeros(1, tokens, 8, 128, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros(1, tokens, 24, 128, dtype=torch.bfloat16)
    log_decay = torch.zeros(1, tokens, 24, dtype=torch.float32)
    beta = torch.ones(1, tokens, 24, dtype=torch.bfloat16)

    with mock.patch.dict(os.environ, {"VLLM_HPU_FLASHINFER_GDN_PREFILL": "1"}):
        result = maybe_run_gdn_prefill(
            q,
            k,
            v,
            log_decay,
            beta,
            None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            chunk_size=64,
            prefill_num_seqs=1,
            prefill_seq_len=tokens,
        )

    assert result is None
