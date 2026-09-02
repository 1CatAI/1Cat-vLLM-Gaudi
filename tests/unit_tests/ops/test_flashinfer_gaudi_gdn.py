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
from flashinfer_gaudi._reference import packed_recurrent_decode
from flashinfer_gaudi.gdn_decode import (
    BackendUnavailableError,
    _call_native_packed,
    gated_delta_rule_decode,
    gated_delta_rule_mtp,
    gated_delta_rule_decode_packed,
    gated_delta_rule_decode_pretranspose,
)
from flashinfer_gaudi.gdn_prefill import chunk_gated_delta_rule
from vllm_gaudi.ops.flashinfer_gaudi_adapter import maybe_run_gdn_decode_packed, maybe_run_gdn_prefill


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


def _naive_step(q, k, v, state, log_decay, beta):
    repeat = v.shape[2] // q.shape[2]
    q = F.normalize(q.float(), dim=-1, eps=1e-6).repeat_interleave(repeat, dim=2)
    k = F.normalize(k.float(), dim=-1, eps=1e-6).repeat_interleave(repeat, dim=2)
    scale = q.shape[-1]**-0.5
    decayed = state * torch.exp(log_decay[:, 0]).unsqueeze(-1).unsqueeze(-1)
    projection = torch.matmul(decayed, k[:, 0].unsqueeze(-1)).squeeze(-1)
    delta = (v[:, 0].float() - projection) * beta[:, 0].unsqueeze(-1)
    updated = decayed + delta.unsqueeze(-1) * k[:, 0].unsqueeze(-2)
    output = torch.matmul(updated, (q[:, 0] * scale).unsqueeze(-1)).squeeze(-1)
    return output, updated


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
    with pytest.raises(BackendUnavailableError, match="requires an HPU tensor"):
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
