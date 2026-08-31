# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for the FlashInfer-compatible Gaudi GDN API."""

from __future__ import annotations

import inspect
import os
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from flashinfer_gaudi import clear_backend_policy_override, set_backend_policy
from flashinfer_gaudi.gdn_decode import (
    BackendUnavailableError,
    gated_delta_rule_decode_packed,
    gated_delta_rule_decode_pretranspose,
)


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
