# SPDX-License-Identifier: Apache-2.0
"""Preserve the canonical checkpoint across speculative-to-T1 transitions."""
import os
from contextlib import ExitStack
from types import MethodType, SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_gaudi.models import qwen3_5


def _extract(monkeypatch, loads, stores, accepted, *, prefix_caching=False):
    metadata = SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=loads.unsqueeze(0),
        store_indices_tensor=stores.unsqueeze(0) if stores is not None else None,
        query_start_loc_p=torch.arange(loads.shape[0] + 1, dtype=torch.int32),
        num_accepted_tokens=accepted,
    )
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(qwen3_5, "get_forward_context", lambda: context)
    attention = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        cache_group_idx=torch.tensor(0),
        kv_cache=(torch.empty(1), torch.empty(1)),
    )
    attention._resolve_state_indices = MethodType(qwen3_5.HPUGatedDeltaNetAttention._resolve_state_indices, attention)
    result = qwen3_5.HPUGatedDeltaNetAttention._extract_metadata(attention, loads.numel())
    return result[3], result[4]


@pytest.mark.parametrize("accepted_count", range(1, 9))
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_transition_reads_accepted_checkpoint_but_writes_canonical_slot(monkeypatch, accepted_count, batch,
                                                                        prefix_caching):
    canonical = 1 + torch.arange(batch, dtype=torch.int32) * 8
    accepted = torch.full((batch, ), accepted_count, dtype=torch.int32)
    loads = canonical + accepted - 1
    actual_load, actual_store = _extract(monkeypatch, loads, canonical, accepted, prefix_caching=prefix_caching)
    assert torch.equal(actual_load, loads)
    assert torch.equal(actual_store, canonical)

    # A subsequent speculative round is told to start from checkpoint zero.
    # Ensure that is where the T1 step actually commits its updated state.
    pool = torch.arange(batch * 8 + 2, dtype=torch.float32)
    updated = pool.index_select(0, actual_load.long()) + 100
    pool.index_copy_(0, actual_store.long(), updated)
    assert torch.equal(pool.index_select(0, canonical.long()), updated)


def test_ordinary_decode_keeps_identical_load_store_objects(monkeypatch):
    indices = torch.tensor([1, 2], dtype=torch.int32)
    loads, stores = _extract(monkeypatch, indices, indices.clone(), None)
    assert loads is stores


def test_missing_store_still_defaults_to_load_for_transition(monkeypatch):
    indices = torch.tensor([4], dtype=torch.int32)
    loads, stores = _extract(monkeypatch, indices, None, torch.tensor([4], dtype=torch.int32))
    assert loads is stores


def test_transition_preserves_padding_destination(monkeypatch):
    load = torch.tensor([4, -1], dtype=torch.int32)
    store = torch.tensor([1, -1], dtype=torch.int32)
    actual_load, actual_store = _extract(monkeypatch, load, store, torch.tensor([4, 1], dtype=torch.int32))
    assert torch.equal(actual_load, load)
    assert torch.equal(actual_store, store)


@pytest.mark.parametrize("accepted_count", [2, 4, 8])
@pytest.mark.parametrize("execution", ["cpu", "hpu", "hpu-compiled"])
def test_transition_forward_commits_recurrent_and_convolution_state(monkeypatch, accepted_count, execution):
    if execution != "cpu" and os.environ.get("FLASHINFER_GAUDI_TEST_HPU") != "1":
        pytest.skip("Set FLASHINFER_GAUDI_TEST_HPU=1 only when a test card is available")
    from flashinfer_gaudi.gdn_decode import gated_delta_rule_decode_packed
    from vllm_gaudi.ops import flashinfer_gaudi_adapter
    from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update

    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN", "1")
    monkeypatch.setattr(flashinfer_gaudi_adapter, "_BACKEND_POLICY", "pytorch")
    device = "cpu" if execution == "cpu" else "hpu"
    generator = torch.Generator().manual_seed(81)
    hidden = torch.randn(1, 8, generator=generator).to(device)
    conv_pool = torch.randn(20, 10, 16, generator=generator).to(device)
    state_pool = torch.randn(20, 2, 4, 4, generator=generator).to(device)
    weight = torch.randn(16, 1, 4, generator=generator).to(device)
    canonical = torch.tensor([1], dtype=torch.int32, device=device)
    loads = torch.tensor([accepted_count], dtype=torch.int32, device=device)
    accepted = torch.tensor([accepted_count], dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        is_prompt=False,
        load_indices_tensor=loads.unsqueeze(0),
        store_indices_tensor=canonical.unsqueeze(0),
        query_start_loc_p=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_accepted_tokens=accepted,
    )
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(qwen3_5, "get_forward_context", lambda: context)
    attention = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        cache_group_idx=torch.tensor(0, device=device),
        kv_cache=(conv_pool, state_pool),
        compact_state_group_offset=None,
        compact_state_group_count=None,
        gqa_interleaved_layout=False,
        tp_size=1,
        num_v_heads=2,
        head_k_dim=4,
        head_v_dim=4,
        qkv_size=16,
        z_size=8,
        in_proj_qkvz=lambda x: (torch.cat((x, x, x), dim=-1), None),
        in_proj_ba=lambda x: (x.new_zeros(x.shape[0], 4), None),
        A_log=torch.zeros(2, device=device),
        dt_bias=torch.zeros(2, device=device),
        conv1d=SimpleNamespace(weight=weight, bias=None),
        activation=None,
        dflash_full_query_conv=False,
        dflash_conv_round_before_activation=False,
        norm=lambda x, z: x,
        out_proj=lambda x: (x, None),
    )
    for name in ("_extract_metadata", "_resolve_state_indices"):
        setattr(attention, name, MethodType(getattr(qwen3_5.HPUGatedDeltaNetAttention, name), attention))

    expected_conv = conv_pool.clone()
    expected_state = state_pool.clone()
    initial_conv = conv_pool.cpu().clone()
    initial_state = state_pool.cpu().clone()
    packed = torch.cat((hidden, hidden), dim=-1)
    convolved = hpu_causal_conv1d_update(
        packed,
        expected_conv,
        weight.view(16, 4),
        activation=None,
        conv_state_indices=canonical,
        num_accepted_tokens=accepted,
        query_start_loc=metadata.query_start_loc_p,
        max_query_len=1,
    )
    g, beta = qwen3_5.hpu_fused_gdn_gating(hidden.new_zeros(2), hidden.new_zeros(1, 2), hidden.new_zeros(1, 2),
                                           hidden.new_zeros(2))
    expected_output, _ = gated_delta_rule_decode_packed(convolved, g, beta, expected_state, loads, canonical)

    def candidate(x):
        # Materialize the tiny stub projection's output view. The HPU strict
        # fallback checker does not annotate a returned as_strided alias.
        return qwen3_5.HPUGatedDeltaNetAttention.forward(attention, x).clone()

    with ExitStack() as stack:
        if execution == "hpu-compiled":
            from habana_frameworks.torch.dynamo.compile_backend import config as hpu_config
            stack.enter_context(mock.patch.object(hpu_config, "use_eager_fallback", False))
            candidate = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
        actual_output = candidate(hidden)
        if device == "hpu":
            torch.hpu.synchronize()

    # Compilation may fuse arithmetic, but it must never change the destination.
    rtol, atol = (2e-5, 2e-6) if execution == "hpu-compiled" else (0, 0)
    torch.testing.assert_close(actual_output.cpu(), expected_output.reshape_as(hidden).cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(conv_pool.cpu(), expected_conv.cpu(), rtol=rtol, atol=atol)
    torch.testing.assert_close(state_pool.cpu(), expected_state.cpu(), rtol=rtol, atol=atol)
    untouched = torch.arange(conv_pool.shape[0]) != 1
    torch.testing.assert_close(conv_pool.cpu()[untouched], initial_conv[untouched], rtol=0, atol=0)
    torch.testing.assert_close(state_pool.cpu()[untouched], initial_state[untouched], rtol=0, atol=0)
