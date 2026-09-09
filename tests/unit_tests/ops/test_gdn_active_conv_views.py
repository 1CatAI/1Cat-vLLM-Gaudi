# SPDX-License-Identifier: Apache-2.0
"""Shared convolution histories retain ownership across native decoder plans."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update
from vllm_gaudi.ops.tp2_prepared_plan import _signature_key


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    for flag in ("VLLM_HPU_FLASHINFER_GDN", "VLLM_HPU_FLASHINFER_GDN_TP2", "VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE",
                 "VLLM_HPU_GDN_ACTIVE_CONV_STATE_VIEWS"):
        monkeypatch.setenv(flag, "1")


def _layer(pool, offset=0):
    return SimpleNamespace(kv_cache=(pool, torch.zeros(98, 4)),
                           tp_size=2,
                           conv_kernel_size=4,
                           compact_state_group_count=3,
                           compact_state_group_offset=offset,
                           cache_config=SimpleNamespace(enable_prefix_caching=False),
                           qkv_size=5120,
                           A_log=torch.zeros(24))


def _metadata(**kwargs):
    result = dict(is_prompt=False,
                  direct_gdn_state=True,
                  load_indices_tensor=torch.ones(3, 1, dtype=torch.int32),
                  query_start_loc_p=torch.tensor([0, 1], dtype=torch.int32))
    result.update(kwargs)
    return SimpleNamespace(**result)


@pytest.mark.parametrize("width", [3, 4])
def test_shared_pool_recurrence_and_reserve_slots_remain_exact(width):
    rng = torch.Generator().manual_seed(843)
    original = (torch.randn(98, width, 5120, generator=rng) * .02).bfloat16()
    actual = original.clone()
    expected = original.clone()
    weights = (torch.randn(5120, 4, generator=rng) * .02).bfloat16()
    layers = [_layer(actual, offset) for offset in range(3)]
    for step in range(8):
        source = (torch.randn(1, 5120, generator=rng) * .02).bfloat16()
        for offset, layer in enumerate(layers):
            HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(), 1)
            view = layer._hpu_active_conv_state
            assert view is not None and view.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr()
            HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(), 1)
            assert view is layer._hpu_active_conv_state
            reference = expected[1 + 32 * offset:2 + 32 * offset, -3:]
            kwargs = dict(direct_state_layout=True, activation="silu", query_start_loc=torch.tensor([0, 1]))
            a = hpu_causal_conv1d_update(source, reference, weights, **kwargs)
            b = hpu_causal_conv1d_update(source, view, weights, **kwargs)
            assert torch.equal(a, b)
        assert torch.equal(actual, expected)
        if step == 3:
            actual[33].zero_()
            expected[33].zero_()
    inactive = torch.ones(98, dtype=torch.bool)
    inactive[[1, 33, 65]] = False
    assert torch.equal(actual[inactive], original[inactive])
    if width == 4:
        assert torch.equal(actual[[1, 65], 0], original[[1, 65], 0])


@pytest.mark.parametrize("mode", ["prefill", "indexed", "accepted", "full_query", "tp1", "batch", "off"])
def test_unsupported_modes_clear_previous_alias(monkeypatch, mode):
    layer = _layer(torch.zeros(98, 3, 5120, dtype=torch.bfloat16))
    metadata = _metadata()
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, metadata, 1)
    assert layer._hpu_active_conv_state is not None
    tokens = 1
    if mode == "prefill":
        metadata.is_prompt = True
    elif mode == "indexed":
        metadata.direct_gdn_state = False
    elif mode == "accepted":
        metadata.num_accepted_tokens = torch.tensor([1])
    elif mode == "full_query":
        metadata.dflash_full_query = True
    elif mode == "tp1":
        layer.tp_size = 1
    elif mode == "batch":
        tokens = 2
        metadata.load_indices_tensor = torch.ones(3, 2, dtype=torch.int32)
    else:
        monkeypatch.setenv("VLLM_HPU_GDN_ACTIVE_CONV_STATE_VIEWS", "0")
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, metadata, tokens)
    assert layer._hpu_active_conv_state is None


def test_new_pool_changes_plan_signature_and_invalidation_releases_alias(monkeypatch):
    from vllm_gaudi.v1.worker.hpu_model_runner import HpuModelAdapter

    monkeypatch.setenv("VLLM_HPU_TP2_STATIC_GROUP_PLAN", "0")
    layer = _layer(torch.zeros(98, 3, 5120, dtype=torch.bfloat16), 1)
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(), 1)
    old = layer._hpu_active_conv_state
    assert _signature_key(old) == _signature_key(layer.kv_cache[0][33:34])
    layer.kv_cache = (torch.ones_like(layer.kv_cache[0]), layer.kv_cache[1])
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(), 1)
    new = layer._hpu_active_conv_state
    assert _signature_key(old) != _signature_key(new)
    new.zero_()
    assert layer.kv_cache[0][33].count_nonzero() == 0
    adapter = SimpleNamespace(_gdn_state_view_layers=(layer, ), gdn_dma_pipeline=lambda: None)
    HpuModelAdapter.invalidate_gdn_state_views(adapter)
    assert layer._hpu_active_conv_state is None and layer._hpu_cached_conv_view is None
    assert layer._hpu_active_conv_source is None and layer._hpu_active_conv_key is None


def test_metadata_delivers_active_history_only_in_ordinary_decode(monkeypatch):
    from vllm_gaudi.models import qwen3_5

    layer = _layer(torch.zeros(98, 3, 5120, dtype=torch.bfloat16), 1)
    layer.cache_group_idx = torch.tensor(1)
    layer._resolve_state_indices = MethodType(HPUGatedDeltaNetAttention._resolve_state_indices, layer)
    metadata = _metadata()
    monkeypatch.setattr(qwen3_5, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, metadata, 1)
    assert HPUGatedDeltaNetAttention._extract_metadata(layer, 1)[1] is layer._hpu_active_conv_state
    metadata.num_accepted_tokens = torch.tensor([1])
    assert HPUGatedDeltaNetAttention._extract_metadata(layer, 1)[1] is layer.kv_cache[0]


@pytest.mark.parametrize("shape,dtype", [((98, 2, 5120), torch.bfloat16), ((97, 3, 5120), torch.bfloat16),
                                         ((98, 3, 5120), torch.float32)])
def test_invalid_layout_rejected_before_state_changes(shape, dtype):
    pool = torch.ones(shape, dtype=dtype)
    layer = _layer(pool)
    with pytest.raises(RuntimeError, match="compact state pool"):
        HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(), 1)
    assert pool.eq(1).all()
