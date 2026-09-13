# SPDX-License-Identifier: Apache-2.0
"""Contracts for views bound before a compiled recurrent decoder group."""

from types import SimpleNamespace
from contextlib import nullcontext

import pytest
import torch

from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention
from vllm_gaudi.ops.flashinfer_gaudi_adapter import maybe_run_gdn_fused_decode_step


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN", "1")
    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN_TP2", "1")
    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN_FUSED_DECODE", "1")


def _layer(pool, offset):
    return SimpleNamespace(kv_cache=(torch.empty(0), pool),
                           compact_state_group_count=3,
                           compact_state_group_offset=offset,
                           cache_config=SimpleNamespace(enable_prefix_caching=False),
                           qkv_size=5120,
                           A_log=torch.zeros(24))


def _metadata(batch, **kwargs):
    return SimpleNamespace(is_prompt=False,
                           direct_gdn_state=True,
                           load_indices_tensor=torch.ones(3, batch, dtype=torch.int32),
                           **kwargs)


def test_binding_preserves_pool_aliases_across_batches_and_slot_reuse():
    pool = torch.arange(98 * 4, dtype=torch.float32).view(98, 4).clone()
    expected = pool.clone()
    layers = [_layer(pool, offset) for offset in range(3)]
    for step, batch in enumerate((1, 8, 2, 32, 1)):
        for offset, layer in enumerate(layers):
            HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(batch), batch)
            view = layer._hpu_active_ssm_state
            assert view is not None and view.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
            HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, _metadata(batch), batch)
            assert layer._hpu_active_ssm_state is view
            view.add_(step + offset + 1)
            expected[1 + offset * 32:1 + offset * 32 + batch].add_(step + offset + 1)
        torch.testing.assert_close(pool, expected, rtol=0, atol=0)
        # Scheduler reset of a retired request must be visible through cached views.
        pool[1].zero_()
        expected[1].zero_()
    replacement = torch.zeros_like(pool)
    layers[0].kv_cache = (torch.empty(0), replacement)
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layers[0], _metadata(1), 1)
    layers[0]._hpu_active_ssm_state.fill_(7)
    assert replacement[1].eq(7).all()
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["prefill", "indexed", "mtp", "prefix", "missing", "bad_pool", "bad_group"])
def test_binding_clears_stale_view_for_unsupported_paths(case):
    layer = _layer(torch.zeros(98, 4), 1)
    metadata = _metadata(1)
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, metadata, 1)
    assert layer._hpu_active_ssm_state is not None
    tokens = 1
    if case == "prefill":
        metadata.is_prompt = True
    elif case == "indexed":
        metadata.direct_gdn_state = False
    elif case == "mtp":
        tokens = 2
    elif case == "prefix":
        layer.cache_config.enable_prefix_caching = True
    elif case == "missing":
        metadata = None
    elif case == "bad_pool":
        layer.kv_cache = (torch.empty(0), torch.zeros(97, 4))
    elif case == "bad_group":
        layer.compact_state_group_offset = 3
    HPUGatedDeltaNetAttention.prepare_decode_state_view(layer, metadata, tokens)
    assert layer._hpu_active_ssm_state is None


def test_active_adapter_updates_exact_rows_over_changing_decode_steps():
    rng = torch.Generator().manual_seed(983)

    def rand(*shape, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=rng) * 0.05).to(dtype)

    # Three non-overlapping views of the same pool, matching group sharing.
    expected = rand(8, 24, 128, 128, dtype=torch.float32)
    actual = expected.clone()
    untouched = actual.clone()
    conv0 = [rand(1, 3, 5120) for _ in range(3)]
    conv1 = [state.clone() for state in conv0]
    weight, a_log, dt_bias = rand(5120, 4), rand(24, dtype=torch.float32) - 2, rand(24)
    for step in range(4):
        packed, a, b = rand(1, 5120), rand(1, 24), rand(1, 24)
        for offset in range(3):
            indices = torch.tensor([offset * 2 + 1], dtype=torch.int32)
            common = dict(mixed_qkv=packed,
                          a=a,
                          b=b,
                          A_log=a_log,
                          dt_bias=dt_bias,
                          conv_weight=weight,
                          conv_bias=None,
                          load_state_indices=indices,
                          direct_conv_state=True,
                          direct_gdn_state=True,
                          direct_state_group_count=3,
                          direct_state_group_offset=offset,
                          scale=128**-0.5)
            reference = maybe_run_gdn_fused_decode_step(**common, conv_state=conv0[offset], ssm_state=expected)
            candidate = maybe_run_gdn_fused_decode_step(**common,
                                                        conv_state=conv1[offset],
                                                        ssm_state=actual.narrow(0, offset * 2 + 1, 1),
                                                        state_is_active_view=True)
            assert reference is not None and candidate is not None
            torch.testing.assert_close(candidate[0], reference[0], rtol=0, atol=0)
            torch.testing.assert_close(conv0[offset], conv1[offset], rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[[0, 2, 4, 6, 7]], untouched[[0, 2, 4, 6, 7]], rtol=0, atol=0)


def test_model_adapter_binds_before_model_call_and_clears_on_prefill(monkeypatch):
    import vllm_gaudi.v1.worker.hpu_model_runner as runner

    monkeypatch.setattr(runner, "set_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(runner, "set_hpu_dp_metadata", lambda *args: nullcontext())
    pool = torch.zeros(98, 4)
    layer = _layer(pool, 1)
    layer.prepare_decode_state_view = lambda metadata, tokens: HPUGatedDeltaNetAttention.prepare_decode_state_view(
        layer, metadata, tokens)

    class Model(torch.nn.Module):

        def forward(self, input_ids):
            if input_ids.shape[1] == 1:
                assert layer._hpu_active_ssm_state is not None
                layer._hpu_active_ssm_state.add_(1)
            else:
                assert layer._hpu_active_ssm_state is None
            return input_ids

    adapter = object.__new__(runner.HpuModelAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.model = Model()
    adapter.pooling_model = False
    adapter.flatten_input = False
    adapter._rotary_prepare_cos_sin = None
    adapter._gdn_state_view_layers = (layer, )
    adapter._gdn_state_views_logged = False
    adapter._gdn_state_view_key = None
    adapter.metadata_processor = SimpleNamespace(process_metadata=lambda metadata, *args: metadata)
    adapter.dtype = torch.bfloat16
    adapter.vllm_config = None
    adapter.dummy_num_input_tokens = 0
    adapter.dummy_num_tokens_across_dp_cpu = None
    for batch, length in ((1, 1), (1, 8), (2, 1), (1, 1)):
        metadata = _metadata(batch)
        metadata.is_prompt = length > 1
        metadata.direct_gdn_state = length == 1
        adapter(input_ids=torch.zeros(batch, length, dtype=torch.int64), attn_metadata=metadata)
    assert pool[33].eq(3).all() and pool[34].eq(1).all()
    assert pool[:33].count_nonzero() == 0 and pool[35:].count_nonzero() == 0
    original = layer.prepare_decode_state_view
    layer.prepare_decode_state_view = lambda *args: pytest.fail("Repeated binding during steady decode")
    adapter(input_ids=torch.zeros(1, 1, dtype=torch.int64), attn_metadata=_metadata(1))
    assert pool[33].eq(4).all()
    layer.prepare_decode_state_view = original
    replacement = torch.zeros_like(pool)
    layer.kv_cache = (torch.empty(0), replacement)
    adapter.invalidate_gdn_state_views()
    assert layer._hpu_cached_ssm_view is None and layer._hpu_active_ssm_source is None
    adapter(input_ids=torch.zeros(1, 1, dtype=torch.int64), attn_metadata=_metadata(1))
    assert replacement[33].eq(1).all() and pool[33].eq(4).all()


def test_tp2_state_binding_requires_explicit_opt_in(monkeypatch):
    from vllm_gaudi.ops.flashinfer_gaudi_adapter import can_bind_gdn_active_state

    monkeypatch.delenv("VLLM_HPU_FLASHINFER_GDN_TP2", raising=False)
    assert can_bind_gdn_active_state(1, 10240, 48)
    assert not can_bind_gdn_active_state(1, 5120, 24)
    monkeypatch.setenv("VLLM_HPU_FLASHINFER_GDN_TP2", "1")
    assert can_bind_gdn_active_state(1, 5120, 24)
