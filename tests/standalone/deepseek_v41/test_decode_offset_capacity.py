# SPDX-License-Identifier: Apache-2.0
"""Keep native decode offsets bounded without changing selected-KV addressing."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_paged_attention import (NATIVE_WORK_TOKENS, PagedCSA2Attention,
                                                         _selected_attention_layout)


@pytest.mark.parametrize('tp', [2, 4])
@pytest.mark.parametrize('layer', [0, 1])
def test_production_context_does_not_expand_decode_offset_table(monkeypatch, tp, layer):
    monkeypatch.setenv('VLLM_HPU_DSV41_OUTPUT_GEMM_LAYOUT', '0')
    monkeypatch.setenv('VLLM_HPU_DSV41_Q_SCALE_ROPE', '0')
    config = dict(num_hidden_layers=2,
                  compress_ratios=[0, 2],
                  num_attention_heads=64,
                  o_groups=8,
                  index_n_heads=32,
                  rms_norm_eps=1e-6,
                  sliding_window=128,
                  kv_source_layer_ids=[1],
                  index_source_layer_ids=[1],
                  candidate_source_layer_id=1)
    shared = SimpleNamespace(length=1048576,
                             runtime_indexer=False,
                             decoded_kv_state=False,
                             layer_start=0,
                             sources={'1': SimpleNamespace()},
                             topk={'1': SimpleNamespace()},
                             rotary_bucket=lambda name, length: torch.empty(length, 64))
    attention = PagedCSA2Attention(torch.nn.Module(), config, layer, shared, None, None, None, 'cpu', tp)
    assert attention.length == 1048576
    assert attention.selected_offsets.numel() == NATIVE_WORK_TOKENS * 512 == 3072
    assert attention.selected_offsets.untyped_storage().nbytes() == 12288
    assert attention.selected_offsets[-1].item() == 3071


@pytest.mark.parametrize('tokens', [1, 2, 6])
def test_last_decode_row_and_invalid_slots_preserve_layout(tokens):
    physical = torch.arange(tokens * 512, dtype=torch.int32).reshape(tokens, 512) + 1000
    selected = physical.clone()
    selected[:, 1] = -1
    selected[:, 2] = selected[:, 0]
    physical[:, 2] = physical[:, 0]
    window = torch.arange(-3, 125, dtype=torch.int32).expand(tokens, -1).clone()
    window[window < 0] = -1
    swa = torch.arange(256, dtype=torch.int32)
    short = torch.arange(NATIVE_WORK_TOKENS * 512, dtype=torch.int32)
    # The preserved parent allocated the full C8192 table for this C1-C6 use.
    legacy = torch.arange(8192 * 512, dtype=torch.int32)
    rows, indices = _selected_attention_layout(physical, selected, window, swa, short)
    old_rows, old_indices = _selected_attention_layout(physical, selected, window, swa, legacy)
    assert torch.equal(rows, old_rows) and torch.equal(indices, old_indices)
    assert rows.shape == (1, 256 + tokens * 512)
    assert indices[-1, -1].item() == 256 + tokens * 512 - 1
    assert (indices[:, :3] == -1).all() and (indices[:, 129] == -1).all()



@pytest.mark.parametrize("tokens", [1, 2, 6])
@pytest.mark.parametrize("ratio", [1, 2])
def test_logical_mla_uses_normal_decode_contract(tokens, ratio):
    owner = SimpleNamespace(paged_mla_logical=True, ratio=ratio, mla_mme=True, direct_selected_kv=True,
                             window=128, shared_prefix_kv=True, search_length=32768, index_ratio=ratio)
    value = SimpleNamespace(shape=(tokens, 5120), device=SimpleNamespace(type="hpu"))
    assert PagedCSA2Attention._uses_logical_mla(owner, value, False)
    assert not PagedCSA2Attention._uses_logical_mla(owner, value, True)
    owner.search_length = 512 * ratio
    assert not PagedCSA2Attention._uses_logical_mla(owner, value, False)


@pytest.mark.parametrize("change", ["capability", "ratio", "mme", "direct", "window", "device", "batch"])
def test_logical_mla_keeps_other_contracts_on_existing_path(change):
    owner = SimpleNamespace(paged_mla_logical=True, ratio=1, mla_mme=True, direct_selected_kv=True,
                             window=128, shared_prefix_kv=True, search_length=32768, index_ratio=1)
    value = SimpleNamespace(shape=(1, 5120), device=SimpleNamespace(type="hpu"))
    if change == "capability":
        owner.paged_mla_logical = False
    elif change == "ratio":
        owner.ratio = 0
    elif change == "mme":
        owner.mla_mme = False
    elif change == "direct":
        owner.direct_selected_kv = False
    elif change == "window":
        owner.window = 256
    elif change == "device":
        value.device.type = "cpu"
    else:
        value.shape = (7, 5120)
    assert not PagedCSA2Attention._uses_logical_mla(owner, value, False)
