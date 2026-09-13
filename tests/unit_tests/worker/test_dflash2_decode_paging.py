# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the actual HPU multi-token decode metadata builder."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


def _inputs(monkeypatch, contexts, lengths, *, mrope_deltas=None, padded_batch=None):
    from vllm_gaudi.v1.worker import hpu_model_runner as module

    # Exercise the real CPU paging logic without acquiring an HPU or creating
    # a model. Only the transfer and metadata-container boundary are replaced.
    monkeypatch.setattr(module, "async_h2d_copy", lambda tensor, device: tensor)
    monkeypatch.setattr(module.HPUAttentionMetadataV1, "make_decode_metadata",
                        lambda **kwargs: SimpleNamespace(**kwargs))
    positions = torch.tensor([
        position for context, length in zip(contexts, lengths, strict=True)
        for position in range(context, context + length)
    ],
                             dtype=torch.int32)
    runner = SimpleNamespace(
        attn_block_size=128,
        bucketing_manager=SimpleNamespace(
            find_decode_bucket=lambda batch, blocks, *args: (padded_batch or batch, 0, blocks)),
        get_dp_padding=lambda count: 0,
        positions_cpu=positions,
        input_ids_cpu=torch.arange(positions.numel(), dtype=torch.int32) + 10,
        uses_mrope=mrope_deltas is not None,
        input_batch=SimpleNamespace(req_ids=[f'request-{index}' for index in range(len(contexts))]),
        requests={
            f'request-{index}': SimpleNamespace(mrope_position_delta=delta)
            for index, delta in enumerate(mrope_deltas or [])
        },
        _PAD_BLOCK_ID=99,
        _resolve_block=lambda block: block,
        _resolve_all_blocks=lambda blocks: blocks,
        pin_memory=False,
        device=torch.device("cpu"),
        interleaved_sliding_window=False,
        model_has_chunked_attention=False,
        num_mamba_like_layers=0,
        use_async_scheduling=False,
        use_contiguous_pa=False,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    runner.get_habana_paged_attn_buffers = MethodType(HPUModelRunner.get_habana_paged_attn_buffers, runner)
    tables = torch.tensor([[7, 3, 11, 16], [20, 14, 18, 15]][:len(contexts)], dtype=torch.int32)
    data = HPUModelRunner._create_decode_input_data(runner, len(contexts), lengths, np.asarray(contexts), tables)
    return data, tables


def _metadata(monkeypatch, contexts, lengths):
    data, tables = _inputs(monkeypatch, contexts, lengths)
    return data.attn_metadata, tables


@pytest.mark.parametrize("context", [0, 120, 121, 126, 127, 128, 249, 254, 255, 256])
@pytest.mark.parametrize("tokens", [1, 2, 7, 8])
def test_each_verification_token_has_its_own_causal_page_extent(monkeypatch, context, tokens):
    metadata, tables = _metadata(monkeypatch, [context], [tokens])
    for token in range(tokens):
        position = context + token
        mask = metadata.block_groups == token
        blocks = metadata.block_list[mask].tolist()
        usage = metadata.block_usage[mask].float().tolist()
        expected = tables[0, :(position // 128 + 1)].tolist()
        assert blocks == expected, (context, token, blocks, expected)
        assert usage == [128] * (len(expected) - 1) + [position % 128 + 1]
        assert sum(usage) == position + 1
        # The current token's write page must also appear in its read pages.
        assert int(metadata.slot_mapping[token, 0]) // 128 == blocks[-1]


def test_ragged_verification_does_not_write_padding_into_a_live_cache(monkeypatch):
    metadata, tables = _metadata(monkeypatch, [126, 11], [8, 3])
    for token in range(3):
        flat_index = 8 + token
        assert int(metadata.slot_mapping[flat_index, 0]) == int(tables[1, 0]) * 128 + 11 + token
    for token in range(3, 8):
        flat_index = 8 + token
        assert int(metadata.slot_mapping[flat_index, 0]) // 128 == 99


@pytest.mark.parametrize("tokens", [1, 2, 3, 7, 8])
@pytest.mark.parametrize("delta", [None, 0, -7, 9])
def test_mrope_decode_preserves_three_axes_for_every_query_token(monkeypatch, tokens, delta):
    data, tables = _inputs(monkeypatch, [126], [tokens], mrope_deltas=[delta])
    expected = torch.arange(126 + (delta or 0), 126 + (delta or 0) + tokens, dtype=torch.int32)
    torch.testing.assert_close(data.position_ids, expected.unsqueeze(0).expand(3, -1))
    assert data.token_ids.shape == (tokens, 1)
    for token in range(tokens):
        position = 126 + token
        # M-RoPE deltas affect model coordinates, never physical KV addresses.
        assert int(data.attn_metadata.slot_mapping[token, 0]) == int(tables[0, position // 128]) * 128 + position % 128


def test_mrope_ragged_decode_keeps_request_padding_between_position_spans(monkeypatch):
    data, _ = _inputs(monkeypatch, [126, 11], [8, 3], mrope_deltas=[-7, 9], padded_batch=3)
    expected = torch.full((3, 24), -1, dtype=torch.int32)
    expected[:, :8] = torch.arange(119, 127, dtype=torch.int32)
    expected[:, 8:11] = torch.arange(20, 23, dtype=torch.int32)
    torch.testing.assert_close(data.position_ids, expected)
