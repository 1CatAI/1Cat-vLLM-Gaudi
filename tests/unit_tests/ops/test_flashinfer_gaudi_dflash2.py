# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for the FlashInfer-Gaudi DFlash2 primitives."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from flashinfer_gaudi import dflash2, top_k
from vllm.v1.attention.backend import AttentionType
from vllm_gaudi.attention.backends.hpu_attn import (
    HPUAttentionImpl,
    _attention_batch_size,
)
from vllm_gaudi.ops.hpu_dflash2 import _rms_norm_reference


@pytest.mark.parametrize("block_size", [7, 8])
def test_grouped_conv_matches_scalar_reference(block_size: int):
    generator = torch.Generator().manual_seed(23)
    batch, hidden_size, groups, group_size, taps = 3, 32, 4, 8, 2
    tokens = batch * block_size
    hidden = torch.randn(tokens, hidden_size, generator=generator)
    delta = torch.randn(tokens, taps, groups, generator=generator)
    base = torch.randn(taps, hidden_size, generator=generator)

    actual = dflash2.grouped_conv(hidden, delta, base, block_size, groups, group_size, taps)
    expected = torch.empty_like(actual)
    base_grouped = base.view(taps, groups, group_size)
    hidden_grouped = hidden.view(tokens, groups, group_size)
    for token in range(tokens):
        request_start = token - token % block_size
        value = (base_grouped[0] + delta[token, 0, :, None]) * hidden_grouped[token]
        for tap in range(1, taps):
            if token - tap >= request_start:
                value += (base_grouped[tap] + delta[token, tap, :, None]) * hidden_grouped[token - tap]
        expected[token] = value.flatten()

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape,weight_shape", [((5, 64), (64, )), ((5, 7, 8, 64), (5, 64))])
def test_dflash_context_rms_norm_matches_layerwise_reference(shape, weight_shape):
    torch.manual_seed(17)
    inputs = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    actual = _rms_norm_reference(inputs, weight, 1e-6)
    if weight.ndim == 1:
        expected = torch.nn.functional.rms_norm(inputs, (shape[-1], ), weight, 1e-6)
    else:
        expected = torch.stack([
            torch.nn.functional.rms_norm(inputs[layer], (shape[-1], ), weight[layer], 1e-6) for layer in range(shape[0])
        ])

    torch.testing.assert_close(actual, expected)


def test_hpu_attention_external_dflash_kv_update_uses_physical_slots():

    class CacheWriter(torch.nn.Module):

        def forward(self, values, cache, slots, **kwargs):
            cache.index_copy_(0, slots, values)
            return cache

    impl = HPUAttentionImpl.__new__(HPUAttentionImpl)
    torch.nn.Module.__init__(impl)
    impl.attn_type = AttentionType.DECODER
    impl.kv_sharing_target_layer_name = None
    impl.num_kv_heads = 2
    impl.head_size = 4
    impl.k_cache = CacheWriter()
    impl.v_cache = CacheWriter()

    key = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    value = -key
    key_cache = torch.zeros(8, 2, 4)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.tensor([1, 4, 7])
    impl.do_kv_cache_update(mock.Mock(), key, value, (key_cache, value_cache, None, None), slots)

    torch.testing.assert_close(key_cache.index_select(0, slots), key)
    torch.testing.assert_close(value_cache.index_select(0, slots), value)


def test_multi_token_decode_uses_expanded_paged_attention_batch():
    metadata = mock.Mock(
        is_prompt=False,
        block_mapping=torch.empty(5, 8),
        seq_lens_tensor=torch.tensor([8], dtype=torch.int32),
    )

    assert _attention_batch_size(metadata, use_merged_prefill=False) == 8


def test_prefill_attention_still_uses_request_batch():
    metadata = mock.Mock(
        is_prompt=True,
        block_mapping=None,
        seq_lens_tensor=torch.ones(3, dtype=torch.int32),
    )

    assert _attention_batch_size(metadata, use_merged_prefill=False) == 3


def test_grouped_conv_does_not_cross_request_boundaries():
    hidden = torch.zeros(16, 16)
    hidden[7] = 10
    delta = torch.zeros(16, 2, 2)
    base = torch.ones(2, 16)
    output = dflash2.grouped_conv(hidden, delta, base, 8, 2, 8, 2)
    assert torch.count_nonzero(output[8]) == 0


def test_score_edges_matches_explicit_lattice():
    generator = torch.Generator().manual_seed(31)
    batch, steps, top_k, vocab, rank = 2, 4, 3, 29, 5
    predecessor = torch.randn(vocab, rank, generator=generator)
    successor = torch.randn(vocab, rank, generator=generator)
    candidates = torch.randint(vocab, (batch, steps, top_k), generator=generator)
    unary = torch.randn(batch, steps, top_k, generator=generator)
    hidden = torch.randn(batch, steps, rank, generator=generator)
    anchors = torch.randint(vocab, (batch, ), generator=generator)

    actual = dflash2.score_edges(predecessor, successor, candidates, unary, hidden, anchors)
    expected = torch.empty_like(actual)
    for b in range(batch):
        for step in range(steps):
            pred_ids = [int(anchors[b])] if step == 0 else candidates[b, step - 1].tolist()
            for p, pred_id in enumerate(pred_ids):
                for c, candidate_id in enumerate(candidates[b, step].tolist()):
                    expected[b, step, p, c] = unary[b, step, c] + torch.dot(predecessor[pred_id] * hidden[b, step],
                                                                            successor[candidate_id])
            if step == 0:
                expected[b, step, 1:] = expected[b, step, :1]
    torch.testing.assert_close(actual, expected)


def _score_select_tensors(device: str = "meta") -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(248320, 256, dtype=torch.bfloat16, device=device),
        torch.empty(248320, 256, dtype=torch.bfloat16, device=device),
        torch.empty(2, 7, 16, dtype=torch.int64, device=device),
        torch.empty(2, 7, 16, dtype=torch.float32, device=device),
        torch.empty(2, 7, 256, dtype=torch.bfloat16, device=device),
        torch.empty(2, dtype=torch.int32, device=device),
    )


def test_score_select_native_shape_gate_matches_official_qwen38_contract():
    inputs = _score_select_tensors()
    assert dflash2._native_score_select_supported(*inputs)
    assert not dflash2._native_score_select_supported(*inputs[:-1], inputs[-1].to(torch.int64))


def test_score_select_auto_does_not_resolve_native_until_promoted():
    inputs = _score_select_tensors()
    native = mock.Mock()

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_score_select_auto_promoted", return_value=False),
            mock.patch.object(dflash2, "native_dflash2_score_select_op", return_value=native),
    ):
        actual = dflash2.maybe_score_and_select_path(*inputs)

    assert actual is None
    native.assert_not_called()


def test_score_select_auto_dispatches_promoted_native_kernel():
    inputs = _score_select_tensors()
    sentinel = torch.empty(2, 7, dtype=torch.int64, device="meta")
    native = mock.Mock(return_value=sentinel)

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_score_select_auto_promoted", return_value=True),
            mock.patch.object(dflash2, "native_dflash2_score_select_op", return_value=native),
    ):
        actual = dflash2.maybe_score_and_select_path(*inputs)

    assert actual is sentinel
    native.assert_called_once_with(*inputs)


def test_select_path_follows_previous_candidate_and_first_tie():
    candidates = torch.tensor([[[10, 11, 12], [20, 21, 22], [30, 31, 32]]])
    scores = torch.full((1, 3, 3, 3), -10.0)
    scores[0, 0, 0] = torch.tensor([0.0, 4.0, 4.0])
    scores[0, 1, 1] = torch.tensor([9.0, 0.0, 0.0])
    scores[0, 2, 0] = torch.tensor([0.0, 1.0, 8.0])
    assert dflash2.select_path(candidates, scores).tolist() == [[11, 20, 32]]


def _production_select_path_inputs(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(83 + batch)
    candidates = torch.randint(248320, (batch, 7, 16), dtype=torch.int64, generator=generator)
    scores = torch.randn(batch, 7, 16, 16, dtype=torch.float32, generator=generator)
    return candidates, scores


def test_select_path_auto_does_not_dispatch_unpromoted_native_kernel():
    candidates, scores = _production_select_path_inputs()
    expected = dflash2.select_path_reference(candidates, scores)
    native = mock.Mock()

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_select_path_auto_promoted", return_value=False),
            mock.patch.object(dflash2, "native_dflash2_select_path_op", return_value=native),
    ):
        actual = dflash2.select_path(candidates, scores)

    torch.testing.assert_close(actual, expected)
    native.assert_not_called()


def test_select_path_auto_dispatches_exact_qwen38_shape_after_promotion():
    candidates, scores = _production_select_path_inputs()
    sentinel = torch.full((candidates.shape[0], 7), 42, dtype=torch.int64)
    native = mock.Mock(return_value=sentinel)

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_select_path_auto_promoted", return_value=True),
            mock.patch.object(dflash2, "native_dflash2_select_path_op", return_value=native),
    ):
        actual = dflash2.select_path(candidates, scores)

    assert actual is sentinel
    native.assert_called_once_with(candidates, scores)


def test_select_path_native_rejects_non_production_score_dtype():
    candidates, scores = _production_select_path_inputs()
    scores = scores.to(torch.bfloat16)
    expected = dflash2.select_path_reference(candidates, scores)
    native = mock.Mock()

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_select_path_auto_promoted", return_value=True),
            mock.patch.object(dflash2, "native_dflash2_select_path_op", return_value=native),
    ):
        actual = dflash2.select_path(candidates, scores)

    torch.testing.assert_close(actual, expected)
    native.assert_not_called()


def test_top_k_matches_torch_and_orders_returned_ties_by_id():
    scores = torch.tensor([[1.0, 7.0, 2.0, 7.0, 3.0], [9.0, 8.0, 7.0, 6.0, 5.0]])
    values, indices = top_k(scores, 3)
    expected_values, _ = torch.topk(scores, 3, dim=-1)
    torch.testing.assert_close(values, expected_values)
    assert indices[0].tolist() == [1, 3, 4]
    torch.testing.assert_close(values, scores.gather(-1, indices))


def _production_top_k_scores(rows: int = 7) -> torch.Tensor:
    return torch.zeros(rows, 248320, dtype=torch.bfloat16)


def test_top_k_auto_keeps_canonical_reference_until_promoted():
    scores = _production_top_k_scores()
    sentinel = (torch.zeros(7, 16, dtype=torch.bfloat16), torch.zeros(7, 16, dtype=torch.int64))
    reference = mock.Mock(return_value=sentinel)
    vendor = mock.Mock()

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_top_k_auto_promoted", return_value=False),
            mock.patch.object(dflash2, "top_k_reference", reference),
            mock.patch.object(dflash2, "_top_k_vendor_cguid", vendor),
    ):
        actual = dflash2.top_k(scores, 16)

    assert actual is sentinel
    reference.assert_called_once_with(scores, 16, sorted=True, deterministic=True)
    vendor.assert_not_called()


def test_top_k_auto_uses_vendor_cguid_for_exact_qwen38_shape():
    scores = _production_top_k_scores()
    sentinel = (torch.zeros(7, 16, dtype=torch.bfloat16), torch.zeros(7, 16, dtype=torch.int64))
    vendor = mock.Mock(return_value=sentinel)

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_top_k_auto_promoted", return_value=True),
            mock.patch.object(dflash2, "native_dflash2_top_k_op", return_value=None),
            mock.patch.object(dflash2, "_top_k_vendor_cguid", vendor),
    ):
        actual = dflash2.top_k(scores, 16)

    assert actual is sentinel
    vendor.assert_called_once_with(scores, 16)


def test_top_k_vendor_cguid_rejects_non_production_vocab_shape():
    scores = torch.zeros(7, 1024, dtype=torch.bfloat16)
    sentinel = (torch.zeros(7, 16, dtype=torch.bfloat16), torch.zeros(7, 16, dtype=torch.int64))
    reference = mock.Mock(return_value=sentinel)
    vendor = mock.Mock()

    with (
            mock.patch.object(dflash2, "_is_hpu", return_value=True),
            mock.patch.object(dflash2, "dflash2_top_k_auto_promoted", return_value=True),
            mock.patch.object(dflash2, "top_k_reference", reference),
            mock.patch.object(dflash2, "_top_k_vendor_cguid", vendor),
    ):
        actual = dflash2.top_k(scores, 16)

    assert actual is sentinel
    reference.assert_called_once()
    vendor.assert_not_called()


def test_attention_bias_matches_flash_attention_window_boundary():
    # W=8 means seven positions to either side, matching FA's (W-1, W-1).
    bias = dflash2.build_attention_bias(
        query_starts=[10],
        first_blocks=[0],
        block_size=4,
        past_width=12,
        query_len=3,
        window=8,
        causal=False,
        dtype=torch.bfloat16,
    )

    assert bias.shape == (1, 1, 3, 15)
    assert torch.isneginf(bias[0, 0, 0, 2])
    assert bias[0, 0, 0, 3] == 0
    assert bias[0, 0, 0, 9] == 0
    assert torch.isneginf(bias[0, 0, 0, 10])
    assert torch.count_nonzero(bias[0, 0, :, 12:]) == 0


def test_attention_bias_encodes_causality_inside_query_block():
    bias = dflash2.build_attention_bias(
        query_starts=[4],
        first_blocks=[0],
        block_size=4,
        past_width=4,
        query_len=3,
        window=None,
        causal=True,
        dtype=torch.float32,
    )

    assert bias[0, 0, 0, 4] == 0
    assert torch.isneginf(bias[0, 0, 0, 5])
    assert bias[0, 0, 1, 5] == 0
    assert torch.isneginf(bias[0, 0, 1, 6])
    assert torch.count_nonzero(bias[0, 0, 2]) == 0


def test_attention_bias_masks_fetched_block_prefix_and_padded_request():
    bias = dflash2.build_attention_bias(
        query_starts=[10, 0],
        first_blocks=[1, 0],
        block_size=4,
        past_width=8,
        query_len=2,
        window=6,
        causal=False,
        dtype=torch.float32,
    )

    # Request zero fetches positions [4, 12); W=6 keeps [5, 10).
    assert torch.isneginf(bias[0, 0, 0, 0])
    assert torch.count_nonzero(bias[0, 0, 0, 1:6]) == 0
    assert torch.isneginf(bias[0, 0, 0, 6])
    # A graph-padding row has no valid context but retains its query block.
    assert torch.isneginf(bias[1, 0, :, :8]).all()
    assert torch.count_nonzero(bias[1, 0, :, 8:]) == 0


def test_device_prepare_keeps_accepted_prefix_and_redirects_rejected_rows():
    sampled = torch.tensor(
        [
            [11, 12, 13, -1, -1, -1, -1, -1],
            [21, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    base_positions = torch.tensor([126, 255, 0], dtype=torch.int64)
    block_table = torch.tensor(
        [
            [4, 7, 9, 99],
            [1, 3, 8, 99],
            [99, 99, 99, 99],
        ],
        dtype=torch.int64,
    )
    prepared = dflash2.prepare_device_inputs(
        sampled,
        base_positions,
        block_table,
        active_mask=torch.tensor([True, True, False]),
        first_blocks=torch.tensor([0, 1, 0]),
        last_blocks=torch.tensor([1, 2, -1]),
        block_size=128,
        num_query_per_req=8,
        mask_token_id=248070,
        pad_block_id=99,
        pad_slot_id=99 * 128,
        max_model_len=4096,
        max_context_blocks=2,
        window=2048,
        causal=False,
        bias_dtype=torch.bfloat16,
    )
    (
        accepted,
        context_positions,
        context_slots,
        input_ids,
        query_positions,
        query_slots,
        block_list,
        context_lens,
        attention_bias,
    ) = prepared

    torch.testing.assert_close(accepted, torch.tensor([3, 1, 0], dtype=torch.int32))
    torch.testing.assert_close(context_positions.reshape(3, 8)[0], torch.tensor([126, 127, 128, 0, 0, 0, 0, 0]))
    torch.testing.assert_close(
        context_slots.reshape(3, 8)[0, :3],
        torch.tensor([4 * 128 + 126, 4 * 128 + 127, 7 * 128]),
    )
    assert context_slots.reshape(3, 8)[0, 3:].ge(99 * 128).all()
    assert context_slots.reshape(3, 8)[0, 3:].lt(100 * 128).all()
    assert input_ids[:, 0].tolist() == [13, 21, 248070]
    assert input_ids[:, 1:].eq(248070).all()
    torch.testing.assert_close(query_positions[0], torch.arange(129, 137))
    torch.testing.assert_close(query_slots[0], torch.arange(7 * 128 + 1, 7 * 128 + 9))
    torch.testing.assert_close(block_list.reshape(3, 2), torch.tensor([[4, 7], [3, 8], [99, 99]]))
    torch.testing.assert_close(context_lens, torch.tensor([129, 128, 0], dtype=torch.int32))
    assert attention_bias.shape == (3, 1, 8, 264)
    assert torch.isneginf(attention_bias[2, :, :, :256]).all()
    assert torch.count_nonzero(attention_bias[2, :, :, 256:]) == 0


def test_device_prepare_attention_bias_matches_sequence_reference():
    sampled = torch.tensor([[8, 9, 10, 11, 12, -1, -1, -1]], dtype=torch.int32)
    first_blocks = torch.tensor([1], dtype=torch.int64)
    prepared = dflash2.prepare_device_inputs(
        sampled,
        base_positions=torch.tensor([250], dtype=torch.int64),
        block_table=torch.tensor([[2, 5, 7, 11]], dtype=torch.int64),
        active_mask=torch.tensor([True]),
        first_blocks=first_blocks,
        last_blocks=torch.tensor([2]),
        block_size=128,
        num_query_per_req=8,
        mask_token_id=99,
        pad_block_id=20,
        pad_slot_id=20 * 128,
        max_model_len=4096,
        max_context_blocks=2,
        window=128,
        causal=True,
        bias_dtype=torch.float32,
    )

    expected = dflash2.build_attention_bias(
        query_starts=[255],
        first_blocks=[1],
        block_size=128,
        past_width=256,
        query_len=8,
        window=128,
        causal=True,
        dtype=torch.float32,
    )
    torch.testing.assert_close(prepared[-1], expected)


def test_shape_validation_rejects_cross_contract_inputs():
    with pytest.raises(ValueError, match="hidden_size"):
        dflash2.grouped_conv(torch.zeros(8, 15), torch.zeros(8, 2, 2), torch.zeros(2, 15), 8, 2, 8, 2)
    with pytest.raises(ValueError, match="scores must have shape"):
        dflash2.select_path(torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 2, 3))
