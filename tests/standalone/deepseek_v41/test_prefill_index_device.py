# SPDX-License-Identifier: Apache-2.0
"""Device codec and changing-shape contracts on an explicitly leased HPU."""

from test_native_moe import HPU, torch

from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4


def assert_encoding_equal(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    assert torch.equal(actual.isnan(), expected.isnan())
    finite_encoding = ~expected.isnan()
    assert torch.equal(actual.view(torch.int16)[finite_encoding], expected.view(torch.int16)[finite_encoding])


@HPU
def test_index_key_codec_all_nibbles_scales_and_signed_underflow():
    op = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2
    codes = torch.arange(4096, dtype=torch.int32) % 16
    scales = torch.arange(4096, dtype=torch.int32) // 16
    packed = torch.cat(((codes | (codes << 4)).byte()[:, None].expand(-1, 64), scales.byte()[:, None].expand(-1, 4)),
                       1).to("hpu")
    pages = torch.arange(32, dtype=torch.int32).to("hpu")
    for start in (0, 2048):
        rows = torch.arange(start, start + 2048, dtype=torch.int32).reshape(1, -1).to("hpu")
        actual = op(packed, pages, rows, 1)[0]
        expected = unpack_fp4(packed[start:start + 2048], 128, 32)
        assert_encoding_equal(actual, expected)


@HPU
def test_index_key_shape_changes_do_not_reuse_stale_scalar_dimensions():
    op = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2
    torch.manual_seed(401)
    packed = torch.randint(0, 256, (4096, 68), dtype=torch.uint8)
    packed[:, 64:] = 125
    packed = packed.to("hpu")
    # Keep these in one process: an isolated shape did not catch stale
    # shape-derived node parameters in the Bridge's eager graph cache.
    for ratio in (1, 2):
        width = 128 // ratio
        pages = torch.randperm(4096 // width).int().to("hpu")
        for tokens, columns in ((1, 1), (3, 17), (128, 2048), (7, 513)):
            rows = torch.randint(0, 4096, (tokens, columns), dtype=torch.int32)
            rows[0, 0] = -1
            rows = rows.to("hpu")
            physical = pages.index_select(0, (rows.clamp_min(0).flatten() // width).long())
            physical = physical.reshape(rows.shape) * width + rows.clamp_min(0) % width
            expected = unpack_fp4(
                packed.index_select(0,
                                    physical.flatten().long()).reshape(tokens, columns, 68), 128, 32)
            expected = torch.where((rows >= 0)[..., None], expected, 0)
            assert_encoding_equal(op(packed, pages, rows, ratio), expected)


@HPU
def test_index_key_request_pages_keep_rows_independent():
    op = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2
    torch.manual_seed(781)
    packed = torch.randint(0, 256, (4096, 68), dtype=torch.uint8)
    packed[:, 64:] = torch.randint(119, 130, (4096, 4), dtype=torch.uint8)
    packed = packed.to("hpu")
    for ratio in (1, 2):
        width = 128 // ratio
        pages = torch.stack([torch.randperm(4096 // width) for _ in range(8)]).int().to("hpu")
        for columns in (17, 513, 2048):
            rows = torch.randint(-1, 4096, (8, columns), dtype=torch.int32)
            rows[0].fill_(-1)
            rows = rows.to("hpu")
            actual = op(packed, pages, rows, ratio)
            expected = torch.cat([op(packed, pages[i].clone(), rows[i:i + 1].clone(), ratio) for i in range(8)])
            assert_encoding_equal(actual, expected)


@HPU
def test_shared_reindex_preserves_candidate_order_and_bf16_ties():
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
    from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import compiled_index_scores, compiled_shared_reindex

    torch.manual_seed(805)
    for ratio, tokens, capacity in ((1, 17, 2048), (2, 3, 4096), (1, 128, 8192)):
        packed = torch.randint(0, 256, (capacity, 68), dtype=torch.uint8)
        packed[:, 64:] = torch.randint(121, 130, (capacity, 4), dtype=torch.uint8)
        packed = packed.to("hpu")
        table = torch.randperm(capacity // (128 // ratio)).int().to("hpu")
        positions = torch.linspace(0, capacity * ratio - 1, tokens).int().to("hpu")
        blocks = torch.stack([torch.randperm(capacity // 8) for _ in range(tokens)]).int()
        blocks[:, 7:13] = -1  # Internal holes must not become valid source rows.
        blocks = blocks.to("hpu")
        query = torch.randn(tokens, 32, 128).bfloat16().to("hpu")
        weights = torch.randn(tokens, 32).bfloat16().to("hpu")
        for tied in (False, True):
            if tied:
                weights.zero_()
            rows = blocks[..., None] * 8 + torch.arange(8, device="hpu", dtype=torch.int32)
            rows = torch.where(blocks[..., None] >= 0, rows, -1).flatten(1)
            best_values = best_rows = None
            for start in range(0, rows.shape[1], 2048):
                current = rows[:, start:start + 2048].contiguous()
                scores = compiled_index_scores(
                    (query.shape, current.shape, ratio, capacity))(query, weights, packed, table, positions, current,
                                                                   ratio, 16, True, True)
                best_values, best_rows = PagedCSA2Attention._merge_topk(best_values, best_rows, scores, current, 512)
            expected = torch.where((best_rows >= 0) & (best_rows < ((positions + 1) // ratio)[:, None]), best_rows,
                                   capacity).sort(-1).values
            expected = torch.where(expected < capacity, expected, -1).int()
            actual = compiled_shared_reindex(
                (query.shape, blocks.shape, ratio, capacity))(query, weights, packed, table, positions, blocks, ratio,
                                                              16, capacity)
            assert torch.equal(actual.cpu(), expected.cpu())
