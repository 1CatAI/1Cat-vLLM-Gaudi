# SPDX-License-Identifier: Apache-2.0
"""Exercise source-tile padding and the normal paged Prefill dispatch."""
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, unpack_fp4
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention, can_partition_prefill_index
from vllm_gaudi.ops.deepseek_v41_prefill_regions import validate_prefill_region_config


@pytest.fixture(scope="module")
def native():
    if os.getenv("DSV41_TEST_HPU") != "1":
        pytest.skip("Requires the qualified HPU runtime and an exclusive device lease")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("tokens,columns,ratio", [(7, 1, 1), (7, 127, 2), (127, 129, 1), (7, 2047, 2)])
def test_padded_source_rows_and_changed_positions(native, monkeypatch, tokens, columns, ratio):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_INDEX_SRAM", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_INDEX_MME", "1")
    torch.manual_seed(5117)
    packed = torch.randint(0, 256, (65536 + 128, 68), dtype=torch.uint8)
    packed[:, 64:] = 125
    packed = packed.to("hpu")
    table = torch.randperm(512).int().to("hpu")
    query = unpack_fp4(pack_fp4(torch.randn(tokens, 32, 128).bfloat16(), 32), 128, 32).to("hpu")
    weights = (torch.randn(tokens, 32) * .02).bfloat16().to("hpu")
    row_owner = torch.arange(columns + 17, dtype=torch.int64, device="hpu") + 512
    rows = row_owner[17:]
    positions = (torch.arange(tokens, dtype=torch.int64) * 53 + 512) * ratio
    owner = SimpleNamespace(ratio=ratio, index_heads=16, cache=SimpleNamespace(index=packed))
    page_rows = 128 // ratio
    owner.shared = SimpleNamespace(block_table=table,
                                   physical_rows=lambda r, _: table[
                                       (r // page_rows).long()] * page_rows + r % page_rows)
    for generation in range(2):
        current = (positions + generation * 271).to("hpu")
        if generation:
            query.neg_()
        expected = PagedCSA2Attention._scores(owner, current, rows, query, weights).cpu()
        actual = PagedCSA2Attention._prefill_scores(owner, current, rows, query, weights).cpu()
        assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))


def test_incomplete_sram_dispatch_fails_before_execution(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_INDEX_SRAM", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_INDEX_MME", "0")
    with pytest.raises(ValueError, match="explicit prefill MME"):
        validate_prefill_region_config()


@pytest.mark.parametrize("tokens,source,ratio", [(7, 512, 2), (127, 2056, 1), (128, 32768, 1)])
def test_reindex_candidate_order_and_visibility(native, tokens, source, ratio):
    from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import compiled_decoded_reindex
    torch.manual_seed(6147)
    query = torch.randn(tokens, 32, 128).bfloat16().to("hpu")
    weights = (torch.randn(tokens, 32) * .02).bfloat16().to("hpu")
    keys = torch.randn(source, 128).bfloat16().to("hpu")
    columns = min(2048, (source + 7) // 8)
    blocks_cpu = torch.randint(-1, (source + 7) // 8, (tokens, columns), dtype=torch.int32)
    blocks_cpu[:, :3] = torch.tensor([-1, 0, 0], dtype=torch.int32)
    blocks = blocks_cpu.to("hpu")
    positions = torch.empty(tokens, device="hpu", dtype=torch.int32)
    old = compiled_decoded_reindex(("test-reindex-parent", tokens, source, ratio))
    new = compiled_decoded_reindex(("test-reindex-native", tokens, source, ratio))
    for generation in range(2):
        positions.copy_((torch.arange(tokens, dtype=torch.int32) * 53 + source // (generation + 1)) * ratio - 1)
        if generation:
            query.neg_()
            weights.neg_()
            blocks.copy_(blocks_cpu.roll(1, 1))
        arguments = (query, weights, keys, positions, blocks, ratio, 16, True)
        expected = old(*arguments, False).cpu()
        actual = new(*arguments, True).cpu()
        assert torch.equal(expected, actual)


def test_incomplete_reindex_contract_fails_before_execution(monkeypatch):
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_INDEX_SRAM", "0")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM", "1")
    monkeypatch.setenv("VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE", "0")
    with pytest.raises(ValueError, match="shared decoded Reindex keys"):
        validate_prefill_region_config()


@pytest.mark.parametrize("tokens,source,expected", [
    (128, 32768, False),
    (512, 32768, False),
    (1024, 32768, True),
    (8192, 32768, True),
    (8192, 512, False),
    (8192, 32769, False),
])
def test_tp_query_partition_uses_only_qualified_graph_shapes(tokens, source, expected):
    assert can_partition_prefill_index(tokens, source) is expected
