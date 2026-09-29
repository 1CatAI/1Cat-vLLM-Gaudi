# SPDX-License-Identifier: Apache-2.0
"""Physical KV pools may exceed a request's logical context capacity."""
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention


@pytest.fixture(scope="module", autouse=True)
def native():
    if os.getenv("DSV41_TEST_HPU") != "1":
        pytest.skip("Requires an exclusive HPU lease and qualified runtime")
    import habana_frameworks.torch.core  # noqa: F401
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("width,group", [(128, 32), (512, 16)])
def test_large_pool_tail_mapping_and_invalid_pages(ratio, width, group):
    torch.manual_seed(7743)
    # Include a partial last page, high physical addresses, invalid signed page
    # IDs and logical padding. Never materialize a full decoded physical pool.
    extent, page_rows = 1885443, 128 // ratio
    tail = extent // page_rows
    packed = torch.zeros(extent, width // 2 + width // group, dtype=torch.uint8, device="hpu")
    chunks = {}
    for page in (0, tail - 1, tail):
        count = min(page_rows, extent - page * page_rows)
        chunk = torch.randint(0, 256, (count, packed.shape[1]), dtype=torch.uint8)
        chunk[:, width // 2:] = 125 if group == 32 else 56
        chunks[page] = chunk
        packed[page * page_rows:page * page_rows + count].copy_(chunk)
    table = torch.zeros(8192, dtype=torch.int32)
    table[:6] = torch.tensor([tail - 1, tail, -1, 2147483647, extent // page_rows + 1, 2**32 // page_rows])
    rows = torch.tensor([0, page_rows - 1, page_rows, page_rows + 2, page_rows + 3,
                         2 * page_rows, 3 * page_rows, 4 * page_rows, 5 * page_rows, -1,
                         8192 * page_rows], dtype=torch.int32)
    for generation in range(2):
        if generation:
            table[:2] = table[:2].flip(0)
        expected = torch.zeros(rows.numel(), width, dtype=torch.bfloat16)
        for index, row in enumerate(rows.tolist()):
            if 0 <= row < table.numel() * page_rows:
                page, offset = int(table[row // page_rows]), row % page_rows
                if page in chunks and offset < len(chunks[page]):
                    expected[index] = unpack_fp4(chunks[page][offset:offset + 1], width, group)[0]
        if width == 128:
            actual = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(
                packed, table.to("hpu"), rows[None].to("hpu"), ratio).cpu()[0]
        else:
            actual = torch.ops.custom_op.custom_deepseek_v41_prefill_main_decode_gaudi2(
                packed, table.to("hpu"), rows.to("hpu"), ratio).cpu()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("local_heads", [8, 16])
def test_large_pool_compound_scores_and_visibility(ratio, local_heads):
    torch.manual_seed(8831)
    extent, page_rows = 1885440, 128 // ratio
    packed = torch.zeros(extent, 68, dtype=torch.uint8, device="hpu")
    table = torch.zeros(8192, dtype=torch.int32, device="hpu")
    high_pages = torch.arange(extent // page_rows - 4, extent // page_rows, dtype=torch.int32)
    chunk = torch.randint(0, 256, (4 * page_rows, 68), dtype=torch.uint8)
    chunk[:, 64:] = 125
    packed[-4 * page_rows:].copy_(chunk)
    query = torch.randn(7, 32, 128).bfloat16().to("hpu")
    weights = (torch.randn(7, 32) * .02).bfloat16().to("hpu")
    rows = torch.arange(128, dtype=torch.int32, device="hpu")
    owner = SimpleNamespace(ratio=ratio, index_heads=local_heads,
                            tensor_parallel_size=32 // local_heads, cache=SimpleNamespace(index=packed))
    owner.shared = SimpleNamespace(block_table=table,
                                   physical_rows=lambda r, _: (table[(r // page_rows).long()] * page_rows +
                                                               r % page_rows))
    for generation in range(2):
        table[:4].copy_(high_pages.roll(generation))
        positions = ((torch.arange(7, dtype=torch.int32, device="hpu") * 29 + 8) * ratio) + generation
        expected = PagedCSA2Attention._scores(owner, positions, rows, query, weights).cpu()
        actual = torch.ops.custom_op.custom_deepseek_v41_prefill_paged_index_scores_gaudi2(
            query, weights, packed, table, positions, rows, ratio, local_heads).cpu()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
