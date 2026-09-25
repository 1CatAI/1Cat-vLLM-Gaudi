# SPDX-License-Identifier: Apache-2.0
"""Full MME selection preserves packed scoring and request page ownership."""
import json
import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank])

import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

from vllm_gaudi.ops.deepseek_v41_full_index_mme import full_index_mme_select as _full_index_mme_select  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_indexer import runtime_index_select  # noqa: E402
from vllm_gaudi.ops.deepseek_v41_math import fp4_roundtrip  # noqa: E402

torch.set_num_threads(1)
torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


def full_index_mme_select(*args, **kwargs):
    return _full_index_mme_select(*args, **kwargs, tiled_keys=os.environ.get("DSV41_TEST_INDEX_TILED_KEYS") == "1")


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_full_mme_exact_selection_and_publication(batch, ratio):
    torch._dynamo.reset()
    torch.manual_seed(2941)
    capacity = 8192
    packed = torch.randint(0, 256, (16384, 68), dtype=torch.uint8)
    packed[:, 64:] = torch.randint(119, 127, (16384, 4), dtype=torch.uint8)
    cache = packed.to("hpu")
    page_rows = 128 // ratio
    pages_cpu = torch.stack([torch.randperm(16384 // page_rows)[:capacity // page_rows] for _ in range(batch)]).int()
    boundary = [2051, 4095, 4096, -1, 512 * ratio - 1, 512 * ratio, 2559, 8191]
    pos_cpu = torch.tensor([boundary[i % len(boundary)] for i in range(batch)], dtype=torch.int32)
    positions, pages = pos_cpu.to("hpu"), pages_cpu.to("hpu")
    candidates = torch.full((batch, 2048), -1, device="hpu", dtype=torch.int32)
    # The production index query is group-32 FP4 round-tripped before TP gather.
    q_source = fp4_roundtrip(torch.randn(batch, 32, 128, device="hpu").bfloat16(), 32)
    query = q_source.clone()
    q_cpu = q_source.cpu()
    weight_cpu = (torch.randn(batch, 32) * 0.1).bfloat16()
    weights = weight_cpu.to("hpu")

    def reference(q, w, p, table):
        return runtime_index_select(q,
                                    w,
                                    cache,
                                    table,
                                    p,
                                    candidates,
                                    ratio=ratio,
                                    capacity=capacity,
                                    publish_candidates=True)

    def changed(q, w, p, table):
        return full_index_mme_select(q,
                                     w,
                                     cache,
                                     table,
                                     p,
                                     candidates,
                                     ratio=ratio,
                                     capacity=capacity,
                                     publish_candidates=True)

    reference = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    changed = torch.compile(changed, backend="hpu_backend", fullgraph=True, dynamic=False)
    for change in range(3):
        # Cross the tactic boundary in both directions, retain padding, and
        # change physical ownership rather than merely changing batch labels.
        active_pos = pos_cpu if change == 0 else (pos_cpu.roll(1) if change == 1 else (pos_cpu + 2048).clamp_max(8191))
        positions.copy_(active_pos.to("hpu"))
        pages.copy_(pages_cpu.roll(change, 0).to("hpu"))
        query.copy_(q_cpu.roll(change, 0).to("hpu"))
        weights.copy_((weight_cpu.roll(change, 0) if change != 2 else torch.zeros_like(weight_cpu)).to("hpu"))
        expected = [x.cpu() for x in reference(query, weights, positions, pages)]
        actual = [x.cpu() for x in changed(query, weights, positions, pages)]
        for name, want, got in zip(("selected", "blocks"), expected, actual):
            assert torch.equal(want, got), (batch, ratio, change, name, int((want != got).sum()))


def test_full_mme_long_candidate_pool_and_stale_hot_outputs():
    torch._dynamo.reset()
    torch.manual_seed(2942)
    packed = torch.randint(0, 256, (32768, 68), dtype=torch.uint8)
    packed[:, 64:] = 124
    cache = packed.to("hpu")
    pages = torch.stack((torch.arange(256), torch.arange(255, -1, -1))).int().to("hpu")
    query = fp4_roundtrip(torch.randn(2, 32, 128, device="hpu").bfloat16(), 32)
    weights = torch.rand(2, 32, device="hpu").bfloat16()
    candidates = torch.full((2, 2048), -1, device="hpu", dtype=torch.int32)
    positions = torch.zeros(2, device="hpu", dtype=torch.int32)

    def compare(fn, p):
        return fn(query, weights, cache, pages, p, candidates, ratio=1, capacity=32768, publish_candidates=True)

    ref = torch.compile(lambda p: compare(runtime_index_select, p), backend="hpu_backend", fullgraph=True)
    run = torch.compile(lambda p: compare(full_index_mme_select, p), backend="hpu_backend", fullgraph=True)
    for pair in ([2051, 20003], [20003, 2051], [-1, 511]):
        positions.copy_(torch.tensor(pair, device="hpu", dtype=torch.int32))
        expected = [x.cpu() for x in ref(positions)]
        actual = [x.cpu() for x in run(positions)]
        assert all(torch.equal(x, y) for x, y in zip(expected, actual))


def test_full_mme_consumes_current_cache_write_before_gather():
    from vllm_gaudi.ops.deepseek_v41_batch_attention import write_state_rows
    torch._dynamo.reset()
    batch, capacity = 4, 4096
    cache = torch.zeros(8192, 68, device="hpu", dtype=torch.uint8)
    pages = torch.arange(32, dtype=torch.int32).repeat(batch, 1).to("hpu")
    pages[1::2] += 32
    positions = torch.full((batch, ), 2051, dtype=torch.int32, device="hpu")
    candidates = torch.full((batch, 2048), -1, dtype=torch.int32, device="hpu")
    query = torch.ones(batch, 32, 128, dtype=torch.bfloat16, device="hpu")
    weights = torch.ones(batch, 32, dtype=torch.bfloat16, device="hpu")
    # Different physical ownership, no overlapping writes in the same step.
    rows_cpu = torch.tensor([2051, 6147, 2050, -1], dtype=torch.int32)
    rows = rows_cpu.to("hpu")
    data_cpu = torch.full((batch, 68), 0x77, dtype=torch.uint8)
    data_cpu[:, 64:] = 127
    data = data_cpu.to("hpu")

    def chain(values, dest):
        done = write_state_rows(cache, values, dest)
        return full_index_mme_select(query,
                                     weights,
                                     cache,
                                     pages,
                                     positions,
                                     candidates,
                                     ratio=1,
                                     capacity=capacity,
                                     publish_candidates=True,
                                     key_ready=done)

    run = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = torch.compile(lambda: runtime_index_select(
        query, weights, cache, pages, positions, candidates, ratio=1, capacity=capacity, publish_candidates=True),
                              backend="hpu_backend",
                              fullgraph=True,
                              dynamic=False)
    for change in range(3):
        cache.zero_()
        payload = data_cpu.clone()
        payload[:, :64] = 0x77 if change != 1 else 0xFF
        data.copy_(payload.to("hpu"))
        write_state_rows(cache, data, rows)
        torch.hpu.synchronize()
        expected = [x.cpu() for x in reference()]
        cache.zero_()
        actual = [x.cpu() for x in run(data, rows)]
        assert all(torch.equal(x, y) for x, y in zip(expected, actual))
        if change != 1:
            assert 2051 in actual[0][0].tolist()


@pytest.mark.parametrize("batch,columns", [(1, 1), (1, 131), (2, 2048), (8, 131), (32, 2048), (64, 257)])
def test_tiled_keys_preserve_raw_rows_and_mme_consumer(batch, columns):
    from vllm_gaudi.ops.deepseek_v41_reindex_mme import _bf16_boundary
    torch._dynamo.reset()
    torch.manual_seed(2947)
    cache = torch.randint(0, 256, (8192, 68), dtype=torch.uint8)
    cache[:, 64:] = torch.randint(118, 130, (8192, 4), dtype=torch.uint8)
    cache = cache.to("hpu")
    pages_cpu = torch.stack([torch.randperm(64) for _ in range(batch)]).int()
    pages = pages_cpu.to("hpu")
    rows_cpu = torch.randint(-128, 8320, (batch, columns), dtype=torch.int32)
    rows = rows_cpu.to("hpu")
    query = torch.randn(batch, 32, 128, device="hpu").bfloat16()
    weights = torch.rand(batch, 32, device="hpu").bfloat16()
    ops = torch.ops.custom_op

    def consume(table, selected, tiled):
        gather = ops.custom_deepseek_v41_index_keys_tiled_gaudi2 if tiled else ops.custom_deepseek_v41_index_keys_gaudi2
        keys = gather(cache, table, selected, 1)
        dots = _bf16_boundary(torch.bmm(query, keys.transpose(1, 2)))
        return ops.custom_deepseek_v41_index_reduce_gaudi2(dots.contiguous(), weights)

    # Complete tile multiples are required by the score reducer; irregular
    # tails still exercise the decoder directly against its established ABI.
    old = torch.compile(lambda p, r: consume(p, r, False), backend="hpu_backend", fullgraph=True)
    new = torch.compile(lambda p, r: consume(p, r, True), backend="hpu_backend", fullgraph=True)
    for change in range(3):
        pages.copy_(pages_cpu.roll(change, 0).to("hpu"))
        rows.copy_((rows_cpu if change == 0 else rows_cpu.flip(1)).to("hpu"))
        ref = ops.custom_deepseek_v41_index_keys_gaudi2(cache, pages, rows, 1).cpu()
        got = ops.custom_deepseek_v41_index_keys_tiled_gaudi2(cache, pages, rows, 1).cpu()
        assert torch.equal(ref, got)
        if columns % 128 == 0:
            assert torch.equal(old(pages, rows).cpu(), new(pages, rows).cpu())


@pytest.mark.parametrize("batch,ratio", [(1, 1), (4, 2), (32, 1), (64, 1)])
def test_tiled_reindex_preserves_sparse_candidate_slots(batch, ratio):
    from vllm_gaudi.ops.deepseek_v41_reindex_mme import reindex_mme_select
    torch._dynamo.reset()
    torch.manual_seed(2949)
    packed = torch.randint(0, 256, (32768, 68), dtype=torch.uint8)
    packed[:, 64:] = torch.randint(119, 127, (32768, 4), dtype=torch.uint8)
    cache = packed.to("hpu")
    page_rows = 128 // ratio
    pages = torch.stack([torch.randperm(32768 // page_rows)[:16384 // page_rows] for _ in range(batch)]).int().to("hpu")
    query = fp4_roundtrip(torch.randn(batch, 32, 128, device="hpu").bfloat16(), 32)
    weights = torch.rand(batch, 32, device="hpu").bfloat16()
    candidates_cpu = torch.stack([torch.randperm(2048) for _ in range(batch)]).int()
    candidates_cpu[:, 1::3] = -1
    candidates = candidates_cpu.to("hpu")
    positions = torch.zeros(batch, dtype=torch.int32, device="hpu")

    def run(pos, pool, tiled):
        return reindex_mme_select(query, weights, cache, pages, pos, pool, ratio=ratio, tiled_keys=tiled)

    old = torch.compile(lambda p, c: run(p, c, False), backend="hpu_backend", fullgraph=True)
    new = torch.compile(lambda p, c: run(p, c, True), backend="hpu_backend", fullgraph=True)
    for change, length in enumerate((2051, 16383, 511)):
        positions.copy_(
            torch.tensor([length if row % 3 != 2 else -1 for row in range(batch)], dtype=torch.int32, device="hpu"))
        candidates.copy_(candidates_cpu.roll(change, -1).to("hpu"))
        assert torch.equal(old(positions, candidates).cpu(), new(positions, candidates).cpu())
