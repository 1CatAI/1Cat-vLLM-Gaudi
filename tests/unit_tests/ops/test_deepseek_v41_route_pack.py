# SPDX-License-Identifier: Apache-2.0
"""Bounded route-island register reuse preserves all materialized weight consumers."""
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

from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402

torch.set_num_threads(1)
torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
def test_native_order_pack_and_reduce_preserve_bytes_and_route_order(batch):
    torch.manual_seed(9910 + batch)
    ops = torch.ops.custom_op

    def chain(x, sx, ids, route, rows):
        inverse = ops.custom_deepseek_v41_route_order_gaudi2(ids)
        packed = ops.custom_deepseek_v41_route_pack_gaudi2(x, sx, ids, route, inverse)
        result = ops.custom_deepseek_v41_route_reduce_gaudi2(rows, inverse)
        return (*packed, inverse, result)

    compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    width, routes = 5120, batch * 6
    for case in range(4):
        ids = torch.randint(0, 384, (1, routes), dtype=torch.int32)
        if case == 1:
            ids.fill_(17)
        elif case == 2:
            ids = (torch.arange(routes, dtype=torch.int32).remainder(384).flip(0)).reshape(1, -1)
        elif case == 3:
            ids[0, ::2] = -1
            ids[0, -1] = 384
        # All 256 bit patterns prove byte movement, including FP8 exceptional encodings.
        x = torch.randint(0, 256, (batch, width), dtype=torch.uint8)
        sx = torch.rand(batch, 1)
        route = torch.rand(1, routes)
        rows = torch.randn(routes, 1, width).bfloat16()
        order = sorted(range(routes), key=lambda i: (int(ids[0, i]), i))
        order = torch.tensor(order)
        inverse = torch.empty(routes, dtype=torch.int32)
        inverse[order] = torch.arange(routes, dtype=torch.int32)
        token = order // 6
        ordered = rows[inverse.long()].reshape(batch, 6, width)
        total = ordered[:, 0].float()
        for i in range(1, 6):
            total = total + ordered[:, i].float()
        expected = (x[token].reshape(routes, 1, width), sx[token].reshape(routes, 1, 1), ids[:, order],
                    route[:, order].reshape(routes, 1), inverse, total.bfloat16())
        actual = compiled(*(t.to("hpu") for t in (x, sx, ids, route, rows)))
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert torch.equal(a.cpu(), e), (batch, case, index)


@pytest.mark.parametrize("batch", [8, 32])
def test_native_route_consumer_matches_original_moe(batch):
    from vllm_gaudi.ops.deepseek_v41_grouped_decode import grouped_decode
    torch.manual_seed(701 + batch)
    experts, h, k = 32, 512, 128
    q13 = torch.randint(-32768, 32768, (experts, 1, h * 64), dtype=torch.int16, device="hpu")
    q2 = torch.randint(-32768, 32768, (experts, h // 256, k * 64), dtype=torch.int16, device="hpu")
    s13 = torch.zeros(experts, 1, h * 8, dtype=torch.int16, device="hpu")
    s2 = torch.zeros(experts, h // 256, k * 8, dtype=torch.int16, device="hpu")
    c13 = torch.ones(experts, 1, 256, dtype=torch.bfloat16, device="hpu")
    c2 = torch.ones(experts, h // 256, 256, dtype=torch.bfloat16, device="hpu")
    lookup = mxfp4_bf16_lut("hpu")

    def candidate(x, ids, weights):
        return grouped_decode(x,
                              ids,
                              weights,
                              q13,
                              q2,
                              s13,
                              s2,
                              lookup,
                              c13,
                              c2,
                              rows=1,
                              reuse_weights=True,
                              native_pack=True)

    def reference(x, ids, weights):
        return grouped_decode(x, ids, weights, q13, q2, s13, s2, lookup, c13, c2, rows=1)

    fast = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    base = torch.compile(reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    for change in range(3):
        x = torch.randn(batch, h).bfloat16().to("hpu")
        ids = torch.randint(0, experts, (batch, 6), dtype=torch.int32)
        if change == 1:
            ids[:] = torch.arange(6)
        ids = ids.to("hpu")
        weights = torch.rand(batch, 6, device="hpu")
        assert torch.equal(base(x, ids, weights).cpu(), fast(x, ids, weights).cpu())
