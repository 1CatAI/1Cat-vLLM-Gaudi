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


@pytest.mark.parametrize("groups", [19, 48])
def test_reuse_matches_independent_decode_for_tails_padding_and_changing_ids(groups):
    torch.manual_seed(4686 + groups)
    experts, h, k = 32, 512, 128
    q13 = torch.randint(-32768, 32768, (experts, 1, h * 64), dtype=torch.int16, device="hpu")
    q2 = torch.randint(-32768, 32768, (experts, h // 256, k * 64), dtype=torch.int16, device="hpu")
    # Offset planes zero => exact FP4 base values; original planes are retained.
    s13 = torch.zeros(experts, 1, h * 8, dtype=torch.int16, device="hpu")
    s2 = torch.zeros(experts, h // 256, k * 8, dtype=torch.int16, device="hpu")
    c13 = torch.ones(experts, 1, 256, dtype=torch.bfloat16, device="hpu")
    c2 = torch.ones(experts, h // 256, 256, dtype=torch.bfloat16, device="hpu")
    base = torch.compile(torch.ops.custom_op.custom_deepseek_v41_grouped_n256_fp8_gaudi2,
                         backend="hpu_backend",
                         fullgraph=True,
                         dynamic=False)
    reuse = torch.compile(torch.ops.custom_op.custom_deepseek_v41_reused_n256_fp8_gaudi2,
                          backend="hpu_backend",
                          fullgraph=True,
                          dynamic=False)
    ids = torch.empty(1, groups, dtype=torch.int32, device="hpu")
    for case in range(4):
        value = torch.randn(groups, h).bfloat16().to("hpu")
        x, sx = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(value)
        route = torch.rand(groups, 1, device="hpu")
        host = torch.arange(groups, dtype=torch.int32).remainder(experts)
        if case == 1:
            host.fill_(7)
        if case == 2:
            host = torch.arange(groups, dtype=torch.int32) // 5
        if case == 3:
            host[::2] = -1
            host[-2:] = experts
        ids.copy_(host[None])
        args = (x.reshape(groups, 1, h), sx.reshape(groups, 1,
                                                    1), ids, route, q13, q2, s13, s2, mxfp4_bf16_lut("hpu"), c13, c2)
        expected = base(*args).cpu()
        actual = reuse(*args).cpu()
        assert torch.equal(expected, actual), (case, (expected.float() - actual.float()).abs().max())
        if case == 3:
            assert torch.count_nonzero(actual[(host < 0) | (host >= experts)]) == 0
