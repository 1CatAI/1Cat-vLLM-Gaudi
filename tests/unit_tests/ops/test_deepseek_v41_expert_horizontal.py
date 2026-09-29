# SPDX-License-Identifier: Apache-2.0
"""Horizontal W13 preserves route order and the complete quantized MoE contract."""
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


@pytest.mark.parametrize("variant", ["horizontal", "prequant", "direct_finalize", "transpose_mme"])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
def test_horizontal_w13_preserves_complete_moe(batch, variant):
    # Each parameter is an independent shape-contract test, not serving warmup.
    torch._dynamo.reset()
    torch.manual_seed(5250 + batch)
    experts, hidden, intermediate = 32, 512, 256
    q13 = torch.randint(-32768, 32768, (experts, 2, hidden * 64), dtype=torch.int16, device="hpu")
    q2 = torch.randint(-32768, 32768, (experts, 2, intermediate * 64), dtype=torch.int16, device="hpu")
    s13 = torch.zeros(experts, 2, hidden * 8, dtype=torch.int16, device="hpu")
    s2 = torch.zeros(experts, 2, intermediate * 8, dtype=torch.int16, device="hpu")
    c13 = torch.ones(experts, 2, 256, dtype=torch.bfloat16, device="hpu")
    c2 = torch.ones(experts, 2, 256, dtype=torch.bfloat16, device="hpu")
    base = torch.compile(torch.ops.custom_op.custom_deepseek_v41_expert_n256_moe_fused_reduce_fp8_gaudi2,
                         backend="hpu_backend",
                         fullgraph=True,
                         dynamic=False)
    operation = getattr(
        torch.ops.custom_op, {
            "horizontal": "custom_deepseek_v41_expert_n256_moe_horizontal_fp8_gaudi2",
            "prequant": "custom_deepseek_v41_expert_n256_moe_prequant_horizontal_fp8_gaudi2",
            "direct_finalize": "custom_deepseek_v41_expert_n256_moe_horizontal_finalize_fp8_gaudi2",
            "transpose_mme": "custom_deepseek_v41_expert_n256_moe_horizontal_transpose_fp8_gaudi2",
        }[variant])
    horizontal = torch.compile(operation, backend="hpu_backend", fullgraph=True, dynamic=False)
    ids = torch.empty(batch, 6, dtype=torch.int32, device="hpu")
    for case in range(4):
        value = torch.randn(batch, hidden).bfloat16().to("hpu")
        route = torch.rand(batch, 6, device="hpu")
        route = route / route.sum(-1, keepdim=True) * 1.5
        host = torch.arange(batch * 6, dtype=torch.int32).reshape(batch, 6).remainder(experts)
        if case == 1:
            host = host.flip(1)
            host[:, 1] = host[:, 0]
        if case == 2:
            host.fill_(7)
            host[::2] = -1
        if case == 3:
            # Nonuniform binary scales exercise both multiplications before
            # each route is rounded; the final sum must keep top6 order.
            c13.copy_(torch.pow(2., torch.randint(-6, 4, c13.shape)).bfloat16())
            c2.copy_(torch.pow(2., torch.randint(-6, 4, c2.shape)).bfloat16())
        ids.copy_(host)
        args = (value, ids, route, q13, q2, s13, s2, mxfp4_bf16_lut("hpu"), c13, c2, True)
        expected = base(*args).cpu()
        if variant == "prequant":
            quantized, scale = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(value)
            args = (*args[:-1], quantized, scale, args[-1])
        actual = horizontal(*args).cpu()
        assert torch.equal(expected, actual), (batch, case, (expected.float() - actual.float()).abs().max())
        if case == 2:
            assert torch.count_nonzero(actual[::2]) == 0
