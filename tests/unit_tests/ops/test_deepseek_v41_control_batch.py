# SPDX-License-Identifier: Apache-2.0
"""Batch control reuse must preserve row ownership and the original MACs."""
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

torch.set_num_threads(1)
torch.hpu.set_device(rank)
torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 7, 8, 16, 32, 64])
def test_control_reuse_exact_with_tail_zero_and_reordered_requests(batch):
    torch.manual_seed(4131)
    weight = torch.randn(24, 20480, dtype=torch.float32).to("hpu")
    reference = torch.compile(torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2,
                              backend="hpu_backend",
                              fullgraph=True,
                              dynamic=False)
    candidate = torch.compile(torch.ops.custom_op.custom_deepseek_v41_control_batch4_f32_gaudi2,
                              backend="hpu_backend",
                              fullgraph=True,
                              dynamic=False)
    source = torch.randn(batch, 20480).bfloat16().float()
    source[0].zero_()
    fixed = source.to("hpu")
    for change in range(3):
        fixed.copy_((source.roll(change, 0) * (1 if change != 1 else -0.125)).to("hpu"))
        expected = reference(fixed, weight).cpu()
        actual = candidate(fixed, weight).cpu()
        assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))


def test_batch_control_rejects_wrong_width():
    candidate = torch.ops.custom_op.custom_deepseek_v41_control_batch4_f32_gaudi2
    with pytest.raises(RuntimeError, match="20480"):
        candidate(torch.empty(4, 20479, device="meta"), torch.empty(24, 20480, device="meta"))
