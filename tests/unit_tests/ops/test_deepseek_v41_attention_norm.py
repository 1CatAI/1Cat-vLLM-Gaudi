# SPDX-License-Identifier: Apache-2.0
"""FlashInfer row primitives adapted to the V4.1 FP32 product boundary."""
import json
import os
from pathlib import Path

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("width", [512, 1280])
@pytest.mark.parametrize("tokens", [1, 3])
def test_norm_edges_and_compiled_inputs(width, tokens):
    fn = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(width)
    x = torch.randn(tokens, width).bfloat16()
    w = torch.randn(width).bfloat16()
    if tokens > 1:
        x[0].zero_()
    hpu = x.to("hpu")
    weight = w.to("hpu")
    records = []
    for factor in (1., 1e-8, 1e8, -.5):
        value = (x * factor).bfloat16()
        hpu.copy_(value.to("hpu"))
        actual = compiled(hpu, weight, 1e-20).cpu()
        eager = fn(hpu, weight, 1e-20).cpu()
        assert torch.equal(actual, eager)
        v = value.float()
        expected = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-20) * w.float()).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=.009, atol=1e-7)
        records.append({"factor": factor, "different": int((actual != expected).sum()),
                        "maximum_absolute_error": float((actual.float() - expected.float()).abs().max()),
                        "eager_compiled_equal": True})
    Path(os.environ["DSV41_RUN_EVIDENCE"], f"norm-{width}-{tokens}.json").write_text(json.dumps(records, indent=2))
