# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the decode-specialized V4.1 mHC gate chain."""
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


def reference(mixes, rrms, scale, base):
    value = mixes * rrms
    pre = torch.sigmoid(value[:, :4] * scale[0] + base[:4]) + 1e-6
    post = torch.sigmoid(value[:, 4:8] * scale[1] + base[4:8]) * 2
    comb = torch.softmax(value[:, 8:].reshape(-1, 4, 4) * scale[2] + base[8:].reshape(1, 4, 4), -1) + 1e-6
    return pre, post, torch.ops.custom_op.custom_deepseek_v4_sinkhorn4_gaudi2(comb.contiguous())


@pytest.mark.parametrize("tokens", [1, 3])
def test_real_chain_shapes_values_and_replay(tokens):
    torch.manual_seed(140914 + tokens)
    inputs = [
        torch.randn(tokens, 24).float().to("hpu"),
        torch.rand(tokens, 1).float().to("hpu"),
        torch.randn(3).float().to("hpu"),
        torch.randn(24).float().to("hpu")
    ]
    candidate = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2
    compiled = torch.compile(candidate, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected = tuple(value.cpu() for value in reference(*inputs))
    actual_gates = candidate(*inputs).cpu()
    replayed_gates = compiled(*inputs).cpu()
    actual = (actual_gates[:, :4], actual_gates[:, 4:8], actual_gates[:, 8:].reshape(-1, 4, 4))
    replayed = (replayed_gates[:, :4], replayed_gates[:, 4:8], replayed_gates[:, 8:].reshape(-1, 4, 4))
    records = []
    for name, old, new, replay in zip(("pre", "post", "comb"), expected, actual, replayed, strict=True):
        torch.testing.assert_close(new, old, rtol=2e-5, atol=2e-6)
        assert torch.equal(new, replay)
        records.append({
            "output": name,
            "maximum_absolute_error": float((new - old).abs().max()),
            "eager_compiled_equal": True
        })
    inputs[0].mul_(1.25)
    changed = compiled(*inputs).cpu()
    assert not torch.equal(replayed_gates, changed)
    Path(os.environ["DSV41_RUN_EVIDENCE"], f"mhc-gates-{tokens}.json").write_text(json.dumps(records, indent=2))
