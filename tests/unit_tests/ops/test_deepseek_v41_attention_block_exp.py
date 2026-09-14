# SPDX-License-Identifier: Apache-2.0
"""Block-lane exponent preparation against the original decoded-state kernel."""
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
OPS = torch.ops.custom_op
EVIDENCE = Path(os.environ["DSV41_RUN_EVIDENCE"])


def exact(name, expected, actual):
    expected, actual = expected.cpu(), actual.cpu()
    dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    expected, actual = expected.view(dtype), actual.view(dtype)
    mismatches = int((expected != actual).sum())
    (EVIDENCE / f"{name}.json").write_text(json.dumps(dict(elements=expected.numel(), mismatches=mismatches)) + "\n")
    if mismatches:
        torch.save(dict(expected=expected, actual=actual), EVIDENCE / f"{name}.pt")
    assert not mismatches, (name, mismatches)


@pytest.mark.parametrize("heads,kind", [(1, "finite"), (32, "finite"), (64, "finite"), (32, "special")])
def test_block_coefficients_exact(heads, kind):
    torch._dynamo.reset()
    torch.manual_seed(4512 + heads)
    swa = torch.randn(1024, 512).bfloat16()
    main = torch.randn(768, 512).bfloat16()
    query = torch.randn(1, heads, 512).bfloat16()
    sink = torch.randn(heads)
    if kind == "special":
        codes = torch.tensor([0, -32768, 1, 127, 128, 32640, -128, 32704, -63], dtype=torch.int16)
        swa[1022] = codes.repeat(57)[:512].view(torch.bfloat16)
        main[255] = codes.roll(3).repeat(57)[:512].view(torch.bfloat16)
        query[0, 0].zero_()
        sink[:4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.])
    swa, main, query, sink = [x.to("hpu") for x in (swa, main, query, sink)]
    scale = torch.tensor([512**-.5], device="hpu")
    completion = torch.zeros(16, dtype=torch.int32, device="hpu")

    def program(q, ids, lengths):
        inputs = (q, swa, main, ids, sink, scale, lengths, completion, completion, 512, 256)
        return (OPS.custom_deepseek_v41_decoded_attn_bf16_gaudi2(*inputs),
                OPS.custom_deepseek_v41_decoded_attn_block_bf16_gaudi2(*inputs))

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation, length in enumerate((0, 1, 63, 64, 65, 127, 128, 129, 511, 640)):
        ids = (torch.arange(640, dtype=torch.int32) * 13 + generation) % 768
        ids[60:68] = torch.tensor([-1, 767, 767, 768, 511, 0, 9999, -1], dtype=torch.int32)
        if kind == "special":
            ids[1:4] = torch.tensor([510, 767, 510], dtype=torch.int32)
        # Changing query/indices and a tensor-valued length reuse the same graph.
        q = query if generation % 2 == 0 else -query
        expected, actual = compiled(q, ids[None].to("hpu"), torch.tensor([length], dtype=torch.int32, device="hpu"))
        for field, (left, right) in enumerate(zip(expected, actual, strict=True)):
            exact(f"block-{heads}-{kind}-{length}-{field}", left, right)
    torch._dynamo.reset()
