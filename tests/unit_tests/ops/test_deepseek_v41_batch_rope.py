# SPDX-License-Identifier: Apache-2.0
"""Concurrent rows preserve the native C1 arithmetic and position mapping."""
import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


@pytest.mark.parametrize("inverse", [False, True])
def test_batch_matches_c1_with_changing_rows_and_long_positions(inverse):
    torch._dynamo.reset()
    name = "custom_deepseek_v41_rope_inverse_bf16_gaudi2" if inverse else "custom_deepseek_v41_rope_bf16_gaudi2"
    op = getattr(torch.ops.custom_op, name)
    compiled = torch.compile(op, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(751)
    table = torch.zeros(1048576, 64)
    positions = torch.tensor([0, 127, 511, 512, 8191, 8192, 131071, 1048575], dtype=torch.int32)
    phase = torch.arange(8).float()[:, None] * torch.linspace(.01, .71, 32)[None, :]
    table[positions.long()] = torch.cat((phase.cos(), phase.sin()), -1)
    table = table.to("hpu")
    source = torch.randn(64, 4, 512).bfloat16()
    source[:, :, 0] = -0.
    position = positions.repeat(8)
    for batch in (1, 4, 8, 32, 64):
        x, p = source[:batch].to("hpu"), position[:batch].to("hpu")
        for change in range(2):
            if change:
                x.copy_(source.flip(0)[:batch].to("hpu"))
                p.copy_(position.flip(0)[:batch].to("hpu"))
            expected = torch.cat(
                [op(x[i:i + 1].contiguous(), p[i:i + 1].contiguous(), table).cpu() for i in range(batch)])
            actual = compiled(x, p, table).cpu()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(actual[..., :-64].view(torch.int16), x.cpu()[..., :-64].view(torch.int16))
    with pytest.raises(RuntimeError, match="RoPE input contract"):
        op(source[:1].expand(65, -1, -1).contiguous().to("hpu"), torch.zeros(65, dtype=torch.int32, device="hpu"),
           table)
