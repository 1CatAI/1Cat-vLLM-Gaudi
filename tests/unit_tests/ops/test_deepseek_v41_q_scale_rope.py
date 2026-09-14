# SPDX-License-Identifier: Apache-2.0
"""Preserve the scaled BF16 boundary and dynamic RoPE position exactly."""
import os

import pytest

if os.environ.get("DSV41_TEST_HPU") != "1":
    pytest.skip("Requires an explicit HPU lease", allow_module_level=True)

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
prepare_environment()
import habana_frameworks.torch.core  # noqa: E402,F401
import torch  # noqa: E402

torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])


def rope_table():
    angles = torch.arange(512).float().unsqueeze(1) * torch.linspace(.01, 1, 32).unsqueeze(0)
    return torch.cat((angles.cos(), angles.sin()), -1).contiguous().to("hpu")


def test_epilogue_preserves_rounding_and_head_mapping():
    fn = torch.ops.custom_op.custom_deepseek_v41_q_scale_rope_gaudi2
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(392)
    product = torch.randn(1, 16384)
    product[0, :8] = torch.tensor([0., -0., 1.00390625, 1.01171875, -1.00390625, -1.01171875, 1e-8, 1e8])
    sw = 2. ** ((torch.arange(16384).reshape(1, -1) % 9).float() - 4.)
    sx = torch.tensor([[.125]])
    x, w, scale = product.to("hpu"), sw.to("hpu"), sx.to("hpu")
    pos, table = torch.tensor([0], dtype=torch.int32, device="hpu"), rope_table()
    for position in (0, 63, 511):
        pos.copy_(torch.tensor([position], dtype=torch.int32, device="hpu"))
        expected = torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(
            (product * sw * sx).bfloat16().reshape(1, 32, 512).to("hpu"), pos, table).cpu().reshape(1, -1)
        assert torch.equal(compiled(x, w, scale, pos, table).cpu(), expected)
        assert torch.equal(fn(x, w, scale, pos, table).cpu(), expected)


def test_complete_projection_changing_inputs():
    fn = torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)

    def original(x, w, sw, pos, table):
        expanded = torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(x, w, sw).reshape(1, 32, 512)
        return torch.ops.custom_op.custom_deepseek_v41_rope_bf16_gaudi2(expanded, pos, table).reshape(1, -1)

    reference = torch.compile(original, backend="hpu_backend", fullgraph=True, dynamic=False)
    torch.manual_seed(910)
    w = torch.randint(-2, 3, (16384, 1280)).bfloat16().to("hpu").to(torch.float8_e4m3fn)
    sw = (2. ** ((torch.arange(16384).reshape(1, -1) % 7).float() - 5.)).to("hpu")
    x = torch.zeros(1, 1280, dtype=torch.bfloat16, device="hpu")
    table, pos = rope_table(), torch.tensor([0], dtype=torch.int32, device="hpu")
    outputs = []
    for generation in range(3):
        x.copy_(torch.randn(1, 1280).bfloat16().to("hpu"))
        pos.copy_(torch.tensor([generation * 255], dtype=torch.int32, device="hpu"))
        actual = compiled(x, w, sw, pos, table).cpu()
        assert torch.equal(actual, reference(x, w, sw, pos, table).cpu())
        assert torch.equal(actual, fn(x, w, sw, pos, table).cpu())
        outputs.append(actual)
    assert not torch.equal(outputs[0], outputs[1])
    with pytest.raises(RuntimeError, match="Q projection requires"):
        fn(x.expand(2, -1).contiguous(), w, sw, pos, table)
