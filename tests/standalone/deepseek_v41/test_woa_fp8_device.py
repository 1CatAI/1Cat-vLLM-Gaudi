# SPDX-License-Identifier: Apache-2.0
"""The native FP8 conversion and full projection require an explicit HPU lease."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_native_moe import HPU, torch
from vllm_gaudi.ops.deepseek_v41_woa_fp8 import covering_scale, decode_gaudi2, encode_gaudi2

QUANT = torch.ops.custom_op.custom_deepseek_v41_woa_quant_gaudi2
WOA = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_gaudi2


@HPU
@pytest.mark.parametrize("tokens", [1, 3, 32, 129, 512])
def test_quantization_group_layout_and_replay(tokens):
    torch.manual_seed(1041)
    compiled = torch.compile(QUANT, backend="hpu_backend", fullgraph=True, dynamic=False)
    device = torch.empty(tokens, 4, 4096, dtype=torch.bfloat16, device="hpu")
    for step in range(2):
        cpu = (torch.randn(tokens, 4, 4096) * (step + 1)).bfloat16()
        cpu[0, 0] = 0
        cpu[0, 1, :16] = torch.tensor(
            [0., -0., 240., -240., 232., 224., 1.0625, 1.1875, 2**-6, 2**-7, 2**-8, 2**-9, 128., -128., 120., -120.])
        values = cpu.float().numpy()
        scales = covering_scale(np.abs(values).max(-1, keepdims=True))
        expected = encode_gaudi2(values / scales).transpose(1, 0, 2).copy()
        device.copy_(cpu)
        actual, actual_scale = compiled(device)
        np.testing.assert_array_equal(actual.cpu().view(torch.uint8).numpy(), expected)
        np.testing.assert_array_equal(actual_scale.cpu().numpy(), scales.transpose(1, 0, 2))


@HPU
@pytest.mark.parametrize("tokens", [1, 3, 32, 512])
def test_complete_projection_and_persistent_replay(tokens):
    torch.manual_seed(4141)
    cpu_weight = torch.randn(4, 4096, 1024).numpy().astype(np.float32)
    scales = covering_scale(np.abs(cpu_weight).max(axis=1, keepdims=True))
    q = encode_gaudi2(cpu_weight / scales)
    weight = torch.from_numpy(q).view(torch.float8_e4m3fn).to("hpu")
    sw = torch.from_numpy(scales).to("hpu")
    x = torch.empty(tokens, 4, 4096, dtype=torch.bfloat16, device="hpu")
    compiled = torch.compile(WOA, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    for step in range(3):
        cpu = (torch.randn(tokens, 4, 4096) * (step + 1)).bfloat16()
        x.copy_(cpu)
        actual = compiled(x, weight, sw).cpu()
        # Independent FP32 CPU dot of exactly the quantized operand values.
        values = cpu.float().numpy()
        sx = covering_scale(np.abs(values).max(-1, keepdims=True))
        a = decode_gaudi2(encode_gaudi2(values / sx)).transpose(1, 0, 2).copy()
        b = decode_gaudi2(q)
        product = torch.bmm(torch.from_numpy(a), torch.from_numpy(b))
        reference = (product * torch.from_numpy(scales) * torch.from_numpy(sx.transpose(1, 0, 2).copy())).permute(
            1, 0, 2).reshape(tokens, 4096).bfloat16()
        torch.testing.assert_close(actual, reference, atol=0.5, rtol=0.008)
        assert torch.linalg.vector_norm(actual.float() -
                                        reference.float()) <= (torch.linalg.vector_norm(reference.float()) * 0.002)
        ordinary = WOA(x, weight, sw).cpu()
        if not torch.equal(ordinary, actual):
            torch.save({
                "compiled": actual,
                "ordinary": ordinary,
                "reference": reference,
                "input": cpu
            },
                       Path(os.environ["DSV41_RUN_EVIDENCE"]) / f"woa-replay-difference-{tokens}-{step}.pt")
        assert torch.equal(ordinary, actual), "same FP8 algorithm differs between ordinary and compiled execution"
        records.append({
            "step":
            step,
            "max_abs_error":
            float((actual.float() - reference.float()).abs().max()),
            "relative_l2":
            float(
                torch.linalg.vector_norm(actual.float() - reference.float()) /
                torch.linalg.vector_norm(reference.float()))
        })
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) /
     f"woa-output-{tokens}.json").write_text(json.dumps(records, indent=2) + '\n')
