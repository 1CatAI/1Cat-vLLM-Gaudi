# SPDX-License-Identifier: Apache-2.0
"""Shared-KV matrix attention: masking, sink, replay and cache dependency."""
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
BF16_PV = os.environ.get("VLLM_HPU_DSV41_MLA_BF16_PV") == "1"
MLA_OP = OPS.custom_deepseek_v41_mla_bf16_pv_gaudi2 if BF16_PV else OPS.custom_deepseek_v41_mla_mme_gaudi2


def reference(q, swa, main, ids, sink, scale, lengths, ready, offset, rows, bf16_pv=False):
    valid = (torch.arange(ids.numel()) < max(0, lengths.item())) & (ids.flatten() >= 0)
    valid &= ids.flatten() < 512 + rows
    if not ready:
        valid.fill_(False)
    selected = torch.cat((swa[offset:offset + 512], main[:rows]))[ids.flatten().clamp(0, 511 + rows)]
    selected = torch.where(valid[:, None], selected, 0).float()
    scores = q[0].float() @ selected.T * scale
    scores[:, ~valid] = -torch.inf
    if bf16_pv:
        full = torch.cat((scores, sink[:, None]), dim=-1)
        exp = torch.exp(full - full.amax(-1, keepdim=True))
        product = exp[:, :-1].bfloat16().float() @ selected
        return (product / exp.sum(-1, keepdim=True)).unsqueeze(0).bfloat16()
    probs = torch.softmax(torch.cat((scores, sink[:, None]), dim=-1), dim=-1)[:, :-1]
    return (probs @ selected).unsqueeze(0).bfloat16()


@pytest.mark.parametrize("width,offset,rows", [(128, 0, 0), (640, 512, 128), (640, 0, 4)])
def test_mask_sink_and_replay(width, offset, rows):
    torch.manual_seed(719 + rows)
    args = [
        torch.randn(1, 32, 512).bfloat16(),
        torch.randn(1024, 512).bfloat16(),
        torch.randn(max(1, rows), 512).bfloat16(),
        torch.zeros(1, width, dtype=torch.int32),
        torch.randn(32),
        torch.tensor([512**-.5]),
        torch.tensor([width], dtype=torch.int32),
        torch.zeros(16, dtype=torch.int32),
        torch.zeros(36, dtype=torch.int32)
    ]
    device = [x.to("hpu") for x in args]
    fn = MLA_OP
    compiled = torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
    records = []
    for generation, length in enumerate([0, 1, 63, 128, width, width + 17, -1]):
        args[0].mul_(-1.125)
        args[3].copy_((torch.arange(width, dtype=torch.int32) * 13 + generation) % (512 + rows))
        args[3][0, :5] = torch.tensor([-1, 0, 0, 512 + rows, 511], dtype=torch.int32)
        args[6].fill_(length)
        for host, hpu in zip(args, device, strict=True):
            hpu.copy_(host)
        actual = compiled(*device, offset, rows).cpu()
        ordinary = fn(*device, offset, rows).cpu()
        assert torch.equal(actual, ordinary), "ordinary / compiled algorithm differs"
        expected = reference(*args[:7], True, offset, rows, bf16_pv=BF16_PV)
        error = (actual.float() - expected.float()).abs()
        records.append({
            "generation": generation,
            "length": length,
            "max_abs": error.max().item(),
            "rmse": error.square().mean().sqrt().item()
        })
        if BF16_PV:
            original = reference(*args[:7], True, offset, rows)
            records[-1]["fp32_reference_rmse"] = (actual.float() - original.float()).square().mean().sqrt().item()
        torch.testing.assert_close(actual, expected, rtol=.02, atol=.004)
    Path(os.environ["DSV41_RUN_EVIDENCE"], f"mla-{width}-{offset}-{rows}.json").write_text(json.dumps(records,
                                                                                                      indent=2))
    device[7].fill_(-1)
    assert torch.count_nonzero(compiled(*device, offset, rows).cpu()) == 0


def test_writer_and_consumer_in_one_recipe():
    torch.manual_seed(23)
    packed = torch.zeros(512, 528, dtype=torch.uint8, device="hpu")
    decoded = torch.zeros(1024, 512, dtype=torch.bfloat16, device="hpu")
    q = torch.randn(1, 32, 512).bfloat16().to("hpu")
    ids = torch.zeros(1, 128, dtype=torch.int32, device="hpu")
    lengths = torch.ones(1, dtype=torch.int32, device="hpu")
    sink = torch.zeros(32, device="hpu")
    scale = torch.tensor([512**-.5], device="hpu")

    def program(value, position):
        done = OPS.custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(packed, value, position, decoded, 512)
        ids = position.expand(1, 128).contiguous()
        output = MLA_OP(q, decoded, decoded, ids, sink, scale, lengths, done, done, 512, 0)
        return output, done

    compiled = torch.compile(program, backend="hpu_backend", fullgraph=True, dynamic=False)
    for generation, slot in enumerate([0, 127, 128, 511, 0]):
        value = torch.randn(1, 512).bfloat16().to("hpu")
        pos = torch.tensor([slot], dtype=torch.int32, device="hpu")
        actual, done = compiled(value, pos)
        ids.fill_(slot)
        expected = OPS.custom_deepseek_v41_decoded_attn_bf16_gaudi2(q, decoded, decoded, ids, sink, scale, lengths,
                                                                    done, done, 512, 0)[0]
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=.02, atol=.004)
        assert (done.cpu() == slot).all()
