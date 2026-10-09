# SPDX-License-Identifier: Apache-2.0
"""Capture real operands for a numerical audit; never used by timed serving."""

import types


def install_probe(attention, rows):
    import torch

    attention.register_buffer(
        "oracle_query", torch.empty((rows, attention.heads, 512), dtype=torch.bfloat16, device="hpu"), False
    )
    attention.register_buffer("oracle_output", torch.empty_like(attention.oracle_query), False)
    attention.register_buffer("oracle_selected", torch.empty((rows, 512), dtype=torch.int32, device="hpu"), False)
    project = attention.project_query
    finish = attention._finish_output

    def query(self, value, positions, **kwargs):
        result = project(value, positions, **kwargs)
        self.oracle_query.copy_(result)
        return result

    def output(self, value, positions, *args, **kwargs):
        self.oracle_output.copy_(value)
        self.oracle_selected.copy_(self.selection.indices[:rows])
        return finish(value, positions, *args, **kwargs)

    attention.project_query = types.MethodType(query, attention)
    attention._finish_output = types.MethodType(output, attention)


def actual_operands(attention, positions):
    import torch

    selected = attention.oracle_selected.cpu()
    positions = positions.cpu().int()
    pages = attention.shared.block_table.cpu()
    width = 128 // attention.ratio
    logical = selected.clamp_min(0).long()
    page_id = logical // width
    valid = (selected >= 0) & (page_id < pages.numel())
    physical = (pages[page_id.clamp_max(pages.numel() - 1)] * width + logical.remainder(width)).clamp_min(0)
    valid = valid & (physical < attention.cache.main.shape[0])
    packed = (
        attention.cache.main.index_select(0, physical.clamp_max(attention.cache.main.shape[0] - 1).flatten().to("hpu"))
        .cpu()
        .reshape(*selected.shape, 288)
    )
    packed = torch.where(valid.unsqueeze(-1), packed, 0)
    return dict(
        query=attention.oracle_query.cpu(),
        output=attention.oracle_output.cpu(),
        swa=attention.swa.cpu(),
        packed_main=packed,
        selected=selected,
        positions=positions,
        sink=attention.weights.attn_sink.cpu(),
        scale=attention.scale.cpu(),
        ratio=attention.ratio,
    )


def audit(left, right, *, reference_transform=None):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4, unpack_swa

    equal_inputs = {
        name: torch.equal(left[name], right[name])
        for name in ("query", "swa", "packed_main", "selected", "positions", "sink", "scale")
    }
    absolute = left["positions"].unsqueeze(-1) - 127 + torch.arange(128)
    window = unpack_swa(left["swa"]).bfloat16()[absolute.remainder(256).long()]
    main = unpack_fp4(left["packed_main"], group=16).bfloat16()
    keys = torch.cat((window, main), dim=1).double()
    valid = torch.cat((absolute >= 0, left["selected"] >= 0), dim=1)
    scores = left["query"].double() @ keys.transpose(-1, -2)
    scores *= left["scale"].double()
    scores = scores.masked_fill(~valid.unsqueeze(1), -torch.inf)
    sink = left["sink"].double().reshape(1, -1, 1).expand(scores.shape[0], -1, -1)
    probability = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
    oracle = probability @ keys
    if reference_transform is not None:
        oracle = reference_transform(oracle)
    rounded = oracle.bfloat16()
    result = dict(
        inputs_exact=equal_inputs,
        internal_bitwise_equal=torch.equal(left["output"], right["output"]),
        oracle="FP64 QK, sink softmax and PV over canonical BF16 decoded KV",
        performance_qualified=False,
    )
    for name, arm in (("parent", left), ("candidate", right)):
        error = arm["output"].double() - oracle
        result[name] = dict(
            max_abs=float(error.abs().max()),
            rms=float(error.square().mean().sqrt()),
            differing_from_rounded_oracle=int((arm["output"] != rounded).sum()),
            elements=rounded.numel(),
            finite=bool(torch.isfinite(arm["output"]).all()),
            relative_l2=float(error.norm() / oracle.norm().clamp_min(1e-30)),
            official_tolerance_close=bool(torch.allclose(arm["output"].double(), oracle, atol=0.5, rtol=0.008)),
        )
    difference = right["output"].float() - left["output"].float()
    result["between_arms"] = dict(max_abs=float(difference.abs().max()), rms=float(difference.square().mean().sqrt()))
    return result
