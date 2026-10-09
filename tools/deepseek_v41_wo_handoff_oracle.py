# SPDX-License-Identifier: Apache-2.0
"""C6 output handoff versus the accepted C1 operators on identical real rows."""
from pathlib import Path


def audit(attention, rank, case):
    import torch

    root = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/request-c6-mla-stacked-pv-oracle-208")
    saved = torch.load(root / f"attention-operands-rank{rank}-case{case}.pt",
                       weights_only=True, map_location="cpu")[0]
    value = saved["output"].to("hpu").contiguous()
    positions = saved["positions"].to("hpu", dtype=torch.int32).contiguous()
    wa, wb = attention.weights.wo_a, attention.weights.wo_b
    actual = torch.ops.custom_op.custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2(
        value, wa.weight, wa.channel_scale, wb.weight, wb.channel_scale,
        positions, attention._rotary_native_table()).cpu()
    rows = []
    for i in range(value.shape[0]):
        rotated = attention._rope(value[i:i + 1].contiguous(), positions[i:i + 1].contiguous(), inverse=True)
        grouped = rotated.reshape(1, attention.groups, attention.heads // attention.groups * 512).contiguous()
        rows.append(attention.project_output_consumer(attention.project_output(grouped)).cpu())
    reference = torch.cat(rows)
    delta = actual.float() - reference.float()
    relative = float(torch.linalg.vector_norm(delta) /
                     torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    close = bool(torch.allclose(actual.float(), reference.float(), atol=.5, rtol=.008))
    return dict(reference="Accepted C1 inverse RoPE, FP8 wo_a/group32 boundary and FP8 wo_b",
                max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                relative_l2=relative, bitwise_equal=torch.equal(actual, reference),
                finite=finite, official_tolerance_close=close,
                passed=finite and close and relative <= .002,
                timed_graph_instrumented=False)
