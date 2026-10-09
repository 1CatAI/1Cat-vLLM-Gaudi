# SPDX-License-Identifier: Apache-2.0
"""Check the actual MLA/BF16 producer through the common C1 FP8 output."""
from pathlib import Path


def audit(attention, rank, case):
    import torch

    path = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/"
                "request-c6-mla-stacked-pv-oracle-208") / f"attention-operands-rank{rank}-case{case}.pt"
    saved = torch.load(path, weights_only=True, map_location="cpu")[0]
    value = saved["output"].to("hpu").contiguous()
    positions = saved["positions"].to("hpu", dtype=torch.int32).contiguous()

    def consume(x, p):
        rotated = attention._rope(x, p, inverse=True)
        grouped = rotated.reshape(x.shape[0], attention.groups, -1).contiguous()
        return attention.project_output_consumer(attention.project_output(grouped))

    evaluate = torch.compile(consume, backend="hpu_backend", fullgraph=True, dynamic=False)
    actual = evaluate(value, positions).cpu().float()
    reference = torch.cat([evaluate(value[i:i + 1].contiguous(), positions[i:i + 1].contiguous()).cpu()
                           for i in range(value.shape[0])]).float()
    delta = actual - reference
    relative = float(delta.norm() / reference.norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    return dict(reference="Actual MLA output; shared C1 inverse RoPE and production output projection",
                max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                relative_l2=relative, exact=torch.equal(actual, reference),
                passed=finite and relative <= .002 and bool(torch.allclose(actual, reference, atol=.5, rtol=.008)),
                source=str(path), teacher_forced_acceptance_qualified=False)
