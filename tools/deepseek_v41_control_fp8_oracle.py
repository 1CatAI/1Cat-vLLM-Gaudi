# SPDX-License-Identifier: Apache-2.0
"""Real control operands through accepted C1 gates and post consumers."""


def audit(layer, residual):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import row_mean_square

    def downstream(r, projection, rrms, value):
        gates = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            projection.contiguous(), rrms.contiguous(), layer.weights.hc_attn_scale, layer.weights.hc_attn_base)
        result, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value, r, gates[:, 4:8].contiguous(), gates[:, 8:].reshape(-1, 4, 4).contiguous(),
            gates[:, :4].contiguous())
        return gates, result, collapsed

    def reference(r, value):
        raw = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(r.flatten(1), layer.hc_attn_fn_mme)
        projection = raw[:, :24] + raw[:, 24:]
        rrms = torch.rsqrt(row_mean_square(r.flatten(1).float()) + layer.eps)
        return projection, rrms, *downstream(r, projection, rrms, value)

    def candidate(r, value):
        operation = (torch.ops.custom_op.custom_deepseek_v41_control_fp8_pair_gaudi2
                     if layer.hc_attn_fn_fp8.shape[0] == 48 else
                     torch.ops.custom_op.custom_deepseek_v41_control_fp8_rrms_gaudi2)
        projection, rrms = operation(
            r.flatten(1), layer.hc_attn_fn_fp8, layer.hc_attn_fn_fp8_channel, layer.eps)
        return projection, rrms, *downstream(r, projection, rrms, value)

    operand = residual, residual[:, 0].contiguous()
    evaluate = [torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                for fn in (reference, candidate)]
    parent, selected = [tuple(value.cpu().float() for value in fn(*operand)) for fn in evaluate]
    checks = []
    for index, (x, y) in enumerate(zip(parent, selected, strict=True)):
        delta = y - x
        relative = float(delta.norm() / x.norm().clamp_min(1e-30))
        finite = bool(torch.isfinite(y).all())
        passed = finite if index == 0 else relative <= (.0001 if index == 1 else .002)
        if index >= 2:
            passed = passed and bool(torch.allclose(x, y, atol=.5, rtol=.008))
        checks.append(dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                           relative_l2=relative, finite=finite, passed=passed))
    return dict(reference="Accepted C1 BF16-hi/lo FP32-output MME/RRMS through C1 gates/post consumer",
                checks=checks, passed=all(row["passed"] for row in checks),
                projection_error_reported_separately=True, teacher_forced_acceptance_qualified=False)
