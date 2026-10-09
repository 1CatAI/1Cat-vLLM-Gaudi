# SPDX-License-Identifier: Apache-2.0
"""Check deferred post against the accepted controller and C1 post consumer."""


def audit(layer, residual):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import row_mean_square

    def reference(r, weight, scale, base, value):
        raw = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(r.flatten(1), weight)
        projection = raw[:, :24] + raw[:, 24:]
        rrms = torch.rsqrt(row_mean_square(r.flatten(1).float()) + layer.eps)
        gates = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            projection.contiguous(), rrms.contiguous(), scale, base)
        result, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value, r, gates[:, 4:8].contiguous(), gates[:, 8:].reshape(-1, 4, 4).contiguous(),
            gates[:, :4].contiguous())
        return result, collapsed, gates

    def candidate(r, weight, scale, base, value):
        raw = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(r.flatten(1), weight)
        projection = raw[:, :24] + raw[:, 24:]
        rrms = torch.rsqrt(row_mean_square(r.flatten(1).float()) + layer.eps)
        control = torch.cat((projection, rrms), -1).contiguous()
        return torch.ops.custom_op.custom_deepseek_v41_mhc_mme_post_collapse_gaudi2(
            value, r, control, scale, base, layer.eps)

    operands = (residual, layer.hc_attn_fn_mme, layer.weights.hc_attn_scale,
                layer.weights.hc_attn_base, residual[:, 0].contiguous())
    functions = [torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                 for fn in (reference, candidate)]
    expected, actual = [tuple(value.cpu() for value in fn(*operands)) for fn in functions]
    checks = []
    for index, (parent, selected) in enumerate(zip(expected, actual, strict=True)):
        delta = selected.float() - parent.float()
        relative = float(delta.norm() / parent.float().norm().clamp_min(1e-30))
        finite = bool(torch.isfinite(selected).all())
        close = (bool(torch.allclose(selected, parent, atol=5e-5, rtol=1e-4)) if index == 2 else
                 bool(torch.allclose(selected.float(), parent.float(), atol=.002, rtol=.016)))
        checks.append(dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                           relative_l2=relative, exact=torch.equal(selected, parent), finite=finite,
                           passed=finite and close and relative <= .002))
    return dict(reference="Existing BF16 hi/lo MME and RRMS; accepted native C1 gates/post/collapse",
                checks=checks, passed=all(check["passed"] for check in checks),
                teacher_forced_acceptance_qualified=False)
