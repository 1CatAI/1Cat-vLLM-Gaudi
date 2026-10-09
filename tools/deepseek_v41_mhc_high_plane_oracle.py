# SPDX-License-Identifier: Apache-2.0
"""Compare the prepared controller through the accepted gates and post consumer."""


def audit(layer, residual):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import row_mean_square

    def consume(r, weight, scale, base, value):
        projection = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(r.flatten(1), weight)
        if weight.shape[0] == 48:
            projection = projection[:, :24] + projection[:, 24:]
        rrms = torch.rsqrt(row_mean_square(r.flatten(1).float()) + layer.eps)
        gates = torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            projection.contiguous(), rrms.contiguous(), scale, base)
        result, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            value, r, gates[:, 4:8].contiguous(), gates[:, 8:].reshape(-1, 4, 4).contiguous(),
            gates[:, :4].contiguous())
        return projection, gates, result, collapsed

    weight = layer.hc_attn_fn_packed
    high = weight.bfloat16()
    low = (weight - high.float()).bfloat16()
    other = layer.weights.hc_attn_scale, layer.weights.hc_attn_base, residual[:, 0].contiguous()
    evaluate = torch.compile(consume, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected = tuple(value.cpu().float() for value in evaluate(residual, torch.cat((high, low)).contiguous(), *other))
    actual = tuple(value.cpu().float() for value in evaluate(residual, high.contiguous(), *other))
    checks = []
    for index, (parent, selected) in enumerate(zip(expected, actual, strict=True)):
        delta = selected - parent
        relative = float(delta.norm() / parent.norm().clamp_min(1e-30))
        finite = bool(torch.isfinite(selected).all())
        close = True if index == 0 else bool(torch.allclose(selected, parent, atol=.5, rtol=.008))
        checks.append(dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                           relative_l2=relative, exact=torch.equal(selected, parent), finite=finite,
                           passed=finite and close and relative <= .002))
    return dict(reference="Prepared BF16 hi/lo MME, accepted C1 gates/post consumer",
                checks=checks, passed=all(check["passed"] for check in checks),
                teacher_forced_acceptance_qualified=False)
