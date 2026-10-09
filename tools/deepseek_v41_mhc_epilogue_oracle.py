# SPDX-License-Identifier: Apache-2.0
"""Isolated gates precision on the actual group-entry residual and weights."""


def audit(layer, residual, *, statistics=False):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import row_mean_square

    if layer.eps != 1e-20 or layer.hc_eps != 1e-6 or layer.iterations != 20:
        raise ValueError("mHC epilogue is confined to the qualified checkpoint constants")
    flat = residual.flatten(1).contiguous()
    weight = layer.hc_attn_fn_mme
    scale, base = layer.weights.hc_attn_scale, layer.weights.hc_attn_base

    def reference(x, w, s, b):
        p = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(x, w)
        if w.shape[0] == 48:
            p = p[:, :24] + p[:, 24:]
        rrms = torch.rsqrt(row_mean_square(x.float()) + layer.eps)
        return torch.ops.custom_op.custom_deepseek_v41_mhc_gates_f32_gaudi2(
            p.contiguous(), rrms.contiguous(), s.contiguous(), b.contiguous())

    def candidate(x, w, s, b):
        p = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(x, w)
        operator = (torch.ops.custom_op.custom_deepseek_v41_mhc_statistics_epilogue_gaudi2 if statistics else
                    torch.ops.custom_op.custom_deepseek_v41_mhc_mme_epilogue_gaudi2)
        return operator(
            p.contiguous(), x, s.contiguous(), b.contiguous())

    functions = [torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                 for fn in (reference, candidate)]
    expected, actual = [fn(flat, weight, scale, base).cpu() for fn in functions]
    delta = actual - expected
    relative = float(delta.norm() / expected.norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    result = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                  relative_l2=relative, bitwise_equal=torch.equal(actual, expected), finite=finite,
                  passed=finite and bool(torch.allclose(actual, expected, atol=5e-5, rtol=1e-4))
                  and relative <= .002)
    return dict(reference="Qualified MME hi/lo projection and existing FP32 RRMS/Sinkhorn gates",
                projections=dict(mhc_gates=result), passed=result["passed"], full_service_quality_passed=False)
