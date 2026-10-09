# SPDX-License-Identifier: Apache-2.0
"""Actual C6 control/RRMS compared with the accepted C1 operator."""


def audit(layer, residual):
    import torch

    x = residual.flatten(1).contiguous()
    w = layer.hc_attn_fn_packed
    parent = torch.cat([
        torch.ops.custom_op.custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(
            x[row:row + 1].contiguous(), w, layer.eps).cpu()
        for row in range(x.shape[0])
    ])
    from vllm_gaudi import envs

    operator = (torch.ops.custom_op.custom_deepseek_v41_mhc_control_tiles_gaudi2
                if envs.VLLM_HPU_DSV41_DSPARK_MHC_CONTROL_TILES else
                torch.ops.custom_op.custom_deepseek_v41_mhc_control_reuse_gaudi2)
    actual = operator(x, w, layer.eps).cpu()
    delta = actual - parent
    relative = float(delta.norm() / parent.norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    result = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                  relative_l2=relative, bitwise_equal=torch.equal(actual, parent), finite=finite,
                  control_max_abs=float(delta[:, :24].abs().max()),
                  rrms_max_abs=float(delta[:, 24:].abs().max()),
                  control_bitwise_equal=torch.equal(actual[:, :24], parent[:, :24]),
                  rrms_bitwise_equal=torch.equal(actual[:, 24:], parent[:, 24:]),
                  passed=finite and bool(torch.allclose(actual, parent, atol=2e-4, rtol=1e-5))
                  and relative <= .002)
    return dict(reference="Accepted C1 FP32 control and BF16 RRMS on the same six residual rows",
                projections=dict(mhc_control=result), passed=result["passed"], full_service_quality_passed=False)
