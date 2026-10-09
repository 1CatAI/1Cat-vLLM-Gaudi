# SPDX-License-Identifier: Apache-2.0
"""Preserved BF16 projection and RoPE precision on real upstream values."""


def audit(layer, residual, previous_pre, positions):
    import torch
    from deepseek_v41_query_norm_oracle import real_query_input
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    attention = layer.attention
    qr = rms_norm(real_query_input(layer, residual, previous_pre),
                  attention.weights.q_norm.weight, attention.eps).contiguous()
    weight = attention.weights.wq_b
    expected = attention._rope(attention.linear(qr, weight).reshape(-1, attention.heads, 512), positions)
    op = torch.ops.custom_op.custom_deepseek_v41_q_bf16_projection_rope_gaudi2
    actual = op(qr, weight.weight.contiguous(), positions.to(torch.int32).contiguous(),
                attention._rotary_native_table(), hasattr(weight, "scale")).reshape(expected.shape)
    torch.hpu.synchronize()
    left, right = expected.cpu().float(), actual.cpu().float()
    delta = right - left
    relative = float(delta.norm() / left.norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(right).all())
    result = dict(reference="Same real BF16 QKV/norm producer, parent BF16 weight/activation codec/projection/RoPE",
                  parent_max_abs=float(delta.abs().max()), parent_relative_l2=relative,
                  bitwise_equal=torch.equal(left, right), finite=finite,
                  actual_producer_shape=list(qr.shape), timed_graph_instrumented=False,
                  tolerance="existing projection atol.5/rtol.008 and relativeL2<=.002")
    result["passed"] = finite and relative <= .002 and bool(torch.allclose(right, left, atol=.5, rtol=.008))
    return result
