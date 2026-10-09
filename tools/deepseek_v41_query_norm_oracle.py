# SPDX-License-Identifier: Apache-2.0
"""Real QKV producer and accepted C1 norm; never inserted in the timed graph."""


def real_query_input(layer, residual, previous_pre):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    # The first retained group layer has no preceding collapse handoff and
    # no Engram update. This is hc_pre's existing BF16 collapse boundary.
    if hasattr(layer.weights, "engram"):
        raise ValueError("Query norm fixture must retain its Engram producer")
    collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).bfloat16()
    value = rms_norm(collapsed, layer.weights.attn_norm.weight, layer.eps)
    attention = layer.attention
    if layer.decode_attention_norm_quant and "fused_qkv_channel" in attention._buffers:
        value, quant, scale = torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(
            collapsed.contiguous(), layer.weights.attn_norm.weight, layer.eps)
        query, _ = attention._project_qkv_input(value, prequant=(quant, scale))
    else:
        query, _ = attention._project_qkv_input(value)
    return query.contiguous()


def audit(layer, residual, previous_pre):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    query = real_query_input(layer, residual, previous_pre)
    attention = layer.attention
    weight = attention.weights.q_norm.weight
    native = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
    parent = rms_norm(query, weight, attention.eps)
    actual = native(query, weight, attention.eps)
    reference = torch.cat(tuple(native(query[i:i + 1].contiguous(), weight, attention.eps)
                                for i in range(query.shape[0])))
    torch.hpu.synchronize()
    x, w = query.cpu().double(), weight.cpu().double()
    ideal = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + attention.eps) * w).bfloat16().float()
    left, right, c1 = parent.cpu().float(), actual.cpu().float(), reference.cpu().float()
    delta = right - left
    relative = float(delta.norm() / left.norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(right).all())
    result = dict(accepted_c1_exact=torch.equal(right, c1),
                  parent_max_abs=float(delta.abs().max()), parent_relative_l2=relative,
                  parent_fp64_max_abs=float((left - ideal).abs().max()),
                  candidate_fp64_max_abs=float((right - ideal).abs().max()), finite=finite,
                  actual_producer_shape=list(query.shape), timed_graph_instrumented=False)
    result["passed"] = (result["accepted_c1_exact"] and finite and relative <= .002
                        and bool(torch.allclose(right, left, atol=.5, rtol=.008)))
    return result
