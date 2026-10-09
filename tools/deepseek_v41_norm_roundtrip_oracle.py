# SPDX-License-Identifier: Apache-2.0
"""Real mHC/QKV producers; keep the accepted C1 BF16/quant boundaries."""


def audit(layer, residual, previous_pre, positions):
    import torch

    if hasattr(layer.weights, "engram"):
        raise ValueError("Retain the Engram producer before qualifying this layer")
    attention = layer.attention
    norm = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
    quant = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
    fused = torch.ops.custom_op.custom_deepseek_v41_norm_roundtrip_bf16_gaudi2

    def entry(x, pre, pos, candidate):
        collapsed = (x.float() * pre.unsqueeze(-1)).sum(1).bfloat16().contiguous()
        if candidate:
            value, rounded = fused(collapsed, layer.weights.attn_norm.weight, layer.eps)
        else:
            value = norm(collapsed, layer.weights.attn_norm.weight, layer.eps)
            rounded = quant(value)
        query, kv = attention._project_qkv_input(value, roundtrip=rounded)
        if candidate:
            qr, qr_rounded = fused(query.contiguous(), attention.weights.q_norm.weight, layer.eps)
        else:
            qr = norm(query.contiguous(), attention.weights.q_norm.weight, layer.eps)
            qr_rounded = quant(qr)
        output = attention.project_query(qr, pos, decode=True, roundtrip=qr_rounded)
        return value, rounded, query, kv, qr, qr_rounded, output

    evaluate = torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
    reference = tuple(x.cpu() for x in evaluate(residual, previous_pre, positions, False))
    candidate = tuple(x.cpu() for x in evaluate(residual, previous_pre, positions, True))
    names = ("input_norm", "input_quant", "qkv_query", "qkv_kv", "query_norm", "query_quant", "query_rope")
    results = {}
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        delta = actual.float() - expected.float()
        results[name] = dict(exact=torch.equal(actual, expected), max_abs=float(delta.abs().max()),
                             relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
                             finite=bool(torch.isfinite(actual).all()))
    return dict(reference="Separate accepted C1 row norm/group32 roundtrip and actual BF16 QKV/query consumers",
                results=results, passed=all(x["exact"] and x["finite"] for x in results.values()),
                producer_rows=residual.shape[0], timed_graph_instrumented=False,
                teacher_forced_acceptance_qualified=False)
