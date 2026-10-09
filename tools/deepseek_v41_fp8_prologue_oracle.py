# SPDX-License-Identifier: Apache-2.0
"""Untimed C1 operator oracle for the actual C6 prologue producer/consumer."""


def audit(layer, residual, previous_pre, positions):
    import torch

    attention = layer.attention
    if hasattr(layer.weights, 'engram'):
        raise ValueError('Retain Engram in the full native chain; this local oracle uses source20')

    def producer(x, pre):
        collapsed = (x.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        return torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(
            collapsed.contiguous(), layer.weights.attn_norm.weight, layer.eps)

    def evaluate(x, pre, pos):
        value, q, scale = producer(x, pre)
        raw_query, raw_kv = attention._project_qkv_input(value, prequant=(q, scale))
        normalized = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(
            raw_query.contiguous(), attention.weights.q_norm.weight, attention.eps)
        query = attention.project_query_input(raw_query.contiguous(), pos)
        kv = attention.project_kv(raw_kv.contiguous(), pos, decode=True)
        fused = attention._project_fp8_qkv_prologue(value, pos, prequant=(q, scale))
        return normalized, query, kv, *fused

    # The numerical oracle is untimed. Performance always comes from the
    # complete captured production layers, peer communication and cache state.
    outputs = torch.compile(evaluate, backend='hpu_backend', fullgraph=True, dynamic=False)(
        residual, previous_pre, positions)
    report = {}
    for name, reference, candidate in zip(('normalized_query', 'query_rope', 'kv_rope'),
                                         outputs[:3], outputs[3:], strict=True):
        x, y = candidate.cpu().float(), reference.cpu().float()
        delta = x - y
        relative = float(delta.norm() / y.norm().clamp_min(1e-30))
        report[name] = dict(exact=torch.equal(x, y), max_abs=float(delta.abs().max()),
                            relative_l2=relative, passed=bool(torch.isfinite(x).all()) and relative <= .002
                            and bool(torch.allclose(x, y, atol=.5, rtol=.008)))
    return dict(reference='Shared C1 FP8 GEMM/normalization/RoPE with identical checkpoint weights',
                outputs=report, passed=all(x['passed'] for x in report.values()),
                timed_native_chain_instrumented=False, teacher_forced_acceptance_qualified=False)
