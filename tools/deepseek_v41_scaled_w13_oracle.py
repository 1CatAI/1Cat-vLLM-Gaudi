# SPDX-License-Identifier: Apache-2.0
"""Real-row FFN producer and accepted C1 routed expert consumer comparison."""


def audit(layer, residual, previous_pre):
    import torch

    moe = layer.moe

    def produce(value, pre):
        collapsed = (value.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        normalized, quantized, scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            collapsed.contiguous(), layer.weights.ffn_norm.weight, layer.eps)
        logits = torch.nn.functional.linear(normalized.float(), moe.weights.gate.weight)
        ids, routing = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
            logits.contiguous(), moe.weights.gate.bias, moe.weights.gate.bias_vl,
            torch.zeros(value.shape[0], dtype=torch.bool, device=value.device))
        return normalized, ids, routing, quantized, scale

    inputs = torch.compile(produce, backend='hpu_backend', fullgraph=True, dynamic=False)(residual, previous_pre)
    value, ids, routing, quantized, scale = inputs

    def selected(value, ids, routing, quantized, scale):
        return moe._forward_n256_fp8(value, ids, routing, prequant=(quantized, scale))

    def reference(value, ids, routing, quantized, scale):
        return moe._forward_n256_fp8(value, ids, routing, ordinary_decode=True, prequant=(quantized, scale))

    candidate = torch.compile(selected, backend='hpu_backend', fullgraph=True, dynamic=False)
    c1 = torch.compile(reference, backend='hpu_backend', fullgraph=True, dynamic=False)
    actual = candidate(*inputs).cpu().float()
    expected = torch.cat([c1(*(x[i:i + 1].contiguous() for x in inputs)).cpu() for i in range(value.shape[0])]).float()
    delta = actual - expected
    relative = float(delta.norm() / expected.norm().clamp_min(1e-30))
    return dict(reference='Accepted one-row C1 N256 FP8 routed expert, same normalized real rows and routes',
                tolerance_source='test_expert_n256.py: component relative RMS below0.005; teacher mean alpha mandatory',
                max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()), relative_l2=relative,
                exact=torch.equal(actual, expected), finite=bool(torch.isfinite(actual).all()),
                passed=bool(torch.isfinite(actual).all()) and relative < .005,
                producer='Actual request residual/pre, checkpoint FFN norm, qualified norm/quant and original router',
                teacher_forced_acceptance_qualified=False)
