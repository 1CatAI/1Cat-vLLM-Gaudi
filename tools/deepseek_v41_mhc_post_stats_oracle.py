# SPDX-License-Identifier: Apache-2.0
"""Actual attention producer, post/FFN consumers, and independent gate packet."""


def audit(layer, residual, previous_pre, positions):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, rms_norm

    w = layer.weights
    # Both arms consume this production attention output, not a fabricated
    # residual row. KV/selection are the actual fixture state of this layer.
    collapsed, pre, post, comb, gates = hc_pre(
        residual, previous_pre, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base,
        layer.eps, layer.hc_eps, layer.iterations, packed_fn=layer.hc_attn_fn_packed,
        decode=True, control_mme_weight=layer.hc_attn_fn_mme, return_gates=True)
    normalized = rms_norm(collapsed, w.attn_norm.weight, layer.eps)
    value = layer.attention(normalized, positions, decode=True)

    def reference(x, r, g):
        updated, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
            x, r, g[:, 4:8].contiguous(), g[:, 8:].reshape(-1, 4, 4).contiguous(), g[:, :4].contiguous())
        norm, routed_q, routed_scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            collapsed, w.ffn_norm.weight, layer.eps)
        shared_q, shared_scale = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(norm)
        return updated, collapsed, norm, routed_q, routed_scale, shared_q, shared_scale

    def candidate(x, r, g):
        return torch.ops.custom_op.custom_deepseek_v41_dspark_mhc_post_norm_statistics_gaudi2(
            x, r, g, w.ffn_norm.weight, layer.eps)

    def consume(values):
        raw = torch.ops.custom_op.custom_deepseek_v41_control_mme_f32_gaudi2(
            values[0].flatten(1).contiguous(), layer.hc_ffn_fn_mme)
        control = raw[:, :24] + raw[:, 24:]
        shared = layer.moe.shared_expert(values[2], prequant=(values[5], values[6]))
        return (*values, control, shared)

    functions = []
    for fn in (reference, candidate):
        def entry(x, r, g, operation=fn):
            return consume(operation(x, r, g))
        functions.append(torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False))
    expected, actual = [tuple(t.cpu() for t in fn(value, residual, gates)) for fn in functions]
    labels = ("residual", "collapse", "norm", "routed_q", "routed_scale",
              "shared_q", "shared_scale", "ffn_control", "shared_output")
    checks = {}
    for label, a, b in zip(labels, expected, actual, strict=True):
        delta = b.float() - a.float()
        relative = float(delta.norm() / a.float().norm().clamp_min(1e-30))
        finite = bool(torch.isfinite(b.float()).all())
        # Quantized bytes may change around FP8 rounding boundaries. Keep the
        # numerical errors and require the actual downstream consumers to fit
        # the established C1 envelope; full40 teacher alpha is a separate gate.
        tolerance = .005 if label == "shared_output" else .002
        passed = finite and (label in ("routed_q", "shared_q") or relative <= tolerance)
        checks[label] = dict(relative_l2=relative, max_abs=float(delta.abs().max()),
                             finite=finite, exact=torch.equal(a, b), tolerance=tolerance, passed=passed)
    return dict(passed=all(v["passed"] for v in checks.values()), checks=checks,
                reference="Independent native C1 gates/post, separate FFN and shared quantizers and real consumers",
                producer="Full current Attention on actual fixture residual/pre/positions/KV",
                teacher_forced_acceptance_qualified=False, timed_graph_instrumented=False)
