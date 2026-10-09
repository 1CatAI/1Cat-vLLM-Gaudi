# SPDX-License-Identifier: Apache-2.0
"""Diagnose constant-section lowering using common C1 numerical operators."""
from pathlib import Path


def audit(parent, candidate, residual, previous_pre, positions, destination):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import hc_pre, rms_norm

    names = ("collapse", "pre", "post", "mix", "attention_norm", "qkv", "query", "kv",
             "ffn_norm", "router", "shared")

    def evaluator(layer):
        def evaluate(x, pre, pos):
            w, attn = layer.weights, layer.attention
            collapsed, pre_out, post, comb = hc_pre(
                x, pre, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, layer.eps,
                layer.hc_eps, layer.iterations, packed_fn=layer.hc_attn_fn_packed, decode=True,
                control_mme_weight=layer.hc_attn_fn_mme)
            norm = rms_norm(collapsed, w.attn_norm.weight, layer.eps)
            q, kv = attn._project_qkv_input(norm)
            query = attn.project_query(rms_norm(q.contiguous(), attn.weights.q_norm.weight, layer.eps),
                                       pos, decode=True)
            kv_out = attn.project_kv(kv.contiguous(), pos, decode=True)
            # The real collapsed input is an actual checkpoint activation;
            # these stateless projections diagnose weight handling only.
            # Full Target state and teacher acceptance remain separate gates.
            ffn_input = rms_norm(collapsed, w.ffn_norm.weight, layer.eps)
            router = layer.moe._router_logits(ffn_input)
            shared = layer.moe.shared_expert(ffn_input)
            return (collapsed, pre_out, post, comb, norm, torch.cat((q, kv), -1), query, kv_out,
                    ffn_input, router, shared)
        return torch.compile(evaluate, backend="hpu_backend", fullgraph=True, dynamic=False)

    torch.hpu.disable_inference_mode()
    reference_fn = evaluator(parent)
    old_batch = tuple(x.cpu() for x in reference_fn(residual, previous_pre, positions))
    reference_rows = [reference_fn(residual[i:i+1].contiguous(), previous_pre[i:i+1].contiguous(),
                                   positions[i:i+1].contiguous()) for i in range(positions.numel())]
    c1 = tuple(torch.cat([row[k].cpu() for row in reference_rows]) for k in range(len(names)))
    torch.hpu.enable_inference_mode()
    current = tuple(x.cpu() for x in evaluator(candidate)(residual, previous_pre, positions))

    def comparison(actual, expected):
        x, y = actual.float(), expected.float()
        delta = x-y
        relative = float(delta.norm()/y.norm().clamp_min(1e-30))
        return dict(max_abs=float(delta.abs().max()), relative_l2=relative,
                    exact=torch.equal(actual, expected), finite=bool(torch.isfinite(x).all()),
                    passed=bool(torch.isfinite(x).all()) and relative <= .002
                    and bool(torch.allclose(x, y, atol=.5, rtol=.008)))

    result = dict(reference="Common C1 row operators on the same actual retained prefix, immutable checkpoint weights",
                  projections={name: dict(candidate_vs_C1=comparison(x, y),
                                          original_C6_vs_C1=comparison(z, y),
                                          candidate_vs_original_C6=comparison(x, z))
                               for name, x, y, z in zip(names, current, c1, old_batch, strict=True)},
                  shape=list(residual.shape), timed=False, teacher_forced_acceptance_qualified=False)
    result["passed"] = all(v["candidate_vs_C1"]["passed"] for v in result["projections"].values())
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(names=names, candidate=current, reference_C1=c1, original_C6=old_batch), destination)
    return result
