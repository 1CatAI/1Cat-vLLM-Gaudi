# SPDX-License-Identifier: Apache-2.0
"""Actual BF16 mHC collapse through C1 norm and the production QKV consumer."""


def audit(layer, residual, previous_pre):
    import torch

    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    if hasattr(layer.weights, 'engram'):
        raise ValueError('Input-norm oracle must retain the Engram producer')
    collapsed = (residual.float() * previous_pre.unsqueeze(-1)).sum(1).bfloat16().contiguous()
    weight = layer.weights.attn_norm.weight

    def consume(x, use_native):
        normalized = (torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(x, weight, layer.eps)
                      if use_native == 'reference' else rms_norm(x, weight, layer.eps, native_decode=use_native))
        query, kv = layer.attention._project_qkv_input(normalized)
        return normalized, torch.cat((query, kv), -1)

    evaluate = torch.compile(consume, backend='hpu_backend', fullgraph=True, dynamic=False)
    old_norm, old_projection = evaluate(collapsed, False)
    new_norm, new_projection = evaluate(collapsed, True)
    # Native arithmetic is shared with C1. M=6 and M=1 MME reductions need
    # only the already established official tolerance, not bit equality.
    c1 = [evaluate(collapsed[row:row + 1].contiguous(), 'reference') for row in range(collapsed.shape[0])]
    torch.hpu.synchronize()
    c1_norm = torch.cat([row[0].cpu() for row in c1])
    c1_projection = torch.cat([row[1].cpu() for row in c1])
    x = collapsed.cpu().double()
    ideal = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + layer.eps) * weight.cpu().double()).bfloat16()
    results = {}
    for name, actual, reference in (('input_norm', new_norm.cpu(), c1_norm),
                                    ('qkv', new_projection.cpu(), c1_projection)):
        delta = actual.float() - reference.float()
        relative = float(delta.norm() / reference.float().norm().clamp_min(1e-30))
        results[name] = dict(max_abs=float(delta.abs().max()), relative_l2=relative,
                             exact=torch.equal(actual, reference), finite=bool(torch.isfinite(actual).all()),
                             passed=bool(torch.isfinite(actual).all()) and relative <= .002
                             and bool(torch.allclose(actual.float(), reference.float(), atol=.5, rtol=.008)))
    norm = new_norm.cpu().float()
    parent = old_norm.cpu().float()
    consumer_delta = new_projection.cpu().float() - old_projection.cpu().float()
    return dict(reference='Same actual collapse and immutable weights, accepted C1 norm and QKV projection',
                input_shape=list(collapsed.shape), projections=results,
                norm_c1_exact=torch.equal(norm, c1_norm.float()),
                parent_max_abs=float((norm - parent).abs().max()),
                parent_fp64_max_abs=float((parent - ideal.float()).abs().max()),
                candidate_fp64_max_abs=float((norm - ideal.float()).abs().max()),
                consumer_parent_max_abs=float(consumer_delta.abs().max()),
                timed_graph_instrumented=False, teacher_forced_acceptance_qualified=False,
                passed=all(row['passed'] for row in results.values()) and results['input_norm']['exact'])
