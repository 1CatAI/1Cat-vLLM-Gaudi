# SPDX-License-Identifier: Apache-2.0
"""C6 dense chain against shared C1 operators on real retained operands.

Operator arithmetic only. Whole-model teacher-forced acceptance is a separate
mandatory gate before this numerical candidate enters the serving batch.
"""
from pathlib import Path


def audit(layer, residual, previous_pre, positions, rank, case, *, input_only=False):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    attention = layer.attention
    if hasattr(layer.weights, 'engram'):
        raise ValueError('Retain the Engram producer for an Engram-owned layer')
    native = torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2

    def projection(x, pre, pos):
        collapsed = (x.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        value, quantized, scale = native(collapsed.contiguous(), layer.weights.attn_norm.weight, layer.eps)
        query, kv = attention._project_qkv_input(value, prequant=(quantized, scale))
        query_output = (attention.project_query(rms_norm(query.contiguous(), attention.weights.q_norm.weight,
                                                         attention.eps), pos, decode=True)
                        if input_only else attention.project_query_input(query.contiguous(), pos))
        return (torch.cat((query, kv), -1), query_output,
                attention.project_kv(kv.contiguous(), pos, decode=True))

    # The producer compound is compiled in production. Eager expansion is
    # not its supported execution contract. This numerical-only wrapper is
    # outside the timed native plan and is never installed in serving.
    evaluate = torch.compile(projection, backend='hpu_backend', fullgraph=True, dynamic=False)
    names = ('qkv', 'query', 'kv')
    batch = dict(zip(names, evaluate(residual, previous_pre, positions), strict=True))
    reference = {name: [] for name in batch}
    for i in range(positions.numel()):
        outputs = evaluate(residual[i:i + 1].contiguous(), previous_pre[i:i + 1].contiguous(),
                           positions[i:i + 1].contiguous())
        for name, value in zip(names, outputs, strict=True):
            reference[name].append(value.cpu())
    # Real MLA consumer inputs previously captured on the same TP geometry;
    # projection is stateless, and uses this candidate's checkpoint weights.
    path = Path('/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/'
                'request-c6-mla-stacked-pv-oracle-208') / f'attention-operands-rank{rank}-case{case}.pt'
    saved = torch.load(path, weights_only=True, map_location='cpu')[0]
    pv = saved['output'].to('hpu').contiguous()
    pos = saved['positions'].to('hpu', dtype=torch.int32).contiguous()

    def output(x, p):
        rotated = attention._rope(x, p, inverse=True)
        grouped = rotated.reshape(x.shape[0], attention.groups, -1).contiguous()
        return attention.project_output_consumer(attention.project_output(grouped))

    evaluate_output = torch.compile(output, backend='hpu_backend', fullgraph=True, dynamic=False)
    batch['output'] = evaluate_output(pv, pos)
    reference['output'] = [evaluate_output(pv[i:i + 1].contiguous(), pos[i:i + 1].contiguous()).cpu()
                           for i in range(pv.shape[0])]
    results = {}
    for name, actual in batch.items():
        x, y = actual.cpu().float(), torch.cat(reference[name]).float()
        delta = x - y
        relative = float(delta.norm() / y.norm().clamp_min(1e-30))
        results[name] = dict(max_abs=float(delta.abs().max()), relative_l2=relative,
                             exact=torch.equal(x, y),
                             passed=bool(torch.isfinite(x).all()) and relative <= .002
                             and bool(torch.allclose(x, y, atol=.5, rtol=.008)))
    return dict(reference='Shared C1 production operators on identical real operands and immutable weights',
                projections=results, passed=all(row['passed'] for row in results.values()),
                output_input_source=str(path), timed_graph_instrumented=False,
                teacher_forced_acceptance_qualified=False, full_service_quality_passed=False)
