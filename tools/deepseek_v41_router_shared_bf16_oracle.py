# SPDX-License-Identifier: Apache-2.0
"""Joint dense FFN against the accepted one-row C1 numerical path, outside timing."""
import copy


def audit(layer, residual, pre):
    import torch

    moe = layer.moe
    def views(module):
        result = copy.copy(module)
        result._buffers = module._buffers.copy()
        result._parameters = module._parameters.copy()
        result._modules = {k: views(v) if v is not None else None for k, v in module._modules.items()}
        return result

    c1 = views(moe)
    c1.router_bf16_gate = True
    c1.weights.gate.weight = c1.weights.gate.weight.bfloat16()
    c1.reduce = lambda x, **kw: x
    selected = views(moe)
    selected.reduce = lambda x, **kw: x

    def produce(residual, pre):
        collapse = (residual.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        return torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
            collapse.contiguous(), layer.weights.ffn_norm.weight, layer.eps)

    x, q, sx = torch.compile(produce, backend='hpu_backend', fullgraph=True, dynamic=False)(residual, pre)
    mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    def candidate(x, q, sx, mask):
        return selected(x, mask, decode=True, prequant=(q, sx))

    def reference(x, q, sx, mask):
        return c1(x, mask, ordinary_decode=True, prequant=(q, sx))

    actual = torch.compile(candidate, backend='hpu_backend', fullgraph=True, dynamic=False)(x, q, sx, mask)
    reference = torch.compile(reference, backend='hpu_backend', fullgraph=True, dynamic=False)
    expected = torch.cat([reference(x[i:i+1].contiguous(), q[i:i+1].contiguous(), sx[i:i+1].contiguous(),
                                    mask[i:i+1].contiguous()) for i in range(x.shape[0])])
    paired = torch.ops.custom_op.custom_deepseek_v41_dense_bf16_pair_gaudi2(x.contiguous())
    sq, ss = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(x.contiguous())
    effective = (sq.float()*ss).bfloat16()
    paired_cpu = paired.cpu()
    original_exact = torch.equal(paired_cpu[:x.shape[0]], x.cpu())
    shared_exact = torch.equal(paired_cpu[x.shape[0]:], effective.cpu())
    joint = torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
        paired.contiguous(), moe.router_shared_bf16_weight)
    first = moe.shared_gate_up_weight.shape[0]
    logits = joint[:x.shape[0], first:first+384].contiguous()
    original_logits = torch.cat([torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
        x[i:i+1].contiguous(), c1.weights.gate.weight) for i in range(x.shape[0])])
    router = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2
    selected_routes = router(logits, c1.weights.gate.bias, c1.weights.gate.bias_vl, mask)
    c1_routes = [router(original_logits[i:i+1].contiguous(), c1.weights.gate.bias, c1.weights.gate.bias_vl,
                        mask[i:i+1].contiguous()) for i in range(x.shape[0])]
    ids_exact = torch.equal(selected_routes[0].cpu(), torch.cat([r[0] for r in c1_routes]).cpu())
    actual, expected = actual.cpu().float(), expected.cpu().float()
    delta = actual-expected
    relative = float(delta.norm()/expected.norm().clamp_min(1e-30))
    log_delta = logits.cpu()-original_logits.cpu()
    log_relative = float(log_delta.norm()/original_logits.cpu().norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    return dict(reference='Accepted C1 one-row BF16 Router + FP8 shared/expert/downstream FFN production path',
                producer='Derived real residual/pre, actual checkpoint FFN norm and epsilon',
                original_input_exact=original_exact, shared_effective_input_exact=shared_exact,
                c1_router_ids_exact=ids_exact, router_logits_relative_l2=log_relative,
                relative_l2=relative, max_abs=float(delta.abs().max()), exact=torch.equal(actual, expected),
                finite=finite, passed=finite and original_exact and shared_exact and ids_exact
                and relative<=.005 and log_relative<=.002,
                tolerance_source='Existing C1 routed FFN component .005 and Router .002 envelopes',
                teacher_forced_acceptance_qualified=False)
