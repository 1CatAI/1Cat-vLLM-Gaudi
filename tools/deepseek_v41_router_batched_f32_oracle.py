# SPDX-License-Identifier: Apache-2.0
"""Batched dense Router FFN against the accepted one-row C1 numerical path, outside timing."""
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
    actual, expected = actual.cpu().float(), expected.cpu().float()
    delta = actual-expected
    relative = float(delta.norm()/expected.norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    return dict(reference='Accepted one-row C1 BF16-valued Router/FP8 FFN production path',
                producer='Derived real residual/pre, actual checkpoint FFN norm and epsilon',
                relative_l2=relative, max_abs=float(delta.abs().max()), exact=torch.equal(actual, expected),
                finite=finite, passed=finite and relative<=.005,
                tolerance_source='Existing C1 routed FFN component .005 envelope',
                teacher_forced_acceptance_qualified=False)
