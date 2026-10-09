# SPDX-License-Identifier: Apache-2.0
"""Check rematerialized input preparation against the qualified C1 producer."""


def audit(layer, residual, pre):
    import torch

    def reference(value, mix, weight):
        collapsed = (value.float() * mix.unsqueeze(-1)).sum(1).bfloat16()
        return torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2(
            collapsed.contiguous(), weight, layer.eps)

    def candidate(value, mix, weight):
        collapsed = (value.float() * mix.unsqueeze(-1)).sum(1).bfloat16()
        return torch.ops.custom_op.custom_deepseek_v41_attention_norm_quant_gaudi2_remat(
            collapsed.contiguous(), weight, layer.eps)

    inputs = residual, pre, layer.weights.attn_norm.weight
    functions = [torch.compile(fn, backend="hpu_backend", fullgraph=True, dynamic=False)
                 for fn in (reference, candidate)]
    outputs = [tuple(value.cpu() for value in fn(*inputs)) for fn in functions]
    checks = []
    for name, parent, selected in zip(("normalized", "FP8 operand", "row scale"), *outputs, strict=True):
        a, b = parent, selected
        if parent.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        checks.append(dict(output=name, byte_exact=torch.equal(a, b), shape=list(parent.shape)))
    return dict(reference="Qualified C1 attention norm, BF16 boundary and power-of-two FP8 quantizer",
                passed=all(row["byte_exact"] for row in checks), checks=checks)
