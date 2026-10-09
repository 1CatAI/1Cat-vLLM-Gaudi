# SPDX-License-Identifier: Apache-2.0
"""Check fused epilogues against the identical joint FP8 producer, outside timing."""


def audit(layer, residual, pre):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    moe = layer.moe
    # A derived real checkpoint input checks arithmetic/layout equivalence.
    # This is not a full-target quality gate; common-prefix alpha remains
    # mandatory before the candidate can enter the serving ledger.
    collapse = (residual.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
    value = rms_norm(collapse, layer.weights.ffn_norm.weight)
    q, sx = torch.ops.custom_op.custom_deepseek_v41_dense_quant_gaudi2(value.contiguous())
    product = torch.ops.hpu.fp8_gemm_v2(q, False, moe.router_shared_weight, True, None,
                                      torch.float32, None, None, None, False)
    rows, first = value.shape[0], moe.router_shared_columns
    mask = torch.zeros(rows, dtype=torch.bool, device=value.device)
    logits = ((product[:, first:first + 384] * moe.router_shared_channel) * sx).contiguous()
    original = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2(
        logits, moe.weights.gate.bias, moe.weights.gate.bias_vl, mask)
    fused = torch.ops.custom_op.custom_deepseek_v41_router_shared_scaled_gaudi2(
        product, moe.weights.gate.bias, moe.weights.gate.bias_vl, mask, moe.router_shared_channel, sx)
    ids = torch.zeros(1, rows, dtype=torch.int32, device=value.device)
    routing = torch.ones(1, rows, dtype=torch.float32, device=value.device)
    shared_original = torch.ops.custom_op.custom_deepseek_v41_shared_silu_quant_gaudi2(
        product[:, :first].contiguous().reshape(rows, 1, first), ids, sx, moe.shared_gate_up_channel, routing)
    shared_fused = torch.ops.custom_op.custom_deepseek_v41_shared_silu_full_product_gaudi2(
        product.reshape(rows, 1, -1), ids, sx, moe.shared_gate_up_channel, routing)
    expected = [v.cpu() for v in (*original, *shared_original)]
    actual = [v.cpu() for v in (*fused, *shared_fused)]
    exact = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(expected, actual, strict=True)]
    return dict(passed=all(exact), router_ids_exact=exact[0], router_weights_exact=exact[1],
                shared_fp8_exact=exact[2], shared_scale_exact=exact[3],
                full_target_acceptance_qualified=False, source='derived real entry residual and checkpoint FFN norm')
