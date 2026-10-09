# SPDX-License-Identifier: Apache-2.0
"""Private static-FP8 arithmetic on real producer operands, not model quality."""


def audit(layer, residual, previous_pre):
    import torch

    from vllm_gaudi import envs
    from vllm_gaudi.ops.deepseek_v41_hw_dense import project
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation, rms_norm
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import decode_e4m3fn, decode_gaudi2

    attention = layer.attention

    def producer(x, pre):
        collapsed = (x.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        value_raw = rms_norm(collapsed, layer.weights.attn_norm.weight, layer.eps)
        value = quantize_activation(value_raw)
        q_width = attention.weights.q_norm.weight.numel()
        q = torch.nn.functional.linear(value, attention._fused_qkv_weight)[:, :q_width].contiguous()
        query_raw = rms_norm(q, attention.weights.q_norm.weight, layer.eps)
        query = quantize_activation(query_raw)
        return ((value_raw, query_raw) if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT
                else (value, query))

    derive = torch.compile(producer, backend="hpu_backend", fullgraph=True, dynamic=False)
    operands = derive(residual, previous_pre)
    query = attention.weights.wq_b
    specs = (
        ("qkv", operands[0], attention.dspark_hw_qkv,
         attention.dspark_hw_qkv_input_scale, attention.dspark_hw_qkv_weight_scale,
         attention._fused_qkv_weight),
        ("query", operands[1], query.dspark_hw_weight, query.dspark_hw_input_scale,
         query.dspark_hw_weight_scale, query.weight),
    )
    evaluate = torch.compile(project, backend="hpu_backend", fullgraph=True, dynamic=False)
    checks = []
    for name, value, weight, sx, sw, original in specs:
        actual = evaluate(value, weight, sx, sw).cpu().float()
        x = value.cpu().float()
        x = (quantize_activation(x).float()
             if envs.VLLM_HPU_DSV41_DSPARK_HW_DENSE_FUSED_QUANT else x)
        q = (x / sx).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        decoded_x = torch.from_numpy(decode_e4m3fn(q) * sx)
        decoded_w = torch.from_numpy(decode_gaudi2(weight.cpu().view(torch.uint8).numpy()) * sw)
        reference = (decoded_x @ decoded_w.t()).bfloat16().float()
        error = actual - reference
        relative = float(error.norm() / reference.norm().clamp_min(1e-30))
        official_operand_reference = (x @ original.cpu().float().t()).bfloat16().float()
        official_error = actual - official_operand_reference
        checks.append(dict(projection=name, max_abs=float(error.abs().max()), relative_l2=relative,
                           encoded_operands_exact=torch.equal(actual, reference),
                           original_bf16_max_abs=float(official_error.abs().max()),
                           original_bf16_relative_l2=float(
                               official_error.norm() / official_operand_reference.norm().clamp_min(1e-30)),
                           input_scale=sx, weight_scale=sw,
                           passed=bool(torch.isfinite(actual).all()) and relative <= .005))
    return dict(checks=checks, passed=all(row["passed"] for row in checks),
                reference="Explicit independent CPU E4M3 operands; real norm/quantization and checkpoint weight",
                teacher_forced_acceptance_qualified=False, full_service_quality_passed=False)
