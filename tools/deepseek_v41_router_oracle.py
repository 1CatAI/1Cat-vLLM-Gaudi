# SPDX-License-Identifier: Apache-2.0
"""Actual C6 router operands against the accepted C1 BF16/MME projection."""
import types
import torch


def install(moe, rows):
    moe.register_buffer("router_oracle_input", torch.empty((rows, 5120), dtype=torch.float32, device="hpu"), False)
    moe.register_buffer("router_oracle_logits", torch.empty((rows, 384), dtype=torch.float32, device="hpu"), False)
    original = moe._router_logits

    def capture(self, value, prefill_router_tokens=0):
        result = original(value, prefill_router_tokens)
        self.router_oracle_input.copy_(value)
        self.router_oracle_logits.copy_(result)
        return result

    moe._router_logits = types.MethodType(capture, moe)


def audit(moe):
    x = moe.router_oracle_input
    weight = moe.weights.gate.weight
    exact_inputs = torch.equal(x.cpu(), x.bfloat16().float().cpu())
    lossless_weight = torch.equal(weight.float().cpu(), weight.bfloat16().float().cpu())
    reference = torch.cat(tuple(torch.ops.custom_op.custom_deepseek_v41_bf16_linear_f32_gaudi2(
        x[i:i + 1].bfloat16().contiguous(), weight.bfloat16().contiguous()) for i in range(x.shape[0])))
    actual, expected = moe.router_oracle_logits.cpu(), reference.cpu()
    difference = actual - expected
    relative = float(difference.norm() / expected.norm().clamp_min(1e-30))
    mask = torch.zeros((x.shape[0],), dtype=torch.bool, device=x.device)
    operation = torch.ops.custom_op.custom_deepseek_v41_router_logits_top6_gaudi2
    ids, routes = operation(moe.router_oracle_logits, moe.weights.gate.bias, moe.weights.gate.bias_vl, mask)
    c1 = [operation(reference[i:i + 1].contiguous(), moe.weights.gate.bias, moe.weights.gate.bias_vl,
                    mask[i:i + 1].contiguous()) for i in range(x.shape[0])]
    ids_exact = torch.equal(ids.cpu(), torch.cat([row[0] for row in c1]).cpu())
    expected_routes = torch.cat([row[1] for row in c1]).cpu()
    route_relative = float((routes.cpu() - expected_routes).norm() / expected_routes.norm().clamp_min(1e-30))
    gate = dict(max_abs=float(difference.abs().max()), relative_l2=relative,
                passed=relative <= .002 and torch.isfinite(actual).all().item())
    return dict(reference="Same actual operands; six accepted C1 BF16-weight/F32-output MME router projections",
                input_bf16_exact=exact_inputs, checkpoint_weight_bf16_lossless=lossless_weight,
                router_ids_exact=ids_exact, routing_relative_l2=route_relative,
                projections=dict(gate=gate),
                passed=exact_inputs and lossless_weight and gate["passed"] and ids_exact and route_relative <= .002,
                full_service_quality_passed=False)
