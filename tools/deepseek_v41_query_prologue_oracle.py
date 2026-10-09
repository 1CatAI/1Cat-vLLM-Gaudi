# SPDX-License-Identifier: Apache-2.0
"""Actual C6 query operands against the accepted C1 fused producer."""
import types


def install(attention, rows):
    import torch

    attention.register_buffer("q_prologue_oracle_input",
                              torch.empty((rows, 1280), dtype=torch.bfloat16, device="hpu"), False)
    attention.register_buffer("q_prologue_oracle_output",
                              torch.empty((rows, attention.heads, 512), dtype=torch.bfloat16, device="hpu"), False)
    attention._q_prologue_oracle_norm = False
    for name, needs_norm in (("project_query", False), ("project_query_input", True)):
        original = getattr(attention, name)

        def capture(self, value, positions, *args, _original=original, _norm=needs_norm, **kwargs):
            self.q_prologue_oracle_input.copy_(value)
            result = _original(value, positions, *args, **kwargs)
            self.q_prologue_oracle_output.copy_(result.reshape_as(self.q_prologue_oracle_output))
            self._q_prologue_oracle_norm = _norm
            return result

        setattr(attention, name, types.MethodType(capture, attention))


def audit(attention, positions):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation

    value = attention.q_prologue_oracle_input.clone()
    actual = attention.q_prologue_oracle_output.cpu()
    weight = attention.weights.wq_b
    pieces = []
    for row in range(positions.numel()):
        x, pos = value[row:row + 1].contiguous(), positions[row:row + 1].contiguous()
        if attention._q_prologue_oracle_norm:
            out = torch.ops.custom_op.custom_deepseek_v41_q_norm_projection_rope_gaudi2(
                x, attention.weights.q_norm.weight, weight.weight, weight.channel_scale, pos,
                attention._rotary_native_table(), attention.eps)
        else:
            x = quantize_activation(x) if hasattr(weight, "scale") else x
            out = torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2(
                x, weight.weight, weight.channel_scale, pos, attention._rotary_native_table())
        pieces.append(out.reshape(1, attention.heads, 512).cpu())
    reference = torch.cat(pieces)
    delta = actual.float() - reference.float()
    relative = float(delta.norm() / reference.float().norm().clamp_min(1e-30))
    finite = bool(torch.isfinite(actual).all())
    result = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                  relative_l2=relative, bitwise_equal=torch.equal(actual, reference), finite=finite,
                  passed=finite and bool(torch.allclose(actual.float(), reference.float(), atol=.5, rtol=.008))
                  and relative <= .002)
    return dict(reference="Accepted C1 query fused producer on identical real operands/FP8 weights",
                includes_norm=attention._q_prologue_oracle_norm, projections=dict(query=result),
                passed=result["passed"], full_service_quality_passed=False)
