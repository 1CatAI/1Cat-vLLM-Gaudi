# SPDX-License-Identifier: Apache-2.0
"""Compare actual C6 projection operands with the accepted one-row operators.

Diagnostic capture only. Never installed in timed serving or performance gates.
"""
import types


def install_probe(attention, rows):
    import torch

    qwidth = attention.weights.wq_a.weight.shape[0]
    kwidth = attention.weights.wkv.weight.shape[0]
    widths = dict(qkv_input=5120, qkv_output=qwidth + kwidth, query_input=1280,
                  query_output=attention.heads * 512, output_input=attention.weights.wo_b.weight.shape[1],
                  output_output=5120)
    for name, width in widths.items():
        attention.register_buffer("projection_oracle_" + name,
                                  torch.empty((rows, width), dtype=torch.bfloat16, device="hpu"), False)
    attention.register_buffer("projection_oracle_quantized",
                              torch.empty((rows, 5120), dtype=torch.float8_e4m3fn, device="hpu"), False)
    attention.register_buffer("projection_oracle_scale", torch.empty((rows, 1), device="hpu"), False)
    original = dict(qkv=attention._project_qkv_input, query=attention.project_query,
                    output=attention.project_output_consumer)
    attention._projection_oracle_original = original
    attention._projection_oracle_prequant = False

    def qkv(self, value, **kwargs):
        self.projection_oracle_qkv_input.copy_(value)
        prequant = kwargs.get("prequant")
        if prequant is not None:
            self.projection_oracle_quantized.copy_(prequant[0])
            self.projection_oracle_scale.copy_(prequant[1].reshape(rows, 1))
            self._projection_oracle_prequant = True
        result = original["qkv"](value, **kwargs)
        self.projection_oracle_qkv_output.copy_(torch.cat(result, dim=-1))
        return result

    def query(self, value, positions, **kwargs):
        self.projection_oracle_query_input.copy_(value)
        result = original["query"](value, positions, **kwargs)
        self.projection_oracle_query_output.copy_(result.flatten(1))
        return result

    def output(self, value):
        self.projection_oracle_output_input.copy_(value)
        result = original["output"](value)
        self.projection_oracle_output_output.copy_(result)
        return result

    attention._project_qkv_input = types.MethodType(qkv, attention)
    attention.project_query = types.MethodType(query, attention)
    attention.project_output_consumer = types.MethodType(output, attention)


def audit(attention, positions):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import quantize_activation

    originals = attention._projection_oracle_original
    # Snapshot captured outputs before running the unmodified one-row methods.
    actual = {name: getattr(attention, "projection_oracle_" + name + "_output").cpu()
              for name in ("qkv", "query", "output")}
    inputs = {name: getattr(attention, "projection_oracle_" + name + "_input").clone()
              for name in ("qkv", "query", "output")}
    expected = dict(qkv=[], query=[], output=[])
    rows = positions.numel()
    for i in range(rows):
        prequant = None
        if attention._projection_oracle_prequant:
            prequant = (attention.projection_oracle_quantized[i:i + 1].contiguous(),
                        attention.projection_oracle_scale[i:i + 1].contiguous())
        expected["qkv"].append(torch.cat(originals["qkv"](inputs["qkv"][i:i + 1].contiguous(),
                                                        prequant=prequant), dim=-1).cpu())
        query_input = inputs["query"][i:i + 1].contiguous()
        weight = attention.weights.wq_b
        query_input = quantize_activation(query_input) if hasattr(weight, "scale") else query_input
        expected["query"].append(torch.ops.custom_op.custom_deepseek_v41_q_projection_rope_gaudi2(
            query_input, weight.weight, weight.channel_scale, positions[i:i + 1].contiguous(),
            attention._rotary_native_table()).flatten(1).cpu())
        expected["output"].append(originals["output"](inputs["output"][i:i + 1].contiguous()).cpu())
    results = {}
    for name in expected:
        reference = torch.cat(expected[name])
        delta = actual[name].float() - reference.float()
        relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
        close = torch.allclose(actual[name].float(), reference.float(), atol=0.5, rtol=0.008)
        results[name] = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                             relative_l2=relative, bitwise_equal=torch.equal(actual[name], reference),
                             finite=bool(torch.isfinite(actual[name]).all()),
                             passed=bool(close and relative <= 0.002))
    return dict(reference="Same real operands and immutable FP8 weights, accepted C1 one-row projection methods",
                tolerance_source="test_woa_fp8_device.py: atol0.5/rtol0.008 AND relativeL2<=0.002",
                prequant_preserved=attention._projection_oracle_prequant, projections=results,
                passed=all(row["passed"] for row in results.values()), full_service_quality_passed=False)


def install_shared_probe(moe, rows):
    import torch

    for name in ("input", "output"):
        moe.register_buffer("shared_oracle_" + name,
                            torch.empty((rows, 5120), dtype=torch.bfloat16, device="hpu"), False)
    original = moe.shared_expert
    moe._shared_oracle_original = original

    def shared(self, value, **kwargs):
        self.shared_oracle_input.copy_(value)
        result = original(value, **kwargs)
        self.shared_oracle_output.copy_(result)
        return result

    moe.shared_expert = types.MethodType(shared, moe)


def audit_shared(moe):
    import torch

    actual = moe.shared_oracle_output.cpu()
    value = moe.shared_oracle_input.clone()
    reference = torch.cat([moe._shared_oracle_original(row.unsqueeze(0)).cpu() for row in value])
    delta = actual.float() - reference.float()
    relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
    result = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                  relative_l2=relative, bitwise_equal=torch.equal(actual, reference),
                  finite=bool(torch.isfinite(actual).all()),
                  passed=bool(torch.allclose(actual.float(), reference.float(), atol=0.5, rtol=0.008)
                              and relative <= 0.002))
    reference_name = ("Accepted C1 FP8 shared expert, identical real input and immutable sidecar weights"
                      if moe.shared_gate_up_channel is not None else
                      "Same actual shared-expert input and immutable joined BF16 weight; C1 one-row shared path")
    return dict(reference=reference_name,
                tolerance_source="Existing native projection check: atol0.5/rtol0.008 AND relativeL2<=0.002",
                projections=dict(shared=result), passed=result["passed"], full_service_quality_passed=False)


def install_publish_probe(attention, rows):
    import torch

    widths = dict(q_input=1280, kv_input=512, query=attention.heads * 512, kv=512, normalized=1280)
    for name, width in widths.items():
        attention.register_buffer("publish_oracle_" + name,
                                  torch.empty((rows, width), dtype=torch.bfloat16, device="hpu"), False)
    original = attention.project_qkv_publish

    def publish(self, q_input, kv_input, positions, decoded):
        self.publish_oracle_q_input.copy_(q_input)
        self.publish_oracle_kv_input.copy_(kv_input)
        result = original(q_input, kv_input, positions, decoded)
        self.publish_oracle_query.copy_(result[0])
        self.publish_oracle_kv.copy_(result[1])
        self.publish_oracle_normalized.copy_(result[3])
        return result

    attention.project_qkv_publish = types.MethodType(publish, attention)


def audit_publish(attention, positions):
    import torch
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa

    q_input, kv_input = attention.publish_oracle_q_input.clone(), attention.publish_oracle_kv_input.clone()
    actual = {name: getattr(attention, "publish_oracle_" + name).cpu()
              for name in ("query", "kv", "normalized")}
    gold = dict(query=[], kv=[], normalized=[])
    for i in range(positions.numel()):
        q, kv, position = q_input[i:i + 1].contiguous(), kv_input[i:i + 1].contiguous(), positions[i:i + 1].contiguous()
        gold["query"].append(attention.project_query_input(q, position).flatten(1).cpu())
        gold["kv"].append(attention.project_kv(kv, position, decode=True).cpu())
        gold["normalized"].append(torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(
            q, attention.weights.q_norm.weight, attention.eps).cpu())
    results = {}
    for name, values in gold.items():
        reference = torch.cat(values)
        delta = actual[name].float() - reference.float()
        relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
        results[name] = dict(max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                             relative_l2=relative, finite=bool(torch.isfinite(actual[name]).all()),
                             bitwise_equal=torch.equal(actual[name], reference),
                             passed=bool(torch.allclose(actual[name].float(), reference.float(), atol=0.5, rtol=0.008)
                                         and relative <= 0.002))
    canonical = pack_swa(torch.cat(gold["kv"]).to("hpu")).cpu()
    published = attention.swa.index_select(0, positions.remainder(256).long()).cpu()
    cache_exact = torch.equal(canonical, published)
    return dict(reference="Accepted C1 Q norm/projection/RoPE and KV norm/RoPE on identical real operands",
                tolerance_source="Projection atol0.5/rtol0.008 AND relativeL2<=0.002; canonical SWA exact",
                projections=results, canonical_swa_exact=cache_exact,
                passed=cache_exact and all(row["passed"] for row in results.values()),
                full_service_quality_passed=False)
