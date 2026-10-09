# SPDX-License-Identifier: Apache-2.0
"""Audit streaming C1 exponent PV using real producers and canonical decoded KV."""
from types import SimpleNamespace


def audit(layer, residual, previous_pre, positions):
    import torch
    from deepseek_v41_query_norm_oracle import real_query_input
    from deepseek_v41_attention_oracle import actual_operands, audit as audit_attention
    from vllm_gaudi.ops.deepseek_v41_math import rms_norm

    attention = layer.attention
    qr = rms_norm(real_query_input(layer, residual, previous_pre),
                  attention.weights.q_norm.weight, attention.eps).contiguous()
    query = attention.project_query(qr, positions, decode=True).contiguous()
    selected = attention.selection.indices[:query.shape[0]].clone().contiguous()
    lengths = torch.full((query.shape[0],), 640, dtype=torch.int32, device=query.device)
    pos = positions.to(torch.int32).contiguous()
    phase = attention._rotary_native_table()
    args = (query, attention.swa, attention.cache.main, selected, pos,
            attention.shared.block_table, attention.weights.attn_sink, attention.scale,
            lengths, attention.ratio)
    parent = torch.ops.custom_op.custom_deepseek_v41_logical_scale_cache_gaudi2(*args, True)
    parent = attention._rope(parent, pos, inverse=True)
    published, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_stream_exp_publish_mla_gaudi2(
        *args, phase)
    reused = torch.ops.custom_op.custom_deepseek_v41_main_stream_exp_reuse_mla_gaudi2(
        query, attention.swa, rows, mask, pos, attention.weights.attn_sink, attention.scale, lengths, phase)
    torch.hpu.synchronize()
    proxy = SimpleNamespace(oracle_query=query, oracle_output=parent, oracle_selected=selected,
                            shared=attention.shared, ratio=attention.ratio, cache=attention.cache,
                            swa=attention.swa, weights=attention.weights, scale=attention.scale)
    reference = actual_operands(proxy, positions)
    table = phase.cpu()[pos.cpu().long()].double()

    def rotate_reference(value):
        # The canonical C1 boundary rounds PV before inverse RoPE. Keep that
        # boundary; rotate in FP64 against the production FP32 phase values.
        value = value.bfloat16().double()
        pairs = value[..., -64:].unflatten(-1, (32, 2))
        real, imag = pairs[..., 0], pairs[..., 1]
        cosine, sine = table[:, None, :32], table[:, None, 32:]
        tail = torch.stack((real * cosine + imag * sine, imag * cosine - real * sine), -1).flatten(-2)
        return torch.cat((value[..., :-64], tail), -1)

    results = {}
    for name, output in (("publish", published), ("reuse", reused)):
        result = audit_attention(reference, dict(reference, output=output.cpu()),
                                 reference_transform=rotate_reference)
        row = result["candidate"]
        result["passed"] = (all(result["inputs_exact"].values()) and row["finite"]
                            and row["official_tolerance_close"] and row["relative_l2"] <= .002)
        result["oracle"] += "; canonical BF16 boundary and FP64 inverse RoPE using production phases"
        results[name] = result
    return dict(passed=all(row["passed"] for row in results.values()), arms=results,
                timed_graph_instrumented=False, actual_producer_shape=list(query.shape), ratio=attention.ratio,
                teacher_forced_acceptance_qualified=False)
