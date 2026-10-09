# SPDX-License-Identifier: Apache-2.0
"""Fresh real QKV producer and current KV, independently audited in FP64."""
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
    operands = (query, attention.swa, attention.cache.main, selected, positions.to(torch.int32).contiguous(),
                attention.shared.block_table, attention.weights.attn_sink, attention.scale,
                lengths, attention.ratio, True)
    parent = torch.ops.custom_op.custom_deepseek_v41_logical_scale_cache_gaudi2(*operands)
    candidate = torch.ops.custom_op.custom_deepseek_v41_logical_mla_compact_stream_mme_gaudi2(*operands)
    torch.hpu.synchronize()
    proxy = SimpleNamespace(oracle_query=query, oracle_output=parent, oracle_selected=selected,
                            shared=attention.shared, ratio=attention.ratio, cache=attention.cache,
                            swa=attention.swa, weights=attention.weights, scale=attention.scale)
    left = actual_operands(proxy, positions)
    right = dict(left, output=candidate.cpu())
    result = audit_attention(left, right)
    row = result["candidate"]
    result["passed"] = (all(result["inputs_exact"].values()) and row["finite"]
                        and row["official_tolerance_close"] and row["relative_l2"] <= .002)
    result["timed_graph_instrumented"] = False
    result["actual_producer_shape"] = list(query.shape)
    result["ratio"] = attention.ratio
    return result
