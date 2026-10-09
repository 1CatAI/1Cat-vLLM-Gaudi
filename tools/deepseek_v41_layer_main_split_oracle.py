# SPDX-License-Identifier: Apache-2.0
"""Audit split PV over real producers/current KV outside the timed graph."""
from types import SimpleNamespace


def audit(layer, residual, previous_pre, positions, *, coherent=False, single_bank=False, flat_qk=False,
          flat_qk_direct=False):
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
    args = (query, attention.swa, attention.cache.main, selected, pos,
            attention.shared.block_table, attention.weights.attn_sink, attention.scale,
            lengths, attention.ratio)
    parent = torch.ops.custom_op.custom_deepseek_v41_logical_scale_cache_gaudi2(*args, True)
    if single_bank:
        published, rows, mask = torch.ops.custom_op.custom_deepseek_v41_main_single_bank_publish_mla_gaudi2(*args)
        reused = torch.ops.custom_op.custom_deepseek_v41_main_single_bank_reuse_mla_gaudi2(
            query, attention.swa, rows, mask, pos, attention.weights.attn_sink, attention.scale, lengths)
    else:
        publish = (torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_direct_publish_mla_gaudi2 if flat_qk_direct else
                   torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_publish_mla_gaudi2 if flat_qk else
                   torch.ops.custom_op.custom_deepseek_v41_main_split_publish_mla_gaudi2)
        published, rows, mask, values = publish(*args)
        reuse = (torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_direct_reuse_mla_gaudi2 if flat_qk_direct else
                 torch.ops.custom_op.custom_deepseek_v41_main_qk_flat_reuse_mla_gaudi2 if flat_qk else
                 torch.ops.custom_op.custom_deepseek_v41_coherent_swa_mla_gaudi2 if coherent else
                 torch.ops.custom_op.custom_deepseek_v41_main_split_reuse_mla_gaudi2)
        reused = reuse(
            query, attention.swa, rows, mask, pos, attention.weights.attn_sink, attention.scale, lengths, values)
    torch.hpu.synchronize()
    proxy = SimpleNamespace(oracle_query=query, oracle_output=parent, oracle_selected=selected,
                            shared=attention.shared, ratio=attention.ratio, cache=attention.cache,
                            swa=attention.swa, weights=attention.weights, scale=attention.scale)
    reference = actual_operands(proxy, positions)
    results = {}
    for name, output in (("publish", published), ("reuse", reused)):
        result = audit_attention(reference, dict(reference, output=output.cpu()))
        row = result["candidate"]
        result["passed"] = (all(result["inputs_exact"].values()) and row["finite"]
                            and row["official_tolerance_close"] and row["relative_l2"] <= .002)
        results[name] = result
    return dict(passed=all(row["passed"] for row in results.values()), arms=results,
                timed_graph_instrumented=False, actual_producer_shape=list(query.shape), ratio=attention.ratio)
