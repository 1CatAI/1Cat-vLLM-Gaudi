# SPDX-License-Identifier: Apache-2.0
"""Audit actual BF16-exponent hardware without instrumenting the timed graph."""
from pathlib import Path
from types import SimpleNamespace


def audit(attention, rank, case):
    import torch
    from deepseek_v41_attention_oracle import actual_operands, audit as audit_attention

    root = Path("/opt/optane/dsv41-builds/dsv41-tp4-dspark-serving/request-c6-mla-stacked-pv-oracle-208")
    original = torch.load(root / f"attention-operands-rank{rank}-case{case}.pt",
                          map_location="cpu", weights_only=True)[0]
    if original["ratio"] != attention.ratio:
        raise ValueError("Saved real query belongs to a different compression ratio")
    query = original["query"].to("hpu").contiguous()
    selected = original["selected"].to("hpu").contiguous()
    positions = original["positions"].to("hpu", dtype=torch.int32).contiguous()
    lengths = torch.full((query.shape[0],), 640, dtype=torch.int32, device="hpu")
    operands = (query, attention.swa, attention.cache.main, selected, positions, attention.shared.block_table,
                attention.weights.attn_sink, attention.scale, lengths, attention.ratio, True)
    parent = torch.ops.custom_op.custom_deepseek_v41_logical_scale_cache_gaudi2(*operands)
    candidate = torch.ops.custom_op.custom_deepseek_v41_logical_scale_exp_pv_gaudi2(*operands)
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
    return result
