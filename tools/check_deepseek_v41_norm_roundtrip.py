# SPDX-License-Identifier: Apache-2.0
"""Actual mHC -> norm/roundtrip -> QKV -> query producer correctness gate.

This compiled primitive check makes no performance claim. The separate real16
gate measures the production native replay including communication/consumers.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()
    import habana_frameworks.torch.core  # noqa: F401
    import torch
    import torch.nn.functional as F
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v41_qkv import concatenate_static_weights
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    load_native_operators(required=("custom_deepseek_v41_norm_roundtrip_bf16_gaudi2",))
    torch.hpu.set_device(0)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    prefix = f"layers.{args.layer}."
    gamma = shard.tensor(prefix + "attn_norm.weight", "hpu")
    q_gamma = shard.tensor(prefix + "attn.q_norm.weight", "hpu")
    q_weight = shard.dense(prefix + "attn.wq_a.weight", "hpu")
    kv_weight = shard.dense(prefix + "attn.wkv.weight", "hpu")
    qkv_weight = concatenate_static_weights(q_weight, kv_weight)
    query_weight = shard.dense(prefix + "attn.wq_b.weight", "hpu")
    del q_weight, kv_weight
    norm = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2
    quant = torch.ops.custom_op.custom_deepseek_v41_quant_roundtrip_bf16_gaudi2
    compound = torch.ops.custom_op.custom_deepseek_v41_norm_roundtrip_bf16_gaudi2

    def chain(residual, pre, fused):
        collapsed = (residual.float() * pre.unsqueeze(-1)).sum(1).bfloat16().contiguous()
        if fused:
            value, rounded = compound(collapsed, gamma, 1e-20)
        else:
            value = norm(collapsed, gamma, 1e-20)
            rounded = quant(value)
        qkv = F.linear(rounded, qkv_weight)
        query = qkv[:, :1280].contiguous()
        if fused:
            query_norm, query_rounded = compound(query, q_gamma, 1e-20)
        else:
            query_norm = norm(query, q_gamma, 1e-20)
            query_rounded = quant(query_norm)
        output = F.linear(query_rounded, query_weight)
        return value, rounded, qkv, query_norm, query_rounded, output

    compiled = torch.compile(chain, backend="hpu_backend", fullgraph=True, dynamic=False)
    rows = []
    names = ("input_norm", "input_roundtrip", "qkv", "query_norm", "query_roundtrip", "query_projection")
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    for case in range(3):
        fixture = torch.load(args.fixtures / f"c6-{case}.pt", map_location="cpu", weights_only=True)
        values = fixture["groups"][args.layer // 4 - 1]
        residual, pre = (values[name].to("hpu") for name in ("residual", "pre"))
        reference = [x.cpu() for x in compiled(residual, pre, False)]
        candidate = [x.cpu() for x in compiled(residual, pre, True)]
        checks = {}
        for name, expected, actual in zip(names, reference, candidate, strict=True):
            delta = actual.float() - expected.float()
            checks[name] = dict(exact=torch.equal(actual, expected), finite=bool(torch.isfinite(actual).all()),
                                max_abs=float(delta.abs().max()),
                                relative_l2=float(delta.norm() / expected.float().norm().clamp_min(1e-30)))
        rows.append(dict(case=case, checks=checks, context_prefix_tokens=fixture["context_prefix_tokens"]))
        passed = all(check["exact"] and check["finite"] for row in rows for check in row["checks"].values())
        (root / "norm-roundtrip.json").write_text(json.dumps(dict(
            status="passed" if passed and case == 2 else "checking" if passed else "failed",
            cases=rows, reference="Accepted C1 separate norm/group32 roundtrip with actual BF16 MME weights",
            performance_measured=False, teacher_forced_acceptance_qualified=False), indent=2) + "\n")
        if not passed:
            raise AssertionError(checks)


if __name__ == "__main__":
    main()
