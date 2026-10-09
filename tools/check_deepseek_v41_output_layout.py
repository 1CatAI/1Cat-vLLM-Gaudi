# SPDX-License-Identifier: Apache-2.0
"""Check target/draft wo_a against the checkpoint layout on one leased HPU."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=manifest["tensor_parallel_size"],
                        pipeline_parallel_size=manifest["pipeline_parallel_size"])
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.models import deepseek_v41_program as program
    from vllm_gaudi.ops.deepseek_v41_attention import CSA2Attention
    from vllm_gaudi.ops.deepseek_v41_paged_attention import PagedCSA2Attention
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard

    torch.hpu.set_device(0)
    shard = PreparedV41Shard(args.prepared, 0, 0)
    config = json.loads((args.prepared / "config.json").read_text())["text_config"]
    tp_size = shard.tensor_parallel_size
    groups, heads = config["o_groups"] // tp_size, config["num_attention_heads"] // tp_size
    inner = heads // groups * config["head_dim"]
    results = []
    for name, prepared, attention in (("layers.0.attn.wo_a.weight", True, PagedCSA2Attention),
                                      ("mtp.0.attn.wo_a.weight", False, CSA2Attention)):
        original = shard.dense(name, "hpu")
        specs = {name: shard.specs[name]}
        tree = program._weight_tree(specs)
        proxy = SimpleNamespace(dense=lambda *args, original=original: original.clone(),
                                check_identity=shard.check_identity)
        program.load_weight_tree(proxy, tree, "hpu", specs)
        owner = SimpleNamespace(weights=tree.get_submodule(name.rpartition(".wo_a.")[0]),
                                woa_fp8=False,
                                woa_output_roundtrip=False,
                                prepared_output=prepared,
                                output_gemm_layout=False,
                                heads=heads,
                                groups=groups)
        attention.prepare_output_weight(owner)
        for rows in (1, 6):
            value = torch.randn(rows,
                                groups,
                                inner,
                                generator=torch.Generator().manual_seed(41 + rows),
                                dtype=torch.bfloat16).to("hpu")
            reference = torch.einsum("tgd,grd->tgr", value, original.reshape(groups, 1024, inner)).flatten(1)
            actual = attention.project_output(owner, value)
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)
            results.append(dict(weight=name, rows=rows, groups=groups, status="passed_exact"))
        del original, tree, owner, reference, actual, value
    result = dict(status="passed",
                  tp_size=tp_size,
                  local_attention_heads=heads,
                  results=results,
                  semantic_qualification=False,
                  performance_claim=False)
    (Path(os.environ["DSV41_RUN_EVIDENCE"]) / "OUTPUT_LAYOUT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
