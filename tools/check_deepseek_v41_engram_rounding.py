# SPDX-License-Identifier: Apache-2.0
"""Explain saved Engram BF16 discrepancies using the exact producer and FP64."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.models.deepseek_v41_program import _weight_tree, load_weight_tree, linear
    from vllm_gaudi.ops.deepseek_v41_engram_fp8 import EngramFP8Sidecar
    from vllm_gaudi.ops.deepseek_v41_math import engram_update
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    torch.set_num_threads(4)
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    shard = PreparedV41Shard(args.prepared, 0, 0)
    specs = {name: spec for name, spec in shard.specs.items()
             if name.startswith(("layers.1.engram.", "layers.14.engram."))}
    tree = _weight_tree(specs)
    load_weight_tree(shard, tree, "hpu", specs,
                     engram_sidecar=EngramFP8Sidecar(args.prepared / "sidecars/engram_fp8", shard))
    eps = json.loads((args.prepared / "config.json").read_text())["text_config"]["rms_norm_eps"]
    root = Path(os.environ["DSV41_RUN_EVIDENCE"])
    report = dict(status="running", cases=[], scope="saved first C8192 chain, both Engram layers")
    with torch.inference_mode():
        torch.manual_seed(9927)
        residual = torch.randn(8192, 4, 5120).bfloat16().to("hpu")
        rows = []
        for rank in range(4):
            torch.manual_seed(5550 + rank)
            rows.append(torch.randn(8192, 6, 256).bfloat16())
        rows = torch.cat(rows, 1).flatten(1).to("hpu")
        active = torch.arange(8192, device="hpu") % 7 != 2
        for layer in (1, 14):
            if layer == 14:
                for _ in range(3):
                    residual.add_(.001)
            w = tree.layers.get_submodule(str(layer)).engram
            kv = linear(rows, w.wkv)
            reproduced = engram_update(residual, kv, w.q_weight, w.k_weight, active, eps).cpu()
            saved = torch.load(args.reference / f"engram-rank0-0-{layer}.pt", weights_only=True)
            actual, expected = saved["actual"][0], saved["expected"][0]
            assert torch.equal(reproduced, expected), "Producer reconstruction differs from the archived chain"
            differences = (actual != expected).nonzero()
            pairs, inverse = torch.unique(differences[:, :2], dim=0, return_inverse=True)
            host_h, host_kv = residual.cpu(), kv.cpu()
            h = host_h[pairs[:, 0], pairs[:, 1]].double()
            k = host_kv[:, :20480].reshape(-1, 4, 5120)[pairs[:, 0], pairs[:, 1]].double()
            qk = (w.q_weight.cpu().double() * w.k_weight.cpu().double()).expand(4, 5120)[pairs[:, 1]]
            products = h * qk * k
            scale = (h.square().mean(-1) + eps).rsqrt() * (k.square().mean(-1) + eps).rsqrt() / 5120**.5
            dot = products.sum(-1) * scale
            # Standard sequential FP32 gamma_n bound, enlarged for both
            # products, the two RMS reductions and their scale operations.
            # This bound is derived from width and precision, not fitted to
            # the observed candidate/reference error.
            unit = torch.finfo(torch.float32).eps / 2
            gamma = (5120 + 16) * unit / (1 - (5120 + 16) * unit)
            dot_bound = 5 * gamma * (products.abs().sum(-1) * scale + dot.abs())

            def gate(value):
                return torch.sigmoid(torch.copysign(value.abs().clamp_min(1e-6).sqrt(), value))

            t, head, column = differences.T
            base = host_h[t, head, column].double()
            value = host_kv[t, 20480 + column].double()
            center = base + gate(dot)[inverse] * value
            left = base + gate(dot - dot_bound)[inverse] * value
            right = base + gate(dot + dot_bound)[inverse] * value
            low, high = torch.minimum(left, right), torch.maximum(left, right)
            a, b = actual[t, head, column].double(), expected[t, head, column].double()
            for output in (a, b):
                exponent = torch.frexp(output.abs())[1]
                half_ulp = torch.pow(2., (exponent - 9).clamp(min=-134)).double()
                assert ((output >= low - half_ulp) & (output <= high + half_ulp)).all()
            row = dict(layer=layer, changed_lanes=len(differences), total_lanes=actual.numel(),
                       producer_reproduced_bitexact=True, fp64_error_bound_passed=True,
                       relative_l2=float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(expected.float())),
                       max_distance_to_two_outputs_midpoint=float((center - (a + b) / 2).abs().max()),
                       fp32_gamma=gamma, checked_pairs=len(pairs))
            report["cases"].append(row)
            (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "passed"
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
