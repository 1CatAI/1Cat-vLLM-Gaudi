#!/usr/bin/env python3
"""Prepare bounded V4.1 grouped-prefill recipes before model residency.

The production model leaves too little free HBM for Synapse to compile a new
large-M expert body beside the 1M KV pool.  This tool owns only one HPU and
materializes the finite expert body shapes used by a scheduler bucket.  The
resulting persistent recipe cache can then be copied into each serving rank's
cache and replayed without compiling during a request.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--groups", type=int, nargs="+", required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens <= 0 or any(group <= 0 or group > 8 for group in args.groups):
        parser.error("tokens must be positive and groups must be in [1, 8]")

    # Native-library paths must be fixed before importing torch/HPU modules.
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment

    prepare_environment()
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut
    from vllm_gaudi.ops.deepseek_v41_grouped_prefill import compiled_project

    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = max(32, len(args.groups) + 4)
    device = torch.device("hpu")
    experts, rows, hidden = 384, 128, 5120

    # These are the sole large resident allocations during preparation.  The
    # data values do not enter the recipe identity; empty tensors avoid an
    # unnecessary multi-GiB initialization while exercising the real kernels.
    # Runtime N256 v3 shapes, after loading the TP2-prepared sidecar.  W13 is
    # [2304, 5120] (fused gate/up) and W2 is [5120, 1152].
    q13 = torch.empty((experts, 9, 327680), dtype=torch.int16, device=device)
    q2 = torch.empty((experts, 20, 73728), dtype=torch.int16, device=device)
    s13 = torch.empty((experts, 9, 40960), dtype=torch.int16, device=device)
    s2 = torch.empty((experts, 20, 9216), dtype=torch.int16, device=device)
    lookup = mxfp4_bf16_lut(device)
    prepared = []
    # Model weights are ordinary tensors, but request-local activations and
    # descriptors are created below execute_model's inference-mode boundary.
    # Dynamo guards both properties, so the offline recipe must reproduce the
    # same mixed contract as the live server.
    with torch.inference_mode():
        for groups in sorted(set(args.groups), reverse=True):
            selected = torch.empty((groups, rows, hidden), dtype=torch.bfloat16, device=device)
            route = torch.empty((groups, rows), dtype=torch.float32, device=device)
            expert_ids = torch.zeros((1, groups), dtype=torch.int32, device=device)
            signature = (groups, rows, experts, True)
            output = compiled_project(signature)(selected, route, expert_ids, q13, q2, s13, s2, lookup, True)
            torch.hpu.synchronize()
            if output.shape != (groups * rows, hidden) or output.dtype != torch.bfloat16:
                raise RuntimeError(f"Unexpected grouped-prefill output: {output.shape} {output.dtype}")
            prepared.append({"signature": list(signature), "output_shape": list(output.shape)})
            del output, selected, route, expert_ids

    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({"prepared": prepared}, indent=2) + "\n")


if __name__ == "__main__":
    main()
