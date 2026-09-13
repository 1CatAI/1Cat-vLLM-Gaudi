# SPDX-License-Identifier: Apache-2.0
"""One-layer candidate hardware trace for a measured local regression."""
import argparse
import json
import os
from pathlib import Path

from check_deepseek_v4_mxfp4_mme import make_inputs
from vllm_gaudi.ops.deepseek_v4_mxfp4 import normal_e8m0_scales

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--normal-scales", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if os.environ.get("HABANA_PROFILE") != "profile_api_light":
        parser.error("set HABANA_PROFILE=profile_api_light before importing torch")
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select free physical modules explicitly")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    inputs, _, _ = make_inputs()
    normal = args.normal_scales and normal_e8m0_scales(*inputs[5:7])
    fn = torch.compile(torch.ops.custom_op.custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2,
                       backend="hpu_backend", fullgraph=True, dynamic=False)
    for _ in range(5):
        fn(*inputs, normal)
    torch.hpu.synchronize()
    print("Starting candidate-only single-layer device trace", flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.HPU],
                                record_shapes=False, with_stack=False, profile_memory=False) as prof:
        for index in range(args.iterations):
            with torch.profiler.record_function(f"indexed_moe_iteration_{index}"):
                fn(*inputs, normal)
        torch.hpu.synchronize()
    prof.export_chrome_trace(str(args.output))
    result = {"iterations": args.iterations, "trace": str(args.output),
              "boundary": "all candidate layer invocations followed by one device drain",
              "profiler": "profile_api_light", "baseline_run": False,
              "normal_scales": normal,
              "performance_qualification": False}
    (args.output.parent / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
