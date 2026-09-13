# SPDX-License-Identifier: Apache-2.0
"""Candidate-only hardware trace for the prepared MXFP4 MoE."""
import argparse
import json
import os
from pathlib import Path

from deepseek_v4_mxfp4_prepared_common import compile_prepared_moe, make_prepared_inputs

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    if os.environ.get("HABANA_PROFILE") != "profile_api_light":
        parser.error("set HABANA_PROFILE=profile_api_light before importing torch")
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select free physical modules explicitly")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    inputs, standard, _, _ = make_prepared_inputs()
    del standard
    fn = compile_prepared_moe()
    for _ in range(5):
        fn(*inputs, True)
    torch.hpu.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as profiler:
        for index in range(args.iterations):
            with torch.profiler.record_function(f"prepared_moe_iteration_{index}"):
                fn(*inputs, True)
        torch.hpu.synchronize()
    profiler.export_chrome_trace(str(args.output))
    result = {
        "iterations": args.iterations,
        "trace": str(args.output),
        "boundary": "all prepared candidate layer invocations followed by one device drain",
        "profiler": "profile_api_light",
        "baseline_run": False,
        "normal_scales": True,
        "performance_qualification": False,
    }
    (args.output.parent / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
