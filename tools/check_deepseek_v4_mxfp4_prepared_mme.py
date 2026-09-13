# SPDX-License-Identifier: Apache-2.0
"""Compile and execute the full prepared MXFP4 MoE for placement inspection."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from deepseek_v4_mxfp4_prepared_common import (
    comparison,
    compile_prepared_moe,
    make_prepared_inputs,
    native_reference,
    selected_native_inputs,
)

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--general", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select free physical modules explicitly")
    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    prepared, standard, ids, _ = make_prepared_inputs(args.sample)
    candidate = compile_prepared_moe()
    native = torch.compile(native_reference, backend="hpu_backend", fullgraph=True, dynamic=False)
    expected = native(*selected_native_inputs(standard, ids)).cpu()
    actual = candidate(*prepared, not args.general).cpu()
    detail = comparison(actual, expected)
    if not detail["exact"]:
        raise RuntimeError(f"prepared complete MoE differs from the native reference: {detail}")
    prepared[1].copy_(ids.flip(1))
    changed = candidate(*prepared, not args.general).cpu()
    prepared[1].copy_(ids)
    replay = candidate(*prepared, not args.general).cpu()
    assert torch.equal(actual, replay) and not torch.equal(actual, changed)
    result = {
        "status": "executed; memory placement requires graph inspection",
        "qualified": False,
        "normal_scales": not args.general,
        "correctness": detail,
        "replay_exact": True,
        "dynamic_ids_observed": True,
        "output_sha256": hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "runtime_injection": False,
        "timing": None,
        "baseline_run": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
