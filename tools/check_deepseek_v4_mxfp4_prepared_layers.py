# SPDX-License-Identifier: Apache-2.0
"""Check one real expert set in every DeepSeek V4 MoE layer."""

import argparse
import json
import random
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import habana_frameworks.torch.core  # noqa: E402, F401
import torch  # noqa: E402

from check_deepseek_v4_mxfp4_prepared_checkpoint import (  # noqa: E402
    CheckpointLayer,
    EXPERTS,
    TOPK,
    _allocate_prepared,
    _install_expert,
    _to_hpu_lists,
)
from deepseek_v4_mxfp4_prepared_common import comparison, native_reference  # noqa: E402
from vllm_gaudi.ops.deepseek_v4_mxfp4 import mxfp4_bf16_lut  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.layers < 1 or args.trials < 1:
        parser.error("layers and trials must be positive")

    torch.ops.load_library(__import__("os").environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    prepared = _allocate_prepared()
    lookup = mxfp4_bf16_lut("hpu")
    global_ids = torch.zeros((1, TOPK), dtype=torch.int32, device="hpu")
    local_ids = torch.arange(TOPK, dtype=torch.int32, device="hpu").view(1, TOPK)
    candidate = torch.compile(
        torch.ops.custom_op.custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2,
        backend="hpu_backend",
        fullgraph=True,
        dynamic=False,
    )
    native = torch.compile(
        native_reference,
        backend="hpu_backend",
        fullgraph=True,
        dynamic=False,
    )
    result = {
        "status": "running",
        "qualified": False,
        "model": str(args.model),
        "tp_rank": args.tp_rank,
        "layers": [],
        "trial_count_per_layer": args.trials,
        "runtime_injection": False,
        "baseline_run": False,
    }

    def save() -> None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    for layer in range(args.layers):
        reader = CheckpointLayer(args.model, layer, args.tp_rank)
        try:
            expert_ids = random.Random(20260908 + layer).sample(range(EXPERTS), TOPK)
            standard_cpu = [reader.loaded_expert(expert) for expert in expert_ids]
            for expert, standard in zip(expert_ids, standard_cpu, strict=True):
                _install_expert(prepared, expert, standard)
            standard_hpu = _to_hpu_lists(standard_cpu)
            scale_codes = tuple(value for standard in standard_cpu for value in (standard[2], standard[3]))
            normal_scales = all(
                int(value.min()) >= 2 and int(value.max()) <= 254
                for value in scale_codes
            )
            layer_result = {
                "layer": layer,
                "expert_ids": expert_ids,
                "normal_scales": normal_scales,
                "trials": [],
            }
            result["layers"].append(layer_result)
            for trial in range(args.trials):
                generator = torch.Generator().manual_seed(
                    20260908 + args.tp_rank * 10_000 + layer * 101 + trial * 1_000_003
                )
                order = torch.randperm(TOPK, generator=generator)
                ordered_experts = [expert_ids[index] for index in order.tolist()]
                global_ids.copy_(torch.tensor([ordered_experts], dtype=torch.int32))
                local_ids.copy_(order.to(torch.int32).view(1, TOPK))
                input_scale = (0.125, 0.75, 2.0, 8.0)[trial % 4]
                x = (
                    torch.randn((1, 4096), generator=generator) * input_scale
                ).bfloat16().to("hpu")
                router_cpu = torch.rand((1, TOPK), generator=generator)
                router_cpu /= router_cpu.sum(dim=-1, keepdim=True)
                router = router_cpu.bfloat16().to("hpu")
                expected = native(
                    x,
                    local_ids,
                    router,
                    standard_hpu[0],
                    standard_hpu[1],
                    standard_hpu[2],
                    standard_hpu[3],
                ).cpu()
                actual = candidate(
                    x,
                    global_ids,
                    router,
                    *prepared,
                    lookup,
                    normal_scales,
                ).cpu()
                detail = comparison(actual, expected)
                layer_result["trials"].append(
                    {
                        "trial": trial,
                        "ordered_expert_ids": ordered_experts,
                        "input_scale": input_scale,
                        "comparison": detail,
                    }
                )
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "trial": trial,
                            "experts": ordered_experts,
                            "comparison": detail,
                        }
                    ),
                    flush=True,
                )
                save()
                if not detail["exact"]:
                    result["status"] = f"failed at layer {layer} trial {trial}"
                    save()
                    raise SystemExit(2)
            torch.hpu.synchronize()
        finally:
            reader.close()

    result["qualified"] = True
    result["status"] = f"passed {args.layers} layers"
    save()


if __name__ == "__main__":
    main()
