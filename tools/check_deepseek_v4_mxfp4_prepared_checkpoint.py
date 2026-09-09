# SPDX-License-Identifier: Apache-2.0
"""Check prepared MXFP4 against native MoE with real checkpoint experts.

The saved four-expert audit sample is useful for arithmetic testing, but it
does not exercise runtime addressing across all 256 expert slots. This tool
reconstructs one TP2 rank exactly as the current HPU loader does, installs the
prepared representation at each global expert ID, and compares groups of six
experts with the native MXFP4 operator.
"""

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v4 import prepare_environment

prepare_environment()

import torch  # noqa: E402
import habana_frameworks.torch.core  # noqa: E402, F401
from safetensors import safe_open  # noqa: E402

from deepseek_v4_mxfp4_prepared_common import (  # noqa: E402
    comparison,
    native_reference,
)
from vllm_gaudi.ops.deepseek_v4_mxfp4 import (  # noqa: E402
    mxfp4_bf16_lut,
    prepare_mxfp4_q16,
    prepare_mxfp4_s16,
)


EXPERTS = 256
TOPK = 6
INTERMEDIATE_PER_RANK = 1024


def _underlying_u8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _loaded_weight_u8(tensor: torch.Tensor) -> torch.Tensor:
    """Reproduce the current HPU int8-to-uint8 parameter copy.

    The HPU copy is a numeric conversion, and negative checkpoint bytes
    saturate to zero. The archived layer-1 TP0 dump is checked against this
    reconstruction before the device comparison starts.
    """
    if tensor.dtype != torch.int8:
        raise TypeError(f"expected checkpoint int8 weight, got {tensor.dtype}")
    return tensor.clamp_min(0).to(torch.uint8)


class CheckpointLayer:
    def __init__(self, model: Path, layer: int, tp_rank: int):
        self.model = model
        self.layer = layer
        self.tp_rank = tp_rank
        self._stack = ExitStack()
        index = json.loads((model / "model.safetensors.index.json").read_text())
        self._weight_map = index["weight_map"]
        prefix = f"layers.{layer}.ffn.experts."
        files = {
            filename
            for name, filename in self._weight_map.items()
            if name.startswith(prefix)
        }
        self._files = {
            filename: self._stack.enter_context(
                safe_open(model / filename, framework="pt", device="cpu")
            )
            for filename in files
        }

    def close(self):
        self._stack.close()

    def _get(self, expert: int, suffix: str) -> torch.Tensor:
        name = f"layers.{self.layer}.ffn.experts.{expert}.{suffix}"
        return self._files[self._weight_map[name]].get_tensor(name)

    def loaded_expert(
        self, expert: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_start = self.tp_rank * INTERMEDIATE_PER_RANK
        row_stop = row_start + INTERMEDIATE_PER_RANK
        packed_start = self.tp_rank * (INTERMEDIATE_PER_RANK // 2)
        packed_stop = packed_start + INTERMEDIATE_PER_RANK // 2
        scale_start = self.tp_rank * (INTERMEDIATE_PER_RANK // 32)
        scale_stop = scale_start + INTERMEDIATE_PER_RANK // 32

        w1 = _loaded_weight_u8(self._get(expert, "w1.weight"))
        w3 = _loaded_weight_u8(self._get(expert, "w3.weight"))
        w2 = _loaded_weight_u8(self._get(expert, "w2.weight"))
        s1 = _underlying_u8(self._get(expert, "w1.scale"))
        s3 = _underlying_u8(self._get(expert, "w3.scale"))
        s2 = _underlying_u8(self._get(expert, "w2.scale"))
        return (
            torch.cat((w1[row_start:row_stop], w3[row_start:row_stop])),
            w2[:, packed_start:packed_stop].contiguous(),
            torch.cat((s1[row_start:row_stop], s3[row_start:row_stop])),
            s2[:, scale_start:scale_stop].contiguous(),
        )


def _verify_archived_sample(reader: CheckpointLayer, sample_path: Path) -> dict:
    sample = torch.load(sample_path, map_location="cpu", weights_only=True, mmap=True)
    if reader.layer != 1 or reader.tp_rank != 0:
        raise ValueError("the archived sample only covers layer 1 TP rank 0")
    fields = ("w13", "w2", "w13_scale", "w2_scale")
    checks = {}
    for slot, expert in enumerate(sample["expert_ids"]):
        loaded = reader.loaded_expert(expert)
        checks[str(expert)] = {
            field: bool(torch.equal(sample[field][slot], value))
            for field, value in zip(fields, loaded, strict=True)
        }
    if not all(value for expert in checks.values() for value in expert.values()):
        raise RuntimeError(f"checkpoint reconstruction differs from archived sample: {checks}")
    return checks


def _allocate_prepared():
    return (
        torch.full((EXPERTS, 16, 131072), 0x1111, dtype=torch.int16, device="hpu"),
        torch.full((EXPERTS, 32, 32768), 0x1111, dtype=torch.int16, device="hpu"),
        torch.full(
            (EXPERTS, 16, 16384),
            120 << 7,
            dtype=torch.int16,
            device="hpu",
        ).view(torch.bfloat16),
        torch.full(
            (EXPERTS, 32, 4096),
            120 << 7,
            dtype=torch.int16,
            device="hpu",
        ).view(torch.bfloat16),
    )


def _install_expert(prepared, expert: int, standard_cpu):
    q13, q2, s13, s2 = prepared
    w13, w2, w13_scale, w2_scale = standard_cpu
    q13[expert].copy_(prepare_mxfp4_q16(w13))
    q2[expert].copy_(prepare_mxfp4_q16(w2))
    s13[expert].copy_(prepare_mxfp4_s16(w13_scale))
    s2[expert].copy_(prepare_mxfp4_s16(w2_scale))


def _to_hpu_lists(group):
    transposed = tuple(zip(*group, strict=True))
    return tuple(tuple(value.to("hpu") for value in field) for field in transposed)


def _prepare_whole_layer_on_hpu(reader: CheckpointLayer):
    """Mirror the model loader instead of preparing selected experts on CPU."""
    standard_cpu = [reader.loaded_expert(expert) for expert in range(EXPERTS)]
    stacked = tuple(
        torch.stack(tuple(value[field] for value in standard_cpu)).to("hpu")
        for field in range(4)
    )
    prepared = (
        prepare_mxfp4_q16(stacked[0]),
        prepare_mxfp4_q16(stacked[1]),
        prepare_mxfp4_s16(stacked[2]),
        prepare_mxfp4_s16(stacked[3]),
    )
    torch.hpu.synchronize()
    return stacked, prepared


def _groups(max_groups: int | None):
    result = []
    for start in range(0, EXPERTS, TOPK):
        group = list(range(start, min(start + TOPK, EXPERTS)))
        group.extend(range(TOPK - len(group)))
        result.append(group)
    if max_groups is not None:
        result = result[:max_groups]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archived-sample", type=Path)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--trials-per-group", type=int, default=1)
    parser.add_argument("--prepare-whole-layer-on-hpu", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        parser.error("select one free physical module explicitly")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("max-groups must be positive")
    if args.trials_per_group < 1:
        parser.error("trials-per-group must be positive")

    torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    result = {
        "status": "running",
        "qualified": False,
        "model": str(args.model),
        "layer": args.layer,
        "tp_rank": args.tp_rank,
        "weight_conversion": "checkpoint int8 numeric cast to HPU uint8; negative bytes saturate to zero",
        "scale_conversion": "bit-preserving float8_e8m0fnu view as uint8",
        "groups": [],
        "trials_per_group": args.trials_per_group,
        "prepare_whole_layer_on_hpu": args.prepare_whole_layer_on_hpu,
        "runtime_injection": False,
        "baseline_run": False,
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    reader = CheckpointLayer(args.model, args.layer, args.tp_rank)
    try:
        if args.archived_sample:
            result["archived_sample"] = {
                "path": str(args.archived_sample),
                "sha256": hashlib.sha256(args.archived_sample.read_bytes()).hexdigest(),
                "checks": _verify_archived_sample(reader, args.archived_sample),
            }
            save()

        stacked_hpu = None
        if args.prepare_whole_layer_on_hpu:
            stacked_hpu, prepared = _prepare_whole_layer_on_hpu(reader)
        else:
            prepared = _allocate_prepared()
        lookup = mxfp4_bf16_lut("hpu")
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
        for group_index, expert_ids in enumerate(_groups(args.max_groups)):
            if stacked_hpu is None:
                standard_cpu = [reader.loaded_expert(expert) for expert in expert_ids]
                for expert, standard in zip(expert_ids, standard_cpu, strict=True):
                    _install_expert(prepared, expert, standard)
                standard_hpu = _to_hpu_lists(standard_cpu)
            else:
                standard_cpu = None
                standard_hpu = tuple(
                    tuple(field[expert] for expert in expert_ids)
                    for field in stacked_hpu
                )
            global_ids = torch.tensor([expert_ids], dtype=torch.int32, device="hpu")
            group_result = {
                "index": group_index,
                "expert_ids": expert_ids,
                "trials": [],
            }
            result["groups"].append(group_result)
            for trial in range(args.trials_per_group):
                generator = torch.Generator().manual_seed(
                    20260908 + args.layer * 1000 + args.tp_rank * 100
                    + group_index + trial * 1_000_003
                )
                order = (
                    torch.arange(TOPK)
                    if trial == 0
                    else torch.randperm(TOPK, generator=generator)
                )
                ordered_expert_ids = [expert_ids[index] for index in order.tolist()]
                global_ids.copy_(torch.tensor([ordered_expert_ids], dtype=torch.int32))
                local_ids.copy_(order.to(torch.int32).view(1, TOPK))
                input_scale = (0.75, 0.125, 2.0, 8.0)[trial % 4]
                x = (torch.randn(1, 4096, generator=generator) * input_scale).bfloat16().to("hpu")
                if trial == 0:
                    router_cpu = torch.tensor([[0.24, 0.21, 0.18, 0.15, 0.12, 0.10]])
                else:
                    router_cpu = torch.rand(1, TOPK, generator=generator)
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
                    True,
                ).cpu()
                detail = comparison(actual, expected)
                group_result["trials"].append({
                    "index": trial,
                    "ordered_expert_ids": ordered_expert_ids,
                    "input_scale": input_scale,
                    "comparison": detail,
                })
                group_result["comparison"] = {
                    "exact": all(item["comparison"]["exact"] for item in group_result["trials"]),
                    "nonzero": sum(item["comparison"]["nonzero"] for item in group_result["trials"]),
                    "max_abs": max(item["comparison"]["max_abs"] for item in group_result["trials"]),
                    "relative_l2_max": max(
                        item["comparison"]["relative_l2"] for item in group_result["trials"]
                    ),
                }
                save()
                print(
                    json.dumps({
                        "group": group_index,
                        "experts": expert_ids,
                        "trial": trial,
                        "input_scale": input_scale,
                        "comparison": detail,
                    }),
                    flush=True,
                )
                if not detail["exact"]:
                    result["status"] = "failed: prepared output differs from native reference"
                    save()
                    raise SystemExit(2)
                del x, router, expected, actual
            del standard_hpu, standard_cpu, global_ids

        covered = sorted({expert for group in result["groups"] for expert in group["expert_ids"]})
        result["covered_experts"] = covered
        result["qualified"] = len(covered) == EXPERTS
        result["status"] = (
            "passed all 256 global expert IDs"
            if result["qualified"]
            else "partial diagnostic passed"
        )
        save()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
