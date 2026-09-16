#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit MiniMax H3 ModelOpt FP8 shards without materializing their tensors."""

from __future__ import annotations

import argparse
import fnmatch
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

from vllm_gaudi.omni.minimax_h3 import (
    map_minimax_h3_dit_weight,
    map_minimax_h3_encoder_weight,
)

_FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
_FLOAT_DTYPES = {"BF16", "F16", "F32"}


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str


def _read_component(component_dir: Path) -> tuple[dict[str, Any], dict[str, TensorMetadata]]:
    config_path = component_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index_paths = sorted(component_dir.glob("*.safetensors.index.json"))
    if len(index_paths) != 1:
        raise ValueError(f"expected one safetensors index in {component_dir}, found {index_paths}")
    weight_map = json.loads(index_paths[0].read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    metadata: dict[str, TensorMetadata] = {}
    for shard, expected_names in sorted(by_shard.items()):
        shard_path = component_dir / shard
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            missing = set(expected_names) - actual_names
            if missing:
                raise ValueError(f"{shard_path} is missing indexed tensors: {sorted(missing)}")
            for name in expected_names:
                tensor_slice = handle.get_slice(name)
                metadata[name] = TensorMetadata(
                    name=name,
                    dtype=str(tensor_slice.get_dtype()),
                    shape=tuple(tensor_slice.get_shape()),
                    shard=shard,
                )
    return config, metadata


def _module_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight")


def _matches_ignore(name: str, ignore: list[str]) -> bool:
    module = _module_name(name)
    return any(module == pattern or pattern in module or fnmatch.fnmatch(module, pattern) for pattern in ignore)


def _is_quantizable(component: str, name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if component == "transformer":
        return any(marker in name for marker in (
            ".attn.to_q.",
            ".attn.to_k.",
            ".attn.to_v.",
            ".attn.to_out.0.",
            ".ff.net.0.proj.",
            ".ff.net.2.",
            ".adaln_proj.linear.",
            "norm_out.linear.",
            "context_embedder.",
        ))
    return name.startswith("model.language_model.layers.") and any(marker in name for marker in (
        ".self_attn.q_proj.",
        ".self_attn.k_proj.",
        ".self_attn.v_proj.",
        ".self_attn.o_proj.",
        ".mlp.gate_proj.",
        ".mlp.up_proj.",
        ".mlp.down_proj.",
    ))


def _audit_component(component_dir: Path, component: str) -> dict[str, Any]:
    config, tensors = _read_component(component_dir)
    quant = config.get("quantization_config") or {}
    errors: list[str] = []
    if quant.get("quant_method") != "modelopt":
        errors.append(f"quant_method is {quant.get('quant_method')!r}, expected 'modelopt'")
    if quant.get("quant_algo") != "FP8_PER_CHANNEL_PER_TOKEN":
        errors.append(f"quant_algo is {quant.get('quant_algo')!r}, expected 'FP8_PER_CHANNEL_PER_TOKEN'")
    ignore = list(quant.get("ignore") or [])
    fp8_layers: list[dict[str, Any]] = []
    target_sources: dict[str, list[tuple[str, str | int | None]]] = defaultdict(list)

    for name, meta in sorted(tensors.items()):
        mapped = (map_minimax_h3_dit_weight(name)
                  if component == "transformer" else map_minimax_h3_encoder_weight(name))
        if mapped is not None:
            target_sources[mapped[0]].append((name, mapped[1]))
        elif name not in {"lm_head.weight", "model.language_model.norm.weight"}:
            errors.append(f"unmapped tensor: {name}")

        if meta.dtype in _FP8_DTYPES:
            if not name.endswith(".weight"):
                errors.append(f"FP8 tensor is not a weight: {name}")
                continue
            scale_name = name[:-len(".weight")] + ".weight_scale"
            scale = tensors.get(scale_name)
            if scale is None:
                errors.append(f"FP8 weight has no scale: {name}")
                continue
            if scale.dtype != "F32":
                errors.append(f"scale is not F32: {scale_name} ({scale.dtype})")
            if len(meta.shape) != 2 or scale.shape != (meta.shape[0], ):
                errors.append(f"per-channel scale shape mismatch: {name} {meta.shape}, {scale_name} {scale.shape}")
            if _matches_ignore(name, ignore):
                errors.append(f"ignored weight is unexpectedly FP8: {name}")
            fp8_layers.append({
                "source": name,
                "target": mapped[0] if mapped else None,
                "shard_id": mapped[1] if mapped else None,
                "weight_shape": list(meta.shape),
                "scale_shape": list(scale.shape),
            })
        elif _is_quantizable(component, name) and not _matches_ignore(name, ignore):
            errors.append(f"non-ignored quantizable weight is not FP8: {name} ({meta.dtype})")
        elif meta.dtype not in _FLOAT_DTYPES and not name.endswith(".weight_scale"):
            errors.append(f"unexpected tensor dtype: {name} ({meta.dtype})")

    for name in tensors:
        if not name.endswith(".weight_scale"):
            continue
        weight_name = name[:-len(".weight_scale")] + ".weight"
        if weight_name not in tensors:
            errors.append(f"orphan scale: {name}")

    # Duplicate targets are legal only for the declared packed projections.
    for target, sources in sorted(target_sources.items()):
        if len(sources) == 1:
            continue
        shard_ids = {shard for _, shard in sources}
        expected = ({"q", "k", "v"} if ".qkv_proj." in target else ({0, 1} if ".gate_up_proj." in target else None))
        if expected is None or shard_ids != expected or len(sources) != len(expected):
            errors.append(f"invalid fused mapping for {target}: {sources}")

    return {
        "component": component,
        "path": str(component_dir),
        "quant_algo": quant.get("quant_algo"),
        "producer": quant.get("producer"),
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(Counter(meta.dtype for meta in tensors.values()).items())),
        "ignored_pattern_count": len(ignore),
        "fp8_linear_count": len(fp8_layers),
        "fp8_layers": fp8_layers,
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }


def audit_checkpoint(partition_dir: Path) -> dict[str, Any]:
    components = [
        _audit_component(partition_dir / "transformer", "transformer"),
        _audit_component(partition_dir / "text_encoder", "text_encoder"),
    ]
    return {
        "partition": partition_dir.name,
        "path": str(partition_dir),
        "components": components,
        "status": "pass" if all(item["status"] == "pass" for item in components) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partitions", nargs="+", type=Path, help="FL2VA or Ref2VA checkpoint directory")
    parser.add_argument("--output", type=Path, help="write the complete JSON report here")
    args = parser.parse_args()

    report = {"schema_version": 1, "partitions": [audit_checkpoint(path.resolve()) for path in args.partitions]}
    report["status"] = "pass" if all(item["status"] == "pass" for item in report["partitions"]) else "fail"
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
