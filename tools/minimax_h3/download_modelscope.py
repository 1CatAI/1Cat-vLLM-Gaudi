#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download an H3 native-FP8 checkpoint exclusively through ModelScope."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


def _patterns(partitions: list[str]) -> list[str]:
    common = [".gitattributes", "LICENSE", "NOTICE", "README.md", "model_index.json", "evaluation/**"]
    return common + [f"{partition}/**" for partition in partitions]


def _validate_native_fp8(root: Path, partitions: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "pass", "partitions": {}}
    for partition in partitions:
        partition_result = {}
        for component in ("transformer", "text_encoder"):
            config_path = root / partition / component / "config.json"
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            quantization = config.get("quantization_config") or {}
            observed = {
                "quant_method": quantization.get("quant_method"),
                "quant_algo": quantization.get("quant_algo"),
                "producer": quantization.get("producer"),
            }
            if observed["quant_method"] != "modelopt" or observed["quant_algo"] != "FP8_PER_CHANNEL_PER_TOKEN":
                raise ValueError(
                    f"{config_path} is not the required native ModelOpt FP8_PER_CHANNEL_PER_TOKEN checkpoint: "
                    f"{observed}")
            partition_result[component] = observed
        result["partitions"][partition] = partition_result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="ModelScope repository ID for the native-FP8 H3 derivative")
    parser.add_argument("--local-dir", type=Path, required=True, help="data-disk destination")
    parser.add_argument("--revision", default="master", help="ModelScope revision or commit")
    parser.add_argument(
        "--partition",
        action="append",
        choices=("FL2VA", "Ref2VA"),
        dest="partitions",
        help="download one partition; repeat for both (default: both)",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path, help="write source and validation metadata here")
    args = parser.parse_args()
    partitions = args.partitions or ["FL2VA", "Ref2VA"]
    destination = args.local_dir.expanduser().resolve()

    # ModelScope is the only network-capable model client imported below.
    # Offline flags also prevent transitive libraries from contacting HF Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    from modelscope import snapshot_download

    resolved = Path(
        snapshot_download(
            model_id=args.model_id,
            revision=args.revision,
            local_dir=str(destination),
            allow_patterns=_patterns(partitions),
            max_workers=args.max_workers,
        )).resolve()
    validation = _validate_native_fp8(resolved, partitions)
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "download_client": "modelscope.snapshot_download",
        "huggingface_network_disabled": True,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_path": str(resolved),
        "partitions": partitions,
        "allow_patterns": _patterns(partitions),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path = args.manifest or destination / "modelscope-download-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
