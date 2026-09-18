#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble the official BF16 H3 Ref2VA partition without duplicate downloads."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "MiniMax/MiniMax-H3"
SHARED_COMPONENTS = ("text_encoder", "video_vae", "audio_vae", "tokenizer", "processor")
DOWNLOAD_PATTERNS = (
    "LICENSE",
    "NOTICE",
    "README.md",
    "configuration.json",
    "Ref2VA/model_index.json",
    "Ref2VA/transformer/**",
)


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: str


def _remote_file(value: object) -> RemoteFile | None:
    if getattr(value, "type", "blob") == "tree":
        return None
    path = getattr(value, "path", None)
    size = getattr(value, "size", None)
    sha256 = getattr(value, "sha256", None)
    if not isinstance(path, str) or not isinstance(size, int) or not isinstance(sha256, str):
        raise ValueError(f"ModelScope returned incomplete file metadata: {value!r}")
    return RemoteFile(path=path, size=size, sha256=sha256)


def _validate_remote_shared_components(files: Iterable[object]) -> dict[str, list[RemoteFile]]:
    """Prove that Ref2VA and FL2VA share byte-identical non-DiT files."""

    indexed = {item.path: item for value in files if (item := _remote_file(value)) is not None}
    result: dict[str, list[RemoteFile]] = {}
    for component in SHARED_COMPONENTS:
        fl_prefix = f"FL2VA/{component}/"
        ref_prefix = f"Ref2VA/{component}/"
        fl_files = {path[len(fl_prefix):]: item for path, item in indexed.items() if path.startswith(fl_prefix)}
        ref_files = {path[len(ref_prefix):]: item for path, item in indexed.items() if path.startswith(ref_prefix)}
        if not fl_files or fl_files.keys() != ref_files.keys():
            raise ValueError(f"ModelScope {component} file sets differ between FL2VA and Ref2VA: "
                             f"fl2va={len(fl_files)}, ref2va={len(ref_files)}")
        mismatches = [
            name for name in sorted(fl_files)
            if (fl_files[name].size, fl_files[name].sha256) != (ref_files[name].size, ref_files[name].sha256)
        ]
        if mismatches:
            raise ValueError(f"ModelScope {component} differs between FL2VA and Ref2VA: {mismatches[:5]}")
        result[component] = [
            RemoteFile(
                path=f"{component}/{name}",
                size=ref_files[name].size,
                sha256=ref_files[name].sha256,
            ) for name in sorted(ref_files)
        ]
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_shared_source(
    source_partition: Path,
    components: dict[str, list[RemoteFile]],
) -> dict[str, dict[str, object]]:
    summaries = {}
    for component, records in components.items():
        digest = hashlib.sha256()
        for record in records:
            path = source_partition / record.path
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != record.size:
                raise ValueError(f"shared H3 file size mismatch: {path}={path.stat().st_size}, expected={record.size}")
            observed_sha256 = _sha256(path)
            if observed_sha256 != record.sha256:
                raise ValueError(f"shared H3 file SHA256 mismatch: {path}={observed_sha256}, expected={record.sha256}")
            digest.update(record.path.encode())
            digest.update(b"\0")
            digest.update(record.sha256.encode())
            digest.update(b"\n")
        summaries[component] = {
            "files": len(records),
            "bytes": sum(record.size for record in records),
            "file_manifest_sha256": digest.hexdigest(),
        }
    return summaries


def _validate_transformer(
    partition: Path,
    remote_files: Iterable[object],
) -> dict[str, object]:
    from safetensors import safe_open

    component = partition / "transformer"
    config_path = component / "config.json"
    index_path = component / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("quantization_config") is not None:
        raise ValueError(f"LightX2V Ref2VA base transformer must be BF16, got {config.get('quantization_config')}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    if len(shards) != 13:
        raise ValueError(f"official Ref2VA transformer must contain 13 shards, got {len(shards)}")

    remote = {
        item.path: item
        for value in remote_files
        if (item := _remote_file(value)) is not None and item.path.startswith("Ref2VA/transformer/")
    }
    tensor_dtypes: dict[str, int] = {}
    shard_hashes = {}
    for name in shards:
        path = component / name
        remote_path = f"Ref2VA/transformer/{name}"
        record = remote.get(remote_path)
        if record is None:
            raise ValueError(f"ModelScope manifest is missing {remote_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.size:
            raise ValueError(f"Ref2VA transformer shard size mismatch: {path}")
        observed_sha256 = _sha256(path)
        if observed_sha256 != record.sha256:
            raise ValueError(f"Ref2VA transformer shard SHA256 mismatch: {path}")
        shard_hashes[name] = observed_sha256
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint:
                dtype = checkpoint.get_slice(key).get_dtype()
                tensor_dtypes[dtype] = tensor_dtypes.get(dtype, 0) + 1
    unsupported = sorted(set(tensor_dtypes) - {"BF16", "F32"})
    if unsupported or not tensor_dtypes.get("BF16"):
        raise ValueError(f"Ref2VA transformer is not BF16: tensor_dtypes={tensor_dtypes}")
    return {
        "format": "bf16",
        "shards": len(shards),
        "bytes": sum((component / name).stat().st_size for name in shards),
        "tensor_dtypes": tensor_dtypes,
        "shard_sha256": shard_hashes,
    }


def _source_partition(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path if path.name == "FL2VA" else path / "FL2VA"


def _link_shared_components(destination: Path, source: Path) -> None:
    for component in SHARED_COMPONENTS:
        target = destination / component
        expected = source / component
        if target.is_symlink():
            if target.resolve() != expected.resolve():
                raise ValueError(f"{target} points to {target.resolve()}, expected {expected.resolve()}")
            continue
        if target.exists():
            raise FileExistsError(f"refusing to replace existing component {target}")
        target.symlink_to(Path(os.path.relpath(expected, target.parent)), target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-components-from",
        type=Path,
        required=True,
        help="official BF16 FL2VA root or partition supplying byte-identical shared components",
    )
    parser.add_argument("--revision", default="master")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    destination = args.local_dir.expanduser().resolve()
    source_partition = _source_partition(args.reuse_components_from)

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        DIFFUSERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    )
    from modelscope import snapshot_download
    from modelscope_hub.api import HubApi
    from modelscope_hub.constants import RepoType

    remote_files = HubApi().list_repo_files(
        MODEL_ID,
        RepoType.MODEL,
        revision=args.revision,
        recursive=True,
    )
    shared_contract = _validate_remote_shared_components(remote_files)
    shared = _validate_shared_source(source_partition, shared_contract)

    resolved = Path(
        snapshot_download(
            model_id=MODEL_ID,
            revision=args.revision,
            local_dir=str(destination),
            allow_patterns=list(DOWNLOAD_PATTERNS),
            max_workers=args.max_workers,
        )).resolve()
    partition = resolved / "Ref2VA"
    index_path = partition / "model_index.json"
    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    release = model_index.get("_minimax_h3") or {}
    if str(release.get("partition", "")).lower() != "ref2va":
        raise ValueError(f"invalid MiniMax-H3 Ref2VA model index at {index_path}")
    transformer = _validate_transformer(partition, remote_files)
    _link_shared_components(partition, source_partition)

    result = {
        "schema_version": 1,
        "status": "pass",
        "download_client": "modelscope.snapshot_download",
        "remote_manifest_client": "modelscope_hub.HubApi.list_repo_files",
        "huggingface_network_disabled": True,
        "base_model_id": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_path": str(resolved),
        "allow_patterns": list(DOWNLOAD_PATTERNS),
        "partition": "Ref2VA",
        "transformer": transformer,
        "reused_components_from": str(source_partition),
        "reused_components": shared,
        "reused_bytes": sum(int(item["bytes"]) for item in shared.values()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    manifest_path = args.manifest or destination / "ref2va-base-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
