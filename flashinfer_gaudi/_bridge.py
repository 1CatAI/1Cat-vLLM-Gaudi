# SPDX-License-Identifier: Apache-2.0
"""Fail-closed loader for the optional, private-ABI mixed-engine adapter."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading

import torch

from flashinfer_gaudi._dispatch import BackendUnavailableError
from flashinfer_gaudi._native import _configure_kernel_database_path

BRIDGE_VERSION = "1.24.1.482"
LIBRARIES = ("libhabana_pytorch2_plugin.upstream.so", "libhabana_pytorch_backend.upstream.so")
_LOCK = threading.Lock()
_LOADED = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict:
    from habana_frameworks.torch.version import __version__

    synapse = ctypes.CDLL("/usr/lib/habanalabs/libSynapse.so")
    get_version = synapse.synDriverGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
    get_version.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(256)
    if get_version(buffer, len(buffer)) != 0:
        raise BackendUnavailableError("Cannot identify the Synapse runtime version")
    return {
        "bridge_version": __version__,
        "synapse_version": buffer.value.decode("ascii"),
        "torch_version": torch.__version__,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def runtime_files(directory: Path) -> dict[str, Path]:
    from habana_frameworks.torch.utils.lib_utils import get_lib_dir

    return {
        "adapter": directory / "flashinfer_gaudi_bridge_ops.so",
        "kernel_database": directory / "libflashinfer_gaudi_kernels.so",
        "synapse": Path("/usr/lib/habanalabs/libSynapse.so"),
        **{
            name: Path(get_lib_dir()) / name
            for name in LIBRARIES
        },
    }


def validate_manifest(manifest: dict, identity: dict, files: dict[str, Path]) -> str:
    """Validate versions and all ELF hashes before executing adapter static initializers."""
    if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
            or manifest.get("target") != "gaudi2"):
        raise BackendUnavailableError("Unsupported native Bridge artifact schema/target")
    if (identity.get("bridge_version") != BRIDGE_VERSION
            or identity.get("synapse_version", "").split(".")[:3] != ["1", "24", "1"]
            or manifest.get("runtime") != identity):
        raise BackendUnavailableError("Native Bridge artifact runtime ABI mismatch; rebuild for the pinned runtime")
    hashes = manifest.get("sha256", {})
    if set(hashes) != set(files):
        raise BackendUnavailableError("Native Bridge artifact file inventory mismatch")
    for name, path in files.items():
        if not path.is_file() or sha256(path) != hashes[name]:
            raise BackendUnavailableError(f"Native Bridge artifact checksum mismatch: {name}")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_bridge_adapter() -> dict:
    """Explicit opt-in; never called by auto dispatch or CPU-only imports."""
    from flashinfer_gaudi import _native
    global _LOADED
    with _LOCK:
        if _LOADED is not None:
            _native._SILU_MUL_QUANT_OP = torch.ops.custom_op.flashinfer_gaudi_silu_mul_quant
            return dict(_LOADED)
        directory = Path(__file__).resolve().parent / "lib"
        try:
            identity = runtime_identity()
            files = runtime_files(directory)
            manifest = json.loads((directory / "bridge_artifact_v1.json").read_text())
            artifact_id = validate_manifest(manifest, identity, files)
            if os.environ.get("PT_HPU_LAZY_MODE", "1") != "0":
                raise BackendUnavailableError("Native Bridge adapter currently requires PT_HPU_LAZY_MODE=0")
            # Detect preloaded libraries from another Bridge installation too.
            mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/" in line}
            for name in LIBRARIES:
                expected = files[name].resolve()
                for value in mapped:
                    if Path(value).name in (name, name.replace(".upstream", "")) and Path(value).resolve() != expected:
                        raise BackendUnavailableError("A different native Bridge library is already mapped")
            _configure_kernel_database_path()
            torch.ops.load_library(str(files["adapter"]))
            _native._SILU_MUL_QUANT_OP = torch.ops.custom_op.flashinfer_gaudi_silu_mul_quant
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(f"Native Bridge adapter unavailable: {type(exc).__name__}: {exc}") from exc
        _LOADED = {"artifact_id": artifact_id, "runtime": identity, "production_promoted": False}
        return dict(_LOADED)
