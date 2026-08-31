# SPDX-License-Identifier: Apache-2.0
"""Lazy loading and discovery of optional native Gaudi kernels."""

from __future__ import annotations

import glob
import os
import threading
from pathlib import Path
from typing import Callable

import torch

_LOAD_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False
_LOADED_PATHS: list[str] = []
_LOAD_ERRORS: list[str] = []


def _library_candidates() -> list[str]:
    candidates: list[str] = []
    configured = os.environ.get("FLASHINFER_GAUDI_NATIVE_LIBRARY", "")
    if configured:
        candidates.extend(path for path in configured.split(os.pathsep) if path)

    package_lib = Path(__file__).resolve().parent / "lib"
    candidates.extend(sorted(glob.glob(str(package_lib / "flashinfer_gaudi_ops*.so"))))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(Path(candidate).expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _configure_kernel_database_path() -> None:
    package_lib = Path(__file__).resolve().parent / "lib"
    kernel_db = package_lib / "libflashinfer_gaudi_kernels.so"
    if not kernel_db.is_file():
        return
    current = [entry for entry in os.environ.get("GC_KERNEL_PATH", "").split(os.pathsep) if entry]
    package_path = str(package_lib)
    if package_path not in current:
        os.environ["GC_KERNEL_PATH"] = os.pathsep.join((package_path, *current))


def load_native_extensions() -> tuple[str, ...]:
    """Load packaged or explicitly configured native libraries once.

    Loading failures are recorded for diagnostics. They are not raised here so
    CPU-only imports and the PyTorch reference fallback remain usable.
    """
    global _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return tuple(_LOADED_PATHS)
    with _LOAD_LOCK:
        if _LOAD_ATTEMPTED:
            return tuple(_LOADED_PATHS)
        _configure_kernel_database_path()
        for candidate in _library_candidates():
            if not os.path.isfile(candidate):
                _LOAD_ERRORS.append(f"{candidate}: file does not exist")
                continue
            try:
                torch.ops.load_library(candidate)
            except Exception as exc:  # pragma: no cover - requires an HPU runtime
                _LOAD_ERRORS.append(f"{candidate}: {type(exc).__name__}: {exc}")
            else:
                _LOADED_PATHS.append(candidate)
        _LOAD_ATTEMPTED = True
    return tuple(_LOADED_PATHS)


def _resolve_op(namespace: str, name: str) -> Callable | None:
    load_native_extensions()
    try:
        op_namespace = getattr(torch.ops, namespace)
        return getattr(op_namespace, name)
    except (AttributeError, RuntimeError):
        return None


def public_packed_gdn_op() -> Callable | None:
    """Return the preferred public packed-GDN op, including the legacy GUID."""
    return (_resolve_op("flashinfer_gaudi", "gdn_decode_packed")
            or _resolve_op("custom_op", "custom_gdn_packed_decode_f32_gaudi2"))


def bridge_packed_gdn_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi_bridge", "gdn_decode_packed")


def native_diagnostics() -> dict[str, object]:
    load_native_extensions()
    return {
        "loaded_libraries": tuple(_LOADED_PATHS),
        "load_errors": tuple(_LOAD_ERRORS),
        "public_packed_gdn": public_packed_gdn_op() is not None,
        "bridge_packed_gdn": bridge_packed_gdn_op() is not None,
    }


def _reset_native_state_for_tests() -> None:
    global _LOAD_ATTEMPTED
    with _LOAD_LOCK:
        _LOAD_ATTEMPTED = False
        _LOADED_PATHS.clear()
        _LOAD_ERRORS.clear()
