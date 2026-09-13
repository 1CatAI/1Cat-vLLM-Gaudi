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
_SILU_AND_MUL_OP: Callable | None = None
_SILU_MUL_QUANT_OP: Callable | None = None
_BLOCK_FP8_DEQUANT_OP: Callable | None = None
_BLOCK_FP8_LINEAR_OP: Callable | None = None
_ADD_RMSNORM_QUANT_OP: Callable | None = None


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
    kernel_path = str(kernel_db)
    if kernel_path in current:
        return

    configured = [kernel_path, *current]
    # Setting GC_KERNEL_PATH replaces SynapseAI's implicit default. Preserve
    # the stock kernel database when this package is the first component to
    # configure the variable.
    system_kernel_db = Path("/usr/lib/habanalabs/libtpc_kernels.so")
    if not current and system_kernel_db.is_file():
        configured.append(str(system_kernel_db))
    os.environ["GC_KERNEL_PATH"] = os.pathsep.join(configured)


def load_native_extensions() -> tuple[str, ...]:
    """Load packaged or explicitly configured native libraries once.

    Loading failures are recorded for diagnostics. They are not raised here so
    CPU-only imports and the PyTorch reference fallback remain usable.
    """
    global _LOAD_ATTEMPTED, _SILU_AND_MUL_OP, _ADD_RMSNORM_QUANT_OP
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
        try:
            _SILU_AND_MUL_OP = torch.ops.custom_op.flashinfer_gaudi_silu_and_mul
        except (AttributeError, RuntimeError):
            _SILU_AND_MUL_OP = None
        try:
            _ADD_RMSNORM_QUANT_OP = torch.ops.custom_op.flashinfer_gaudi_add_rmsnorm_quant
        except (AttributeError, RuntimeError):
            _ADD_RMSNORM_QUANT_OP = None
        # Private compound ops are published only by the verified Bridge
        # loader, not by probing an arbitrary externally registered symbol.
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


def silu_and_mul_op() -> Callable | None:
    # The public CustomOp API requires custom_op for full torch.compile support.
    # Registration is immutable after the one-shot loader. Resolve it at
    # compilation rather than installing guards on loader locks/path lists.
    if not _LOAD_ATTEMPTED:
        load_native_extensions()
    return _SILU_AND_MUL_OP


def public_mtp_gdn_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "gdn_mtp_packed")


def public_mtp_prepared_op() -> Callable | None:
    return _resolve_op("custom_op", "flashinfer_gaudi_gdn_mtp_prepared")


def native_dflash2_grouped_conv_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_grouped_conv")


def native_dflash2_select_path_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_select_path")


def native_dflash2_score_select_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_score_select")


def native_dflash2_top_k_op() -> Callable | None:
    return _resolve_op("flashinfer_gaudi", "dflash2_top_k")


def native_diagnostics() -> dict[str, object]:
    load_native_extensions()
    return {
        "loaded_libraries": tuple(_LOADED_PATHS),
        "load_errors": tuple(_LOAD_ERRORS),
        "public_packed_gdn": public_packed_gdn_op() is not None,
        "bridge_packed_gdn": bridge_packed_gdn_op() is not None,
        "silu_and_mul": silu_and_mul_op() is not None,
        "silu_and_mul_quant": silu_mul_quant_op() is not None,
        "block_fp8_dequant": block_fp8_dequant_op() is not None,
        "block_fp8_linear": block_fp8_linear_op() is not None,
        "fused_add_rmsnorm_quant": add_rmsnorm_quant_op() is not None,
        "public_mtp_gdn": public_mtp_gdn_op() is not None,
        "public_mtp_prepared": public_mtp_prepared_op() is not None,
        "dflash2_grouped_conv": native_dflash2_grouped_conv_op() is not None,
        "dflash2_select_path": native_dflash2_select_path_op() is not None,
        "dflash2_score_select": native_dflash2_score_select_op() is not None,
        "dflash2_top_k": native_dflash2_top_k_op() is not None,
    }


def _reset_native_state_for_tests() -> None:
    global _LOAD_ATTEMPTED, _SILU_AND_MUL_OP, _SILU_MUL_QUANT_OP
    global _BLOCK_FP8_DEQUANT_OP, _BLOCK_FP8_LINEAR_OP
    global _ADD_RMSNORM_QUANT_OP
    with _LOAD_LOCK:
        _LOAD_ATTEMPTED = False
        _SILU_AND_MUL_OP = None
        _SILU_MUL_QUANT_OP = None
        _BLOCK_FP8_DEQUANT_OP = None
        _BLOCK_FP8_LINEAR_OP = None
        _ADD_RMSNORM_QUANT_OP = None
        _LOADED_PATHS.clear()
        _LOAD_ERRORS.clear()


def silu_mul_quant_op() -> Callable | None:
    if not _LOAD_ATTEMPTED:
        load_native_extensions()
    return _SILU_MUL_QUANT_OP


def block_fp8_dequant_op() -> Callable | None:
    # Private ops are published only by load_bridge_adapter before compile.
    return _BLOCK_FP8_DEQUANT_OP


def block_fp8_linear_op() -> Callable | None:
    return _BLOCK_FP8_LINEAR_OP


def add_rmsnorm_quant_op() -> Callable | None:
    if not _LOAD_ATTEMPTED:
        load_native_extensions()
    return _ADD_RMSNORM_QUANT_OP


# SynapseAI reads GC_KERNEL_PATH when its graph compiler is initialized. Set
# the packaged kernel database path as soon as this lightweight loader module
# is imported; the PyTorch registration extension itself remains lazy.
_configure_kernel_database_path()
