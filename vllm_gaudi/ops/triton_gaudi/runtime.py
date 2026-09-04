# SPDX-License-Identifier: Apache-2.0
"""Policy and diagnostics for the Gaudi2-native Triton fast path."""

from __future__ import annotations

from collections import Counter
from enum import Enum
import logging
import os
from pathlib import Path
import threading

import torch

logger = logging.getLogger(__name__)


class FastPathMode(str, Enum):
    OFF = "off"
    HYBRID = "hybrid"
    STRICT = "strict"


class FastPathUnavailable(RuntimeError):
    """Raised when strict mode cannot execute a Gaudi-native Triton kernel."""


_lock = threading.Lock()
_prepared = False
_available = False
_prepare_error: str | None = None
_mode_value: FastPathMode | None = None
_warned: set[str] = set()
_counters: Counter[str] = Counter()

# The split causal-conv + GDN path currently wins end-to-end only for the
# batch-eight decode bucket. Larger buckets lose to the vendor graph, while
# non-power-of-two graph geometries do not yet preserve every conv-cache
# mutation. Keep hybrid Pareto-safe and strict graph execution fail-closed.
_GDN_CONV_SPLIT_HYBRID_BATCH = 8
_GDN_CONV_SPLIT_GRAPH_BATCHES = frozenset((1, 8, 32))
_DYNAMIC_QUANT_MAX_ROWS = 32
_SILU_DYNAMIC_QUANT_MAX_COLS = 4096


def _mode() -> FastPathMode:
    global _mode_value
    if _mode_value is not None:
        return _mode_value
    value = os.environ.get("VLLM_HPU_TRITON_MODE", "off").strip().lower()
    try:
        _mode_value = FastPathMode(value)
        return _mode_value
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in FastPathMode)
        raise ValueError(f"VLLM_HPU_TRITON_MODE must be one of: {choices}") from exc


@torch.compiler.assume_constant_result
def _compile_fast_path_mode() -> str:
    """Freeze rollout policy before Dynamo builds an HPU graph."""
    mode = _mode()
    if mode is not FastPathMode.OFF and not (_prepared and _available):
        raise FastPathUnavailable("Gaudi Triton must be prepared before torch.compile and before "
                                  "the first HPU allocation; call prepare_if_enabled() during "
                                  "operator registration")
    return mode.value


def _block_size() -> int:
    value = int(os.environ.get("VLLM_HPU_TRITON_BLOCK_SIZE", "256"))
    if value <= 0 or value > 65536:
        raise ValueError("VLLM_HPU_TRITON_BLOCK_SIZE must be in [1, 65536]")
    return value


def _warn_once(key: str, message: str) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)


def _fail_or_fallback(reason: str, lhs: "torch.Tensor", rhs: "torch.Tensor") -> "torch.Tensor":
    _counters[f"fallback.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable(f"Gaudi Triton strict mode rejected vector_add: {reason}")
    _warn_once(reason, f"Gaudi Triton fast path unavailable ({reason}); using the HPU vendor path")
    return lhs + rhs


def prepare_if_enabled() -> bool:
    """Prepare the perf-library/cache ABI without initializing a fake GPU model."""
    global _available, _prepare_error, _prepared

    mode = _mode()
    if mode is FastPathMode.OFF:
        return False
    if _prepared:
        if not _available:
            reason = _prepare_error or "unknown initialization failure"
            if mode is FastPathMode.STRICT:
                raise FastPathUnavailable(f"Gaudi Triton strict mode cannot initialize: {reason}")
            _warn_once("initialization", f"Gaudi Triton fast path is disabled: {reason}")
        return _available
    with _lock:
        if not _prepared:
            try:
                configured_cache = os.environ.get("VLLM_HPU_TRITON_CACHE_DIR")
                if configured_cache:
                    cache_path = Path(configured_cache).expanduser().resolve()
                    os.environ.setdefault("TRITON_GAUDI_ARTIFACT_DIR", str(cache_path))

                from triton.backends.gaudi.driver import prepare_environment, validate_bridge_launch_abi

                prepare_environment()
                validate_bridge_launch_abi()
                from vllm_gaudi.ops.triton_gaudi.fusion import (
                    register_silu_dynamic_quant_fusion_pass, )

                register_silu_dynamic_quant_fusion_pass()
                _available = True
            except (ImportError, OSError, RuntimeError) as exc:
                _prepare_error = str(exc)
                _available = False
            _prepared = True

    if not _available:
        reason = _prepare_error or "unknown initialization failure"
        if mode is FastPathMode.STRICT:
            raise FastPathUnavailable(f"Gaudi Triton strict mode cannot initialize: {reason}")
        _warn_once("initialization", f"Gaudi Triton fast path is disabled: {reason}")
    return _available


def vector_add(lhs: "torch.Tensor", rhs: "torch.Tensor") -> "torch.Tensor":
    """Add equal contiguous HPU tensors through the native TPC fast path."""
    if _mode() is FastPathMode.OFF:
        _counters["vendor.off"] += 1
        return lhs + rhs
    if lhs.device.type != "hpu" or rhs.device.type != "hpu":
        return _fail_or_fallback("non_hpu_tensor", lhs, rhs)
    if lhs.shape != rhs.shape:
        return _fail_or_fallback("shape_mismatch", lhs, rhs)
    if lhs.dtype != rhs.dtype or str(lhs.dtype) not in ("torch.float32", "torch.bfloat16"):
        return _fail_or_fallback("unsupported_dtype", lhs, rhs)
    if not lhs.is_contiguous() or not rhs.is_contiguous():
        return _fail_or_fallback("non_contiguous", lhs, rhs)
    if lhs.numel() == 0:
        return _fail_or_fallback("zero_size", lhs, rhs)
    if not prepare_if_enabled():
        return _fail_or_fallback("initialization", lhs, rhs)

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import vector_add as launch_vector_add

        output = launch_vector_add(lhs, rhs, _block_size())
        _counters["triton.vector_add"] += 1
        return output
    except Exception as exc:
        if _mode() is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton vector_add compilation or launch failed") from exc
        _warn_once("launch", f"Gaudi Triton vector_add failed ({exc}); using the HPU vendor path")
        _counters["fallback.launch"] += 1
        return lhs + rhs


def _reject_fused_add_rms_norm(reason: str) -> None:
    _counters[f"fallback.fused_add_rms_norm.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable(f"Gaudi Triton strict mode rejected fused_add_rms_norm: {reason}")
    _warn_once(
        f"fused_add_rms_norm.{reason}",
        f"Gaudi Triton fused add+RMSNorm unavailable ({reason}); using the HPU vendor path",
    )
    return None


def _fused_add_rms_norm_rejection_reason(
    hidden_states: "torch.Tensor",
    residual: "torch.Tensor",
    weight: "torch.Tensor",
) -> str | None:
    if hidden_states.device.type != "hpu" or residual.device.type != "hpu" or weight.device.type != "hpu":
        return "non_hpu_tensor"
    if hidden_states.device != residual.device or hidden_states.device != weight.device:
        return "device_mismatch"
    if hidden_states.shape != residual.shape or hidden_states.ndim == 0:
        return "shape_mismatch"
    n_cols = hidden_states.shape[-1]
    if weight.ndim != 1 or weight.numel() != n_cols:
        return "weight_shape"
    if n_cols <= 0 or n_cols > 8192 or hidden_states.numel() == 0:
        return "unsupported_size"
    if hidden_states.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return "unsupported_dtype"
    if not hidden_states.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        return "non_contiguous"
    return None


def fused_add_rms_norm(
    hidden_states: "torch.Tensor",
    residual: "torch.Tensor",
    weight: "torch.Tensor",
    epsilon: float,
) -> tuple["torch.Tensor", "torch.Tensor"] | None:
    """Run the Gaudi2 BF16 fused residual-add and RMSNorm fast path."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode == FastPathMode.OFF.value:
            return None
        # The eager kernel clears its performance gate, but current
        # hpu_backend fullgraph recipes do not. Keep hybrid rollout
        # performance-safe; strict mode remains the explicit CI/A-B path.
        if compile_mode == FastPathMode.HYBRID.value:
            return None
        rejection_reason = _fused_add_rms_norm_rejection_reason(hidden_states, residual, weight)
        if rejection_reason is not None:
            if compile_mode == FastPathMode.STRICT.value:
                raise FastPathUnavailable(f"Gaudi Triton strict mode rejected fused_add_rms_norm: {rejection_reason}")
            return None
        from vllm_gaudi.ops.triton_gaudi.kernels import fused_add_rms_norm as launch_fused_add_rms_norm

        return launch_fused_add_rms_norm(hidden_states, residual, weight, float(epsilon))

    mode = _mode()
    if mode is FastPathMode.OFF:
        _counters["vendor.fused_add_rms_norm.off"] += 1
        return None
    rejection_reason = _fused_add_rms_norm_rejection_reason(hidden_states, residual, weight)
    if rejection_reason is not None:
        return _reject_fused_add_rms_norm(rejection_reason)
    if mode is FastPathMode.HYBRID:
        _counters["vendor.fused_add_rms_norm.performance_gate"] += 1
        return None
    if not prepare_if_enabled():
        return _reject_fused_add_rms_norm("initialization")

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import fused_add_rms_norm as launch_fused_add_rms_norm

        output = launch_fused_add_rms_norm(hidden_states, residual, weight, float(epsilon))
        _counters["triton.fused_add_rms_norm"] += 1
        return output
    except Exception as exc:
        if _mode() is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton fused_add_rms_norm compilation or launch failed") from exc
        _warn_once(
            "fused_add_rms_norm.launch",
            f"Gaudi Triton fused add+RMSNorm failed ({exc}); using the HPU vendor path",
        )
        _counters["fallback.fused_add_rms_norm.launch"] += 1
        return None


def _reject_dynamic_quant(reason: str) -> None:
    _counters[f"fallback.dynamic_quant.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable(f"Gaudi Triton strict mode rejected dynamic_quant: {reason}")
    return None


def _dynamic_quant_rejection_reason(input_tensor: "torch.Tensor", ) -> str | None:
    if input_tensor.device.type != "hpu":
        return "non_hpu_tensor"
    if input_tensor.ndim != 2 or input_tensor.shape[0] <= 0:
        return "shape_mismatch"
    rows, n_cols = input_tensor.shape
    if rows > _DYNAMIC_QUANT_MAX_ROWS:
        return "prefill_shape"
    if n_cols <= 0 or n_cols > 16384 or input_tensor.numel() == 0:
        return "unsupported_size"
    if input_tensor.dtype != torch.bfloat16:
        return "unsupported_dtype"
    if not input_tensor.is_contiguous():
        return "non_contiguous"
    return None


def dynamic_quant(input_tensor: "torch.Tensor", ) -> tuple["torch.Tensor", "torch.Tensor"] | None:
    """Quantize decode-sized BF16 rows with one Gaudi2-native TPC node."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode == FastPathMode.OFF.value:
            return None
        if compile_mode == FastPathMode.HYBRID.value:
            return None
        rejection_reason = _dynamic_quant_rejection_reason(input_tensor)
        if rejection_reason is not None:
            if compile_mode == FastPathMode.STRICT.value:
                raise FastPathUnavailable("Gaudi Triton strict mode rejected dynamic_quant: "
                                          f"{rejection_reason}")
            return None
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            dynamic_quant as launch_dynamic_quant, )

        return launch_dynamic_quant(input_tensor)

    if _mode() is FastPathMode.OFF:
        _counters["vendor.dynamic_quant.off"] += 1
        return None
    rejection_reason = _dynamic_quant_rejection_reason(input_tensor)
    if rejection_reason is not None:
        return _reject_dynamic_quant(rejection_reason)
    if _mode() is FastPathMode.HYBRID:
        _counters["vendor.dynamic_quant.performance_gate"] += 1
        return None
    if not prepare_if_enabled():
        return _reject_dynamic_quant("initialization")

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            dynamic_quant as launch_dynamic_quant, )

        output = launch_dynamic_quant(input_tensor)
        _counters["triton.dynamic_quant"] += 1
        return output
    except Exception as exc:
        if _mode() is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton dynamic quantization compilation or launch failed") from exc
        _warn_once(
            "dynamic_quant.launch",
            "Gaudi Triton dynamic quantization failed "
            f"({exc}); using the HPU vendor path",
        )
        _counters["fallback.dynamic_quant.launch"] += 1
        return None


def _reject_silu_and_mul_dynamic_quant(reason: str) -> None:
    _counters[f"fallback.silu_and_mul_dynamic_quant.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable("Gaudi Triton strict mode rejected "
                                  f"silu_and_mul_dynamic_quant: {reason}")
    return None


def _silu_and_mul_dynamic_quant_rejection_reason(input_tensor: torch.Tensor, ) -> str | None:
    if input_tensor.device.type != "hpu":
        return "non_hpu_tensor"
    if (input_tensor.ndim != 2 or input_tensor.shape[0] <= 0 or input_tensor.shape[1] <= 0
            or input_tensor.shape[1] % 2):
        return "shape_mismatch"
    rows, input_width = input_tensor.shape
    n_cols = input_width // 2
    if (rows > (1 << 32) - 1 or n_cols <= 128 or n_cols > _SILU_DYNAMIC_QUANT_MAX_COLS or input_tensor.numel() == 0):
        return "unsupported_size"
    if input_tensor.dtype != torch.bfloat16:
        return "unsupported_dtype"
    if not input_tensor.is_contiguous():
        return "non_contiguous"
    return None


def silu_and_mul_dynamic_quant(input_tensor: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse SwiGLU and per-row E4M3 quantization into one TPC node."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode == FastPathMode.OFF.value:
            return None
        if compile_mode == FastPathMode.HYBRID.value:
            return None
        rejection_reason = _silu_and_mul_dynamic_quant_rejection_reason(input_tensor)
        if rejection_reason is not None:
            if compile_mode == FastPathMode.STRICT.value:
                raise FastPathUnavailable("Gaudi Triton strict mode rejected "
                                          "silu_and_mul_dynamic_quant: "
                                          f"{rejection_reason}")
            return None
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            silu_and_mul_dynamic_quant as launch_fused, )

        return launch_fused(input_tensor)

    mode = _mode()
    if mode is FastPathMode.OFF:
        _counters["vendor.silu_and_mul_dynamic_quant.off"] += 1
        return None
    rejection_reason = _silu_and_mul_dynamic_quant_rejection_reason(input_tensor)
    if rejection_reason is not None:
        return _reject_silu_and_mul_dynamic_quant(rejection_reason)
    if mode is FastPathMode.HYBRID:
        _counters["vendor.silu_and_mul_dynamic_quant.performance_gate"] += 1
        return None
    if not prepare_if_enabled():
        return _reject_silu_and_mul_dynamic_quant("initialization")

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            silu_and_mul_dynamic_quant as launch_fused, )

        output = launch_fused(input_tensor)
        _counters["triton.silu_and_mul_dynamic_quant"] += 1
        return output
    except Exception as exc:
        if mode is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton fused SiLU dynamic quantization "
                                      "compilation or launch failed") from exc
        _warn_once(
            "silu_and_mul_dynamic_quant.launch",
            "Gaudi Triton fused SiLU dynamic quantization failed "
            f"({exc}); using the HPU vendor path",
        )
        _counters["fallback.silu_and_mul_dynamic_quant.launch"] += 1
        return None


def _reject_silu_and_mul(reason: str) -> None:
    _counters[f"fallback.silu_and_mul.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable(f"Gaudi Triton strict mode rejected silu_and_mul: {reason}")
    _warn_once(
        f"silu_and_mul.{reason}",
        f"Gaudi Triton SiLU-and-mul unavailable ({reason}); using the HPU vendor path",
    )
    return None


def _silu_and_mul_rejection_reason(input_tensor: "torch.Tensor") -> str | None:
    if input_tensor.device.type != "hpu":
        return "non_hpu_tensor"
    if input_tensor.ndim == 0 or input_tensor.shape[-1] <= 0 or input_tensor.shape[-1] % 2:
        return "shape_mismatch"
    n_cols = input_tensor.shape[-1] // 2
    if n_cols <= 128 or n_cols > 65536 or input_tensor.numel() == 0:
        return "unsupported_size"
    if input_tensor.dtype != torch.bfloat16:
        return "unsupported_dtype"
    if not input_tensor.is_contiguous():
        return "non_contiguous"
    return None


def silu_and_mul(input_tensor: "torch.Tensor") -> "torch.Tensor" | None:
    """Run the Gaudi2 BF16 fused SwiGLU activation fast path."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode == FastPathMode.OFF.value:
            return None
        if compile_mode == FastPathMode.HYBRID.value:
            return None
        rejection_reason = _silu_and_mul_rejection_reason(input_tensor)
        if rejection_reason is not None:
            if compile_mode == FastPathMode.STRICT.value:
                raise FastPathUnavailable(f"Gaudi Triton strict mode rejected silu_and_mul: {rejection_reason}")
            return None
        from vllm_gaudi.ops.triton_gaudi.kernels import silu_and_mul as launch_silu_and_mul

        return launch_silu_and_mul(input_tensor)

    mode = _mode()
    if mode is FastPathMode.OFF:
        _counters["vendor.silu_and_mul.off"] += 1
        return None
    rejection_reason = _silu_and_mul_rejection_reason(input_tensor)
    if rejection_reason is not None:
        return _reject_silu_and_mul(rejection_reason)
    if mode is FastPathMode.HYBRID:
        _counters["vendor.silu_and_mul.performance_gate"] += 1
        return None
    if not prepare_if_enabled():
        return _reject_silu_and_mul("initialization")

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import silu_and_mul as launch_silu_and_mul

        output = launch_silu_and_mul(input_tensor)
        _counters["triton.silu_and_mul"] += 1
        return output
    except Exception as exc:
        if _mode() is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton SiLU-and-mul compilation or launch failed") from exc
        _warn_once(
            "silu_and_mul.launch",
            f"Gaudi Triton SiLU-and-mul failed ({exc}); using the HPU vendor path",
        )
        _counters["fallback.silu_and_mul.launch"] += 1
        return None


def _reject_gdn_decode_packed(reason: str) -> None:
    _counters[f"fallback.gdn_decode_packed.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable(f"Gaudi Triton strict mode rejected gdn_decode_packed: {reason}")
    _warn_once(
        f"gdn_decode_packed.{reason}",
        f"Gaudi Triton packed GDN decode unavailable ({reason}); using the HPU vendor path",
    )
    return None


def _gdn_decode_packed_rejection_reason(
    state_cache: "torch.Tensor",
    packed_qkv: "torch.Tensor",
    gate_a: "torch.Tensor",
    gate_b: "torch.Tensor",
    a_log: "torch.Tensor",
    dt_bias: "torch.Tensor",
    state_indices: "torch.Tensor",
) -> str | None:
    tensors = (
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
    )
    if any(tensor.device.type != "hpu" for tensor in tensors):
        return "non_hpu_tensor"
    if any(tensor.device != state_cache.device for tensor in tensors[1:]):
        return "device_mismatch"
    if state_cache.ndim != 4 or tuple(state_cache.shape[1:]) != (48, 128, 128):
        return "state_shape"
    if packed_qkv.ndim != 2 or packed_qkv.shape[0] <= 0 or packed_qkv.shape[1] != 10240:
        return "packed_shape"
    batch = packed_qkv.shape[0]
    if gate_a.shape != (batch, 48) or gate_b.shape != gate_a.shape:
        return "gate_shape"
    if a_log.shape != (48, ) or dt_bias.shape != (48, ) or state_indices.shape != (batch, ):
        return "parameter_shape"
    expected_dtypes = (
        torch.float32,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.int32,
    )
    if any(tensor.dtype != dtype for tensor, dtype in zip(tensors, expected_dtypes)):
        return "unsupported_dtype"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "non_contiguous"
    return None


def gdn_decode_packed(
    state_cache: "torch.Tensor",
    packed_qkv: "torch.Tensor",
    gate_a: "torch.Tensor",
    gate_b: "torch.Tensor",
    a_log: "torch.Tensor",
    dt_bias: "torch.Tensor",
    state_indices: "torch.Tensor",
) -> "torch.Tensor" | None:
    """Run fused Qwen3.5 single-token GDN decode and mutate its state cache."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode == FastPathMode.OFF.value:
            return None
        if compile_mode == FastPathMode.HYBRID.value:
            return None
        rejection_reason = _gdn_decode_packed_rejection_reason(
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
        )
        if rejection_reason is not None:
            if compile_mode == FastPathMode.STRICT.value:
                raise FastPathUnavailable("Gaudi Triton strict mode rejected gdn_decode_packed: "
                                          f"{rejection_reason}")
            return None
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            gdn_decode_packed as launch_gdn_decode_packed, )

        return launch_gdn_decode_packed(
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
        )

    if _mode() is FastPathMode.OFF:
        _counters["vendor.gdn_decode_packed.off"] += 1
        return None
    if _mode() is FastPathMode.HYBRID:
        _counters["vendor.gdn_decode_packed.performance_gate"] += 1
        return None
    rejection_reason = _gdn_decode_packed_rejection_reason(
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
    )
    if rejection_reason is not None:
        return _reject_gdn_decode_packed(rejection_reason)
    if not prepare_if_enabled():
        return _reject_gdn_decode_packed("initialization")

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            gdn_decode_packed as launch_gdn_decode_packed, )

        output = launch_gdn_decode_packed(
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
        )
        _counters["triton.gdn_decode_packed"] += 1
        return output
    except Exception as exc:
        if _mode() is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton gdn_decode_packed compilation or launch failed") from exc
        _warn_once(
            "gdn_decode_packed.launch",
            f"Gaudi Triton packed GDN decode failed ({exc}); using the HPU vendor path",
        )
        _counters["fallback.gdn_decode_packed.launch"] += 1
        return None


def _reject_gdn_decode_conv_packed(reason: str) -> None:
    _counters[f"fallback.gdn_decode_conv_packed.{reason}"] += 1
    if _mode() is FastPathMode.STRICT:
        raise FastPathUnavailable("Gaudi Triton strict mode rejected gdn_decode_conv_packed: "
                                  f"{reason}")
    _warn_once(
        f"gdn_decode_conv_packed.{reason}",
        "Gaudi Triton fused causal-conv + GDN unavailable "
        f"({reason}); using the HPU vendor path",
    )
    return None


def _gdn_decode_conv_packed_rejection_reason(
    conv_state: "torch.Tensor",
    state_cache: "torch.Tensor",
    packed_qkv: "torch.Tensor",
    gate_a: "torch.Tensor",
    gate_b: "torch.Tensor",
    a_log: "torch.Tensor",
    dt_bias: "torch.Tensor",
    state_indices: "torch.Tensor",
    conv_weight_t: "torch.Tensor",
) -> str | None:
    tensors = (
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
    )
    if any(tensor.device.type != "hpu" for tensor in tensors):
        return "non_hpu_tensor"
    if any(tensor.device != conv_state.device for tensor in tensors[1:]):
        return "device_mismatch"
    if (conv_state.ndim != 3 or conv_state.shape[0] <= 0 or tuple(conv_state.shape[1:]) != (3, 10240)):
        return "conv_state_shape"
    if (state_cache.ndim != 4 or state_cache.shape[0] <= 0 or tuple(state_cache.shape[1:]) != (48, 128, 128)):
        return "state_shape"
    if (packed_qkv.ndim != 2 or packed_qkv.shape[0] <= 0 or packed_qkv.shape[1] != 10240):
        return "packed_shape"
    batch = packed_qkv.shape[0]
    if gate_a.shape != (batch, 48) or gate_b.shape != gate_a.shape:
        return "gate_shape"
    if (a_log.shape != (48, ) or dt_bias.shape != (48, ) or state_indices.shape != (batch, )
            or conv_weight_t.shape != (4, 10240)):
        return "parameter_shape"
    expected_dtypes = (
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.int32,
        torch.bfloat16,
    )
    if any(tensor.dtype != dtype for tensor, dtype in zip(tensors, expected_dtypes)):
        return "unsupported_dtype"
    if any(not tensor.is_contiguous() for tensor in tensors):
        return "non_contiguous"
    return None


def gdn_decode_conv_packed(
    conv_state: "torch.Tensor",
    state_cache: "torch.Tensor",
    packed_qkv: "torch.Tensor",
    gate_a: "torch.Tensor",
    gate_b: "torch.Tensor",
    a_log: "torch.Tensor",
    dt_bias: "torch.Tensor",
    state_indices: "torch.Tensor",
    conv_weight_t: "torch.Tensor",
) -> "torch.Tensor" | None:
    """Fuse width-4 causal-conv, SiLU, and Qwen3.5 recurrent decode."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode in (
                FastPathMode.OFF.value,
                FastPathMode.HYBRID.value,
        ):
            return None
        rejection_reason = _gdn_decode_conv_packed_rejection_reason(
            conv_state,
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
            conv_weight_t,
        )
        if rejection_reason is not None:
            raise FastPathUnavailable("Gaudi Triton strict mode rejected gdn_decode_conv_packed: "
                                      f"{rejection_reason}")
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            gdn_decode_conv_packed as launch_gdn_decode_conv_packed, )

        return launch_gdn_decode_conv_packed(
            conv_state,
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
            conv_weight_t,
        )

    mode = _mode()
    if mode is FastPathMode.OFF:
        _counters["vendor.gdn_decode_conv_packed.off"] += 1
        return None
    if mode is FastPathMode.HYBRID:
        _counters["vendor.gdn_decode_conv_packed.performance_gate"] += 1
        return None
    rejection_reason = _gdn_decode_conv_packed_rejection_reason(
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
    )
    if rejection_reason is not None:
        return _reject_gdn_decode_conv_packed(rejection_reason)
    if not prepare_if_enabled():
        return _reject_gdn_decode_conv_packed("initialization")

    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            gdn_decode_conv_packed as launch_gdn_decode_conv_packed, )

        output = launch_gdn_decode_conv_packed(
            conv_state,
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
            conv_weight_t,
        )
        _counters["triton.gdn_decode_conv_packed"] += 1
        return output
    except Exception as exc:
        raise FastPathUnavailable("Gaudi Triton gdn_decode_conv_packed compilation or launch failed") from exc


def gdn_decode_conv_split_packed(
    conv_state: "torch.Tensor",
    state_cache: "torch.Tensor",
    packed_qkv: "torch.Tensor",
    gate_a: "torch.Tensor",
    gate_b: "torch.Tensor",
    a_log: "torch.Tensor",
    dt_bias: "torch.Tensor",
    state_indices: "torch.Tensor",
    conv_weight_t: "torch.Tensor",
) -> "torch.Tensor" | None:
    """Run Q/K conv then tile-parallel fused value-conv + GDN."""
    if torch.compiler.is_compiling():
        compile_mode = _compile_fast_path_mode()
        if compile_mode == FastPathMode.OFF.value:
            return None
        batch = packed_qkv.shape[0]
        if (compile_mode == FastPathMode.HYBRID.value and batch != _GDN_CONV_SPLIT_HYBRID_BATCH):
            return None
        if batch not in _GDN_CONV_SPLIT_GRAPH_BATCHES:
            raise FastPathUnavailable("Gaudi Triton strict mode rejected "
                                      "gdn_decode_conv_split_packed: unsupported_graph_batch")
        rejection_reason = _gdn_decode_conv_packed_rejection_reason(
            conv_state,
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
            conv_weight_t,
        )
        if rejection_reason is not None:
            if compile_mode == FastPathMode.STRICT.value:
                raise FastPathUnavailable("Gaudi Triton strict mode rejected "
                                          "gdn_decode_conv_split_packed: "
                                          f"{rejection_reason}")
            return None
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            gdn_decode_conv_split_packed as launch_split_gdn, )

        return launch_split_gdn(
            conv_state,
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
            conv_weight_t,
        )

    mode = _mode()
    if mode is FastPathMode.OFF:
        _counters["vendor.gdn_decode_conv_split_packed.off"] += 1
        return None
    if (mode is FastPathMode.HYBRID and packed_qkv.shape[0] != _GDN_CONV_SPLIT_HYBRID_BATCH):
        _counters["vendor.gdn_decode_conv_split_packed.performance_gate"] += 1
        return None
    rejection_reason = _gdn_decode_conv_packed_rejection_reason(
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
    )
    if rejection_reason is not None:
        if mode is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton strict mode rejected "
                                      "gdn_decode_conv_split_packed: "
                                      f"{rejection_reason}")
        _counters[f"fallback.gdn_decode_conv_split_packed.{rejection_reason}"] += 1
        return None
    if not prepare_if_enabled():
        if mode is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton split fused GDN cannot initialize")
        _counters["fallback.gdn_decode_conv_split_packed.initialization"] += 1
        return None
    try:
        from vllm_gaudi.ops.triton_gaudi.kernels import (
            gdn_decode_conv_split_packed as launch_split_gdn, )

        output = launch_split_gdn(
            conv_state,
            state_cache,
            packed_qkv,
            gate_a,
            gate_b,
            a_log,
            dt_bias,
            state_indices,
            conv_weight_t,
        )
        _counters["triton.gdn_decode_conv_split_packed"] += 1
        return output
    except Exception as exc:
        if mode is FastPathMode.STRICT:
            raise FastPathUnavailable("Gaudi Triton split fused GDN compilation or launch failed") from exc
        _warn_once(
            "gdn_decode_conv_split_packed.launch",
            "Gaudi Triton split fused GDN failed "
            f"({exc}); using the HPU vendor path",
        )
        _counters["fallback.gdn_decode_conv_split_packed.launch"] += 1
        return None


def diagnostics() -> dict[str, object]:
    """Return stable, side-effect-free state for CI and serving diagnostics."""
    return {
        "mode": _mode().value,
        "prepared": _prepared,
        "available": _available,
        "prepare_error": _prepare_error,
        "counters": dict(sorted(_counters.items())),
    }


def _reset_for_tests() -> None:
    global _available, _mode_value, _prepare_error, _prepared
    with _lock:
        _prepared = False
        _available = False
        _mode_value = None
        _prepare_error = None
        _warned.clear()
        _counters.clear()
