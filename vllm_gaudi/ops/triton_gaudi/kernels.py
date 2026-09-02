# SPDX-License-Identifier: Apache-2.0

"""Triton source for Gaudi2-native vLLM kernels.

This module is imported only after the runtime has verified that the Gaudi
backend and matching Bridge launch ABI are installed.
"""

import functools
import json
import os
import threading

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.gaudi import GaudiConfig, GaudiKernelArtifactV1


_graph_artifact_lock = threading.Lock()
_graph_artifact_handles: dict[tuple[str, int], int] = {}
QK_CONV_TILE = 256


@functools.lru_cache(maxsize=None)
def _elementwise_schedule() -> dict[str, object]:
    return GaudiConfig(
        unroll=1,
        pipeline_depth=1,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=16)
def _rms_norm_schedule(block_size: int) -> dict[str, object]:
    return GaudiConfig(
        unroll=1,
        pipeline_depth=1,
        engine="tpc",
        mode="strict",
        vlm_budget_bytes=block_size * 2,
    ).as_backend_options()


@functools.lru_cache(maxsize=None)
def _silu_and_mul_schedule() -> dict[str, object]:
    return GaudiConfig(
        unroll=1,
        pipeline_depth=1,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=None)
def _dynamic_quant_schedule() -> dict[str, object]:
    return GaudiConfig(
        unroll=1,
        pipeline_depth=1,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=4)
def _gdn_decode_schedule(value_tile: int) -> dict[str, object]:
    return GaudiConfig(
        unroll=value_tile,
        pipeline_depth=2,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=None)
def _gdn_decode_conv_schedule() -> dict[str, object]:
    return GaudiConfig(
        unroll=1,
        pipeline_depth=2,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=None)
def _gdn_qk_conv_schedule() -> dict[str, object]:
    return GaudiConfig(
        unroll=1,
        pipeline_depth=2,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=4)
def _gdn_value_conv_schedule(value_tile: int) -> dict[str, object]:
    return GaudiConfig(
        unroll=value_tile,
        pipeline_depth=2,
        engine="tpc",
        mode="strict",
    ).as_backend_options()


@functools.lru_cache(maxsize=32)
def _compile_fused_add_rms_norm(n_cols: int) -> tuple[GaudiKernelArtifactV1, int]:
    block_size = triton.next_power_of_2(n_cols)
    source = triton.compiler.ASTSource(
        fn=_fused_add_rms_norm_kernel,
        signature={
            "hidden_states": "*bf16",
            "residual": "*bf16",
            "weight": "*bf16",
            "output": "*bf16",
            "residual_output": "*bf16",
            "epsilon": "fp32",
            "N_COLS": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"N_COLS": n_cols, "BLOCK_SIZE": block_size},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_rms_norm_schedule(block_size),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"]), block_size


@functools.lru_cache(maxsize=128)
def _compile_silu_and_mul(n_cols: int, block_size: int) -> GaudiKernelArtifactV1:
    source = triton.compiler.ASTSource(
        fn=_silu_and_mul_kernel,
        signature={
            "input_ptr": "*bf16",
            "output_ptr": "*bf16",
            "N_COLS": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"N_COLS": n_cols, "BLOCK_SIZE": block_size},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_silu_and_mul_schedule(),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"])


@functools.lru_cache(maxsize=32)
def _compile_dynamic_quant(n_cols: int) -> tuple[GaudiKernelArtifactV1, int]:
    block_size = triton.next_power_of_2(n_cols)
    source = triton.compiler.ASTSource(
        fn=_dynamic_quant_kernel,
        signature={
            "input_ptr": "*bf16",
            "output_ptr": "*fp8e4nv",
            "scale_ptr": "*fp32",
            "N_COLS": "constexpr",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"N_COLS": n_cols, "BLOCK_SIZE": block_size},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_dynamic_quant_schedule(),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"]), block_size


@functools.lru_cache(maxsize=4)
def _compile_gdn_decode_packed(value_tile: int) -> GaudiKernelArtifactV1:
    source = triton.compiler.ASTSource(
        fn=_gdn_decode_packed_kernel,
        signature={
            "state_cache": "*fp32",
            "packed_qkv": "*bf16",
            "gate_a": "*bf16",
            "gate_b": "*bf16",
            "a_log": "*fp32",
            "dt_bias": "*fp32",
            "state_indices": "*i32",
            "output": "*bf16",
            "state_slots": "i32",
            "VALUE_TILE": "constexpr",
        },
        constexprs={"VALUE_TILE": value_tile},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_gdn_decode_schedule(value_tile),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"])


@functools.lru_cache(maxsize=1)
def _compile_gdn_decode_conv_packed() -> GaudiKernelArtifactV1:
    source = triton.compiler.ASTSource(
        fn=_gdn_decode_conv_packed_kernel,
        signature={
            "conv_state": "*bf16",
            "state_cache": "*fp32",
            "packed_qkv": "*bf16",
            "gate_a": "*bf16",
            "gate_b": "*bf16",
            "a_log": "*fp32",
            "dt_bias": "*fp32",
            "state_indices": "*i32",
            "conv_weight_t": "*bf16",
            "output": "*bf16",
            "conv_slots": "i32",
            "state_slots": "i32",
        },
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_gdn_decode_conv_schedule(),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"])


@functools.lru_cache(maxsize=1)
def _compile_gdn_qk_conv_packed() -> GaudiKernelArtifactV1:
    source = triton.compiler.ASTSource(
        fn=_gdn_qk_conv_packed_kernel,
        signature={
            "conv_state": "*bf16",
            "packed_qkv": "*bf16",
            "state_indices": "*i32",
            "conv_weight_t": "*bf16",
            "qk_output": "*bf16",
            "conv_slots": "i32",
            "QK_TILE": "constexpr",
        },
        constexprs={"QK_TILE": QK_CONV_TILE},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_gdn_qk_conv_schedule(),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"])


@functools.lru_cache(maxsize=4)
def _compile_gdn_decode_value_conv_packed(
    value_tile: int,
) -> GaudiKernelArtifactV1:
    source = triton.compiler.ASTSource(
        fn=_gdn_decode_value_conv_packed_kernel,
        signature={
            "conv_state": "*bf16",
            "state_cache": "*fp32",
            "qk_conv": "*bf16",
            "packed_qkv": "*bf16",
            "gate_a": "*bf16",
            "gate_b": "*bf16",
            "a_log": "*fp32",
            "dt_bias": "*fp32",
            "state_indices": "*i32",
            "conv_weight_t": "*bf16",
            "output": "*bf16",
            "conv_slots": "i32",
            "state_slots": "i32",
            "VALUE_TILE": "constexpr",
        },
        constexprs={"VALUE_TILE": value_tile},
    )
    compiled = triton.compile(
        source,
        target=GPUTarget("gaudi", "gaudi2"),
        options=_gdn_value_conv_schedule(value_tile),
    )
    return GaudiKernelArtifactV1.from_bytes(compiled.asm["gabin"])


def _materialize_graph_artifact(artifact: GaudiKernelArtifactV1, device: int) -> None:
    key = (artifact.artifact_hash, device)
    if key in _graph_artifact_handles:
        return
    from habana_frameworks.torch import _hpu_C

    manifest = json.dumps(artifact.manifest, sort_keys=True, separators=(",", ":"))
    with _graph_artifact_lock:
        if key not in _graph_artifact_handles:
            handle = _hpu_C._triton_gaudi_register_artifact(
                artifact.artifact_hash,
                artifact.elf,
                manifest,
                device,
            )
            _graph_artifact_handles[key] = handle


@functools.lru_cache(maxsize=32)
def _prepare_fused_add_rms_norm_cached(n_cols: int) -> tuple[str, int]:
    artifact, block_size = _compile_fused_add_rms_norm(n_cols)
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash, block_size


@torch.compiler.assume_constant_result
def _prepare_fused_add_rms_norm(n_cols: int) -> tuple[str, int]:
    return _prepare_fused_add_rms_norm_cached(n_cols)


@functools.lru_cache(maxsize=128)
def _prepare_silu_and_mul_cached(n_cols: int, block_size: int) -> str:
    artifact = _compile_silu_and_mul(n_cols, block_size)
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash


@torch.compiler.assume_constant_result
def _prepare_silu_and_mul(n_cols: int, block_size: int) -> str:
    return _prepare_silu_and_mul_cached(n_cols, block_size)


@functools.lru_cache(maxsize=32)
def _prepare_dynamic_quant_cached(n_cols: int) -> tuple[str, int]:
    artifact, block_size = _compile_dynamic_quant(n_cols)
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash, block_size


@torch.compiler.assume_constant_result
def _prepare_dynamic_quant(n_cols: int) -> tuple[str, int]:
    return _prepare_dynamic_quant_cached(n_cols)


@functools.lru_cache(maxsize=4)
def _prepare_gdn_decode_packed_cached(value_tile: int) -> str:
    artifact = _compile_gdn_decode_packed(value_tile)
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash


@torch.compiler.assume_constant_result
def _prepare_gdn_decode_packed(value_tile: int) -> str:
    return _prepare_gdn_decode_packed_cached(value_tile)


@functools.lru_cache(maxsize=1)
def _prepare_gdn_decode_conv_packed_cached() -> str:
    artifact = _compile_gdn_decode_conv_packed()
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash


@torch.compiler.assume_constant_result
def _prepare_gdn_decode_conv_packed() -> str:
    return _prepare_gdn_decode_conv_packed_cached()


@functools.lru_cache(maxsize=1)
def _prepare_gdn_qk_conv_packed_cached() -> str:
    artifact = _compile_gdn_qk_conv_packed()
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash


@torch.compiler.assume_constant_result
def _prepare_gdn_qk_conv_packed() -> str:
    return _prepare_gdn_qk_conv_packed_cached()


@functools.lru_cache(maxsize=4)
def _prepare_gdn_decode_value_conv_packed_cached(value_tile: int) -> str:
    artifact = _compile_gdn_decode_value_conv_packed(value_tile)
    _materialize_graph_artifact(artifact, torch.hpu.current_device())
    return artifact.artifact_hash


@torch.compiler.assume_constant_result
def _prepare_gdn_decode_value_conv_packed(value_tile: int) -> str:
    return _prepare_gdn_decode_value_conv_packed_cached(value_tile)


@torch.compiler.assume_constant_result
def _gdn_decode_value_tile() -> int:
    value = int(os.environ.get("VLLM_HPU_TRITON_GDN_VALUE_TILE", "16"))
    if value not in (16, 32, 64, 128):
        raise ValueError(
            "VLLM_HPU_TRITON_GDN_VALUE_TILE must be 16, 32, 64, or 128")
    return value


@torch.compiler.assume_constant_result
def _silu_and_mul_block_size(n_cols: int) -> int:
    value = int(os.environ.get("VLLM_HPU_TRITON_SILU_BLOCK_SIZE", "128"))
    if value < 128 or value > 1024 or value & (value - 1):
        raise ValueError("VLLM_HPU_TRITON_SILU_BLOCK_SIZE must be a power of two in [128, 1024]")
    while value >= n_cols and value > 128:
        value //= 2
    if value >= n_cols:
        raise ValueError("Gaudi Triton SiLU-and-mul requires an output width greater than 128")
    return value


@triton.jit
def _vector_add_kernel(lhs, rhs, output, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    lhs_values = tl.load(lhs + offsets, mask=mask, other=0.0)
    rhs_values = tl.load(rhs + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, lhs_values + rhs_values, mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    hidden_states,
    residual,
    weight,
    output,
    residual_output,
    epsilon,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N_COLS
    offsets = row * N_COLS + columns
    hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0).to(tl.float32)
    previous = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = (hidden + previous).to(tl.bfloat16)
    tl.store(residual_output + offsets, summed, mask=mask)
    summed_fp32 = summed.to(tl.float32)
    variance = tl.sum(summed_fp32 * summed_fp32, axis=0) / N_COLS
    reciprocal_rms = tl.rsqrt(variance + epsilon)
    scale = tl.load(weight + columns, mask=mask, other=0.0).to(tl.float32)
    normalized = summed_fp32 * reciprocal_rms * scale
    tl.store(output + offsets, normalized, mask=mask)


@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    columns = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = columns < N_COLS
    input_row = row * (2 * N_COLS)
    output_row = row * N_COLS
    gate_offsets = input_row + columns
    up_offsets = gate_offsets + N_COLS
    output_offsets = output_row + columns
    gate = tl.load(input_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
    result = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + output_offsets, result, mask=mask)


@triton.jit
def _dynamic_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < N_COLS
    offsets = row * N_COLS + columns
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32)
    abs_max = tl.max(tl.abs(values), axis=0)
    scale = (abs_max + 1.0e-8) / 240.0
    tl.store(scale_ptr + row, scale)
    quantized = values / scale
    tl.store(output_ptr + offsets, quantized, mask=mask)


@triton.jit
def _gdn_decode_packed_kernel(
    state_cache,
    packed_qkv,
    gate_a,
    gate_b,
    a_log,
    dt_bias,
    state_indices,
    output,
    state_slots,
    VALUE_TILE: tl.constexpr,
):
    """Qwen3.5 single-token recurrent GDN update.

    The logical Triton program owns VALUE_TILE value rows for one value head
    and one batch item.  Q/K stay in their compact 16-head packed layout; the
    backend maps each group of three value heads to one Q/K head without a
    materialized repeat.
    """
    value_tile = tl.program_id(0)
    value_head = tl.program_id(1)
    batch = tl.program_id(2)
    key_head = value_head // 3
    key_offsets = tl.arange(0, 128)

    packed_row = batch * 10240
    q_offsets = packed_row + key_head * 128 + key_offsets
    k_offsets = packed_row + 2048 + key_head * 128 + key_offsets
    q = tl.load(packed_qkv + q_offsets).to(tl.float32)
    k = tl.load(packed_qkv + k_offsets).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6) * 0.08838834764831845
    k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

    gate_offset = batch * 48 + value_head
    gate_x = tl.load(gate_a + gate_offset).to(tl.float32)
    gate_x += tl.load(dt_bias + value_head)
    softplus = tl.where(gate_x <= 20.0, tl.log(1.0 + tl.exp(gate_x)), gate_x)
    decay = tl.exp(-tl.exp(tl.load(a_log + value_head)) * softplus)
    beta = tl.sigmoid(tl.load(gate_b + gate_offset).to(tl.float32))

    raw_slot = tl.load(state_indices + batch)
    state_slot = ((raw_slot % state_slots) + state_slots) % state_slots
    value_start = value_tile * VALUE_TILE
    for value_offset in tl.static_range(0, VALUE_TILE):
        value_row = value_start + value_offset
        state_offsets = (((state_slot * 48 + value_head) * 128 + value_row) * 128 + key_offsets)
        state = tl.load(state_cache + state_offsets)
        state *= decay
        projection = tl.sum(state * k, axis=0)
        value = tl.load(packed_qkv + packed_row + 4096 + value_head * 128 + value_row).to(tl.float32)
        delta = (value - projection) * beta
        state += delta * k
        result = tl.sum(state * q, axis=0)
        tl.store(state_cache + state_offsets, state)
        tl.store(output + (batch * 48 + value_head) * 128 + value_row, result)


@triton.jit
def _gdn_decode_conv_packed_kernel(
    conv_state,
    state_cache,
    packed_qkv,
    gate_a,
    gate_b,
    a_log,
    dt_bias,
    state_indices,
    conv_weight_t,
    output,
    conv_slots,
    state_slots,
):
    """Fuse width-4 causal convolution with Qwen3.5 GDN decode.

    One program owns a key head and its three value heads. This is the
    dependency-safe granularity: every compact Q/K convolution channel and
    every value channel has exactly one writer, so both mutable caches can be
    updated without an inter-program barrier.
    """
    key_head = tl.program_id(0)
    batch = tl.program_id(1)
    key_offsets = tl.arange(0, 128)
    packed_row = batch * 10240
    raw_slot = tl.load(state_indices + batch)
    conv_slot = ((raw_slot % conv_slots) + conv_slots) % conv_slots
    state_slot = ((raw_slot % state_slots) + state_slots) % state_slots

    q_channels = key_head * 128 + key_offsets
    k_channels = 2048 + q_channels
    q_raw = tl.load(packed_qkv + packed_row + q_channels)
    k_raw = tl.load(packed_qkv + packed_row + k_channels)

    q_h0 = tl.load(conv_state + (conv_slot * 3 + 0) * 10240 + q_channels)
    q_h1 = tl.load(conv_state + (conv_slot * 3 + 1) * 10240 + q_channels)
    q_h2 = tl.load(conv_state + (conv_slot * 3 + 2) * 10240 + q_channels)
    k_h0 = tl.load(conv_state + (conv_slot * 3 + 0) * 10240 + k_channels)
    k_h1 = tl.load(conv_state + (conv_slot * 3 + 1) * 10240 + k_channels)
    k_h2 = tl.load(conv_state + (conv_slot * 3 + 2) * 10240 + k_channels)

    q_conv = q_h0.to(tl.float32) * tl.load(conv_weight_t + 0 * 10240 + q_channels).to(tl.float32)
    q_conv += q_h1.to(tl.float32) * tl.load(conv_weight_t + 1 * 10240 + q_channels).to(tl.float32)
    q_conv += q_h2.to(tl.float32) * tl.load(conv_weight_t + 2 * 10240 + q_channels).to(tl.float32)
    q_conv += q_raw.to(tl.float32) * tl.load(conv_weight_t + 3 * 10240 + q_channels).to(tl.float32)
    k_conv = k_h0.to(tl.float32) * tl.load(conv_weight_t + 0 * 10240 + k_channels).to(tl.float32)
    k_conv += k_h1.to(tl.float32) * tl.load(conv_weight_t + 1 * 10240 + k_channels).to(tl.float32)
    k_conv += k_h2.to(tl.float32) * tl.load(conv_weight_t + 2 * 10240 + k_channels).to(tl.float32)
    k_conv += k_raw.to(tl.float32) * tl.load(conv_weight_t + 3 * 10240 + k_channels).to(tl.float32)
    q_rounded = q_conv.to(tl.bfloat16).to(tl.float32)
    k_rounded = k_conv.to(tl.bfloat16).to(tl.float32)
    q = (q_rounded * tl.sigmoid(q_rounded)).to(tl.bfloat16).to(tl.float32)
    k = (k_rounded * tl.sigmoid(k_rounded)).to(tl.bfloat16).to(tl.float32)

    tl.store(conv_state + (conv_slot * 3 + 0) * 10240 + q_channels, q_h1)
    tl.store(conv_state + (conv_slot * 3 + 1) * 10240 + q_channels, q_h2)
    tl.store(conv_state + (conv_slot * 3 + 2) * 10240 + q_channels, q_raw)
    tl.store(conv_state + (conv_slot * 3 + 0) * 10240 + k_channels, k_h1)
    tl.store(conv_state + (conv_slot * 3 + 1) * 10240 + k_channels, k_h2)
    tl.store(conv_state + (conv_slot * 3 + 2) * 10240 + k_channels, k_raw)

    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6) * 0.08838834764831845
    k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

    for head_group in tl.static_range(0, 3):
        value_head = key_head * 3 + head_group
        gate_offset = batch * 48 + value_head
        gate_x = tl.load(gate_a + gate_offset).to(tl.float32)
        gate_x += tl.load(dt_bias + value_head)
        softplus = tl.where(gate_x <= 20.0, tl.log(1.0 + tl.exp(gate_x)), gate_x)
        decay = tl.exp(-tl.exp(tl.load(a_log + value_head)) * softplus)
        beta = tl.sigmoid(tl.load(gate_b + gate_offset).to(tl.float32))

        for value_row in tl.range(0, 128):
            channel = 4096 + value_head * 128 + value_row
            raw_value = tl.load(packed_qkv + packed_row + channel)
            value_h0 = tl.load(conv_state + (conv_slot * 3 + 0) * 10240 + channel)
            value_h1 = tl.load(conv_state + (conv_slot * 3 + 1) * 10240 + channel)
            value_h2 = tl.load(conv_state + (conv_slot * 3 + 2) * 10240 + channel)
            value_conv = value_h0.to(tl.float32) * tl.load(conv_weight_t + 0 * 10240 + channel).to(tl.float32)
            value_conv += value_h1.to(tl.float32) * tl.load(conv_weight_t + 1 * 10240 + channel).to(tl.float32)
            value_conv += value_h2.to(tl.float32) * tl.load(conv_weight_t + 2 * 10240 + channel).to(tl.float32)
            value_conv += raw_value.to(tl.float32) * tl.load(conv_weight_t + 3 * 10240 + channel).to(tl.float32)
            value_rounded = value_conv.to(tl.bfloat16).to(tl.float32)
            value = (value_rounded * tl.sigmoid(value_rounded)).to(tl.bfloat16).to(tl.float32)

            tl.store(conv_state + (conv_slot * 3 + 0) * 10240 + channel, value_h1)
            tl.store(conv_state + (conv_slot * 3 + 1) * 10240 + channel, value_h2)
            tl.store(conv_state + (conv_slot * 3 + 2) * 10240 + channel, raw_value)

            state_offsets = (((state_slot * 48 + value_head) * 128 + value_row) * 128 + key_offsets)
            state = tl.load(state_cache + state_offsets)
            state *= decay
            projection = tl.sum(state * k, axis=0)
            delta = (value - projection) * beta
            state += delta * k
            result = tl.sum(state * q, axis=0)
            tl.store(state_cache + state_offsets, state)
            tl.store(output + (batch * 48 + value_head) * 128 + value_row, result)


@triton.jit
def _gdn_qk_conv_packed_kernel(
    conv_state,
    packed_qkv,
    state_indices,
    conv_weight_t,
    qk_output,
    conv_slots,
    QK_TILE: tl.constexpr,
):
    """Update compact Q/K width-4 convolution state and emit SiLU Q/K."""
    channel_block = tl.program_id(0)
    batch = tl.program_id(1)
    channels = channel_block * QK_TILE + tl.arange(0, QK_TILE)
    raw_slot = tl.load(state_indices + batch)
    conv_slot = ((raw_slot % conv_slots) + conv_slots) % conv_slots
    raw = tl.load(packed_qkv + batch * 10240 + channels)
    history0 = tl.load(
        conv_state + (conv_slot * 3 + 0) * 10240 + channels)
    history1 = tl.load(
        conv_state + (conv_slot * 3 + 1) * 10240 + channels)
    history2 = tl.load(
        conv_state + (conv_slot * 3 + 2) * 10240 + channels)
    conv = history0.to(tl.float32) * tl.load(
        conv_weight_t + 0 * 10240 + channels).to(tl.float32)
    conv += history1.to(tl.float32) * tl.load(
        conv_weight_t + 1 * 10240 + channels).to(tl.float32)
    conv += history2.to(tl.float32) * tl.load(
        conv_weight_t + 2 * 10240 + channels).to(tl.float32)
    conv += raw.to(tl.float32) * tl.load(
        conv_weight_t + 3 * 10240 + channels).to(tl.float32)
    rounded = conv.to(tl.bfloat16).to(tl.float32)
    activated = (rounded * tl.sigmoid(rounded)).to(tl.bfloat16)
    tl.store(conv_state + (conv_slot * 3 + 0) * 10240 + channels, history1)
    tl.store(conv_state + (conv_slot * 3 + 1) * 10240 + channels, history2)
    tl.store(conv_state + (conv_slot * 3 + 2) * 10240 + channels, raw)
    tl.store(qk_output + batch * 4096 + channels, activated)


@triton.jit
def _gdn_decode_value_conv_packed_kernel(
    conv_state,
    state_cache,
    qk_conv,
    packed_qkv,
    gate_a,
    gate_b,
    a_log,
    dt_bias,
    state_indices,
    conv_weight_t,
    output,
    conv_slots,
    state_slots,
    VALUE_TILE: tl.constexpr,
):
    """Fuse value convolution into the tile-parallel recurrent GDN update."""
    value_tile = tl.program_id(0)
    value_head = tl.program_id(1)
    batch = tl.program_id(2)
    key_head = value_head // 3
    key_offsets = tl.arange(0, 128)
    qk_row = batch * 4096
    q = tl.load(qk_conv + qk_row + key_head * 128 + key_offsets).to(
        tl.float32)
    k = tl.load(
        qk_conv + qk_row + 2048 + key_head * 128 + key_offsets).to(
            tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
    q *= 0.08838834764831845
    k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)

    gate_offset = batch * 48 + value_head
    gate_x = tl.load(gate_a + gate_offset).to(tl.float32)
    gate_x += tl.load(dt_bias + value_head)
    softplus = tl.where(
        gate_x <= 20.0,
        tl.log(1.0 + tl.exp(gate_x)),
        gate_x,
    )
    decay = tl.exp(-tl.exp(tl.load(a_log + value_head)) * softplus)
    beta = tl.sigmoid(tl.load(gate_b + gate_offset).to(tl.float32))

    raw_slot = tl.load(state_indices + batch)
    conv_slot = ((raw_slot % conv_slots) + conv_slots) % conv_slots
    state_slot = ((raw_slot % state_slots) + state_slots) % state_slots
    value_start = value_tile * VALUE_TILE
    packed_row = batch * 10240
    for value_offset in tl.static_range(0, VALUE_TILE):
        value_row = value_start + value_offset
        channel = 4096 + value_head * 128 + value_row
        raw_value = tl.load(packed_qkv + packed_row + channel)
        history0 = tl.load(
            conv_state + (conv_slot * 3 + 0) * 10240 + channel)
        history1 = tl.load(
            conv_state + (conv_slot * 3 + 1) * 10240 + channel)
        history2 = tl.load(
            conv_state + (conv_slot * 3 + 2) * 10240 + channel)
        value_conv = history0.to(tl.float32) * tl.load(
            conv_weight_t + 0 * 10240 + channel).to(tl.float32)
        value_conv += history1.to(tl.float32) * tl.load(
            conv_weight_t + 1 * 10240 + channel).to(tl.float32)
        value_conv += history2.to(tl.float32) * tl.load(
            conv_weight_t + 2 * 10240 + channel).to(tl.float32)
        value_conv += raw_value.to(tl.float32) * tl.load(
            conv_weight_t + 3 * 10240 + channel).to(tl.float32)
        value_rounded = value_conv.to(tl.bfloat16).to(tl.float32)
        value = (value_rounded * tl.sigmoid(value_rounded)).to(
            tl.bfloat16).to(tl.float32)
        tl.store(
            conv_state + (conv_slot * 3 + 0) * 10240 + channel,
            history1,
        )
        tl.store(
            conv_state + (conv_slot * 3 + 1) * 10240 + channel,
            history2,
        )
        tl.store(
            conv_state + (conv_slot * 3 + 2) * 10240 + channel,
            raw_value,
        )

        state_offsets = (
            ((state_slot * 48 + value_head) * 128 + value_row) * 128 +
            key_offsets)
        state = tl.load(state_cache + state_offsets)
        state *= decay
        projection = tl.sum(state * k, axis=0)
        delta = (value - projection) * beta
        state += delta * k
        result = tl.sum(state * q, axis=0)
        tl.store(state_cache + state_offsets, state)
        tl.store(
            output + (batch * 48 + value_head) * 128 + value_row,
            result,
        )


def vector_add(lhs: torch.Tensor, rhs: torch.Tensor, block_size: int) -> torch.Tensor:
    output = torch.empty_like(lhs)
    n_elements = lhs.numel()
    _vector_add_kernel[(triton.cdiv(n_elements, block_size), )](
        lhs,
        rhs,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
        backend_options=_elementwise_schedule(),
    )
    return output


def fused_add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_cols = hidden_states.shape[-1]
    if n_cols <= 0:
        raise ValueError("RMSNorm hidden size must be positive")
    artifact_hash, block_size = _prepare_fused_add_rms_norm(n_cols)
    output, residual_output = torch.ops.triton_gaudi.fused_add_rms_norm.default(
        hidden_states.view(-1),
        residual.view(-1),
        weight,
        artifact_hash,
        block_size,
        n_cols,
        hidden_states.numel() // n_cols,
        float(epsilon),
    )
    return output.view(hidden_states.shape), residual_output.view(residual.shape)


def fused_add_rms_norm_direct(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct recipe launch retained for ABI diagnostics and A/B testing."""
    n_cols = hidden_states.shape[-1]
    if n_cols <= 0:
        raise ValueError("RMSNorm hidden size must be positive")
    output = torch.empty_like(hidden_states)
    residual_output = torch.empty_like(residual)
    block_size = triton.next_power_of_2(n_cols)
    rows = hidden_states.numel() // n_cols
    _fused_add_rms_norm_kernel[(rows, )](
        hidden_states,
        residual,
        weight,
        output,
        residual_output,
        epsilon=float(epsilon),
        N_COLS=n_cols,
        BLOCK_SIZE=block_size,
        backend_options=_rms_norm_schedule(block_size),
    )
    return output, residual_output


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    input_width = input_tensor.shape[-1]
    if input_width <= 0 or input_width % 2:
        raise ValueError("SiLU-and-mul input width must be positive and even")
    n_cols = input_width // 2
    rows = input_tensor.numel() // input_width
    block_size = _silu_and_mul_block_size(n_cols)
    artifact_hash = _prepare_silu_and_mul(n_cols, block_size)
    output = torch.ops.triton_gaudi.silu_and_mul.default(
        input_tensor.view(rows, input_width),
        artifact_hash,
        block_size,
        n_cols,
        rows,
    )
    return output.view(*input_tensor.shape[:-1], n_cols)


def silu_and_mul_direct(input_tensor: torch.Tensor) -> torch.Tensor:
    """Direct recipe launch retained for ABI diagnostics and A/B testing."""
    input_width = input_tensor.shape[-1]
    if input_width <= 0 or input_width % 2:
        raise ValueError("SiLU-and-mul input width must be positive and even")
    n_cols = input_width // 2
    rows = input_tensor.numel() // input_width
    output = torch.empty((rows, n_cols), dtype=input_tensor.dtype, device=input_tensor.device)
    block_size = _silu_and_mul_block_size(n_cols)
    _silu_and_mul_kernel[(triton.cdiv(n_cols, block_size), rows)](
        input_tensor.view(rows, input_width),
        output,
        N_COLS=n_cols,
        BLOCK_SIZE=block_size,
        backend_options=_silu_and_mul_schedule(),
    )
    return output.view(*input_tensor.shape[:-1], n_cols)


def dynamic_quant(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch graph-native row-wise BF16 to Gaudi2 E4M3 quantization."""
    if input_tensor.ndim != 2:
        raise ValueError("Gaudi Triton dynamic quantization requires a 2D tensor")
    rows, n_cols = input_tensor.shape
    artifact_hash, block_size = _prepare_dynamic_quant(n_cols)
    quantized, scale = torch.ops.triton_gaudi.dynamic_quant.default(
        input_tensor.view(-1),
        artifact_hash,
        block_size,
        n_cols,
        rows,
    )
    return quantized.view(input_tensor.shape), scale.view(rows, 1)


def dynamic_quant_direct(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct recipe launch retained for quantization ABI diagnostics."""
    if input_tensor.ndim != 2:
        raise ValueError("Gaudi Triton dynamic quantization requires a 2D tensor")
    rows, n_cols = input_tensor.shape
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=input_tensor.device)
    block_size = triton.next_power_of_2(n_cols)
    _dynamic_quant_kernel[(rows, )](
        input_tensor,
        output,
        scale,
        N_COLS=n_cols,
        BLOCK_SIZE=block_size,
        backend_options=_dynamic_quant_schedule(),
    )
    return output, scale


def gdn_decode_packed(
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Launch the graph-native, state-mutating Qwen3.5 decode kernel."""
    value_tile = _gdn_decode_value_tile()
    artifact_hash = _prepare_gdn_decode_packed(value_tile)
    return torch.ops.triton_gaudi.gdn_decode_packed.default(
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        artifact_hash,
        value_tile,
    )


def gdn_decode_packed_direct(
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Direct recipe launch retained for mutation and scheduling diagnostics."""
    value_tile = _gdn_decode_value_tile()
    batch = packed_qkv.shape[0]
    output = torch.empty(
        (batch, 48, 128),
        dtype=torch.bfloat16,
        device=packed_qkv.device,
    )
    _gdn_decode_packed_kernel[(128 // value_tile, 48, batch)](
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        output,
        state_cache.shape[0],
        VALUE_TILE=value_tile,
        backend_options=_gdn_decode_schedule(value_tile),
    )
    return output


def gdn_decode_conv_packed(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Launch fused causal-conv + GDN as one graph-native TPC node."""
    artifact_hash = _prepare_gdn_decode_conv_packed()
    return torch.ops.triton_gaudi.gdn_decode_conv_packed.default(
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
        artifact_hash,
    )


def gdn_decode_conv_packed_direct(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Direct fused launch retained for ABI and mutation diagnostics."""
    batch = packed_qkv.shape[0]
    output = torch.empty(
        (batch, 48, 128),
        dtype=torch.bfloat16,
        device=packed_qkv.device,
    )
    _gdn_decode_conv_packed_kernel[(16, batch)](
        conv_state,
        state_cache,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
        output,
        conv_state.shape[0],
        state_cache.shape[0],
        backend_options=_gdn_decode_conv_schedule(),
    )
    return output


def gdn_decode_conv_split_packed(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Launch Q/K conv then tile-parallel fused value-conv + GDN."""
    value_tile = _gdn_decode_value_tile()
    qk_artifact_hash = _prepare_gdn_qk_conv_packed()
    value_artifact_hash = _prepare_gdn_decode_value_conv_packed(value_tile)
    qk_conv = torch.ops.triton_gaudi.gdn_qk_conv_packed.default(
        conv_state,
        packed_qkv,
        state_indices,
        conv_weight_t,
        qk_artifact_hash,
    )
    return torch.ops.triton_gaudi.gdn_decode_value_conv_packed.default(
        conv_state,
        state_cache,
        qk_conv,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
        value_artifact_hash,
        value_tile,
    )


def gdn_decode_conv_split_packed_direct(
    conv_state: torch.Tensor,
    state_cache: torch.Tensor,
    packed_qkv: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    conv_weight_t: torch.Tensor,
) -> torch.Tensor:
    """Direct two-kernel launch retained for ABI diagnostics."""
    batch = packed_qkv.shape[0]
    qk_conv = torch.empty(
        (batch, 4096),
        dtype=torch.bfloat16,
        device=packed_qkv.device,
    )
    _gdn_qk_conv_packed_kernel[(4096 // QK_CONV_TILE, batch)](
        conv_state,
        packed_qkv,
        state_indices,
        conv_weight_t,
        qk_conv,
        conv_state.shape[0],
        QK_TILE=QK_CONV_TILE,
        backend_options=_gdn_qk_conv_schedule(),
    )
    value_tile = _gdn_decode_value_tile()
    output = torch.empty(
        (batch, 48, 128),
        dtype=torch.bfloat16,
        device=packed_qkv.device,
    )
    _gdn_decode_value_conv_packed_kernel[(128 // value_tile, 48, batch)](
        conv_state,
        state_cache,
        qk_conv,
        packed_qkv,
        gate_a,
        gate_b,
        a_log,
        dt_bias,
        state_indices,
        conv_weight_t,
        output,
        conv_state.shape[0],
        state_cache.shape[0],
        VALUE_TILE=value_tile,
        backend_options=_gdn_value_conv_schedule(value_tile),
    )
    return output
