# SPDX-License-Identifier: Apache-2.0
"""Preparation helpers for optional DeepSeek V4 MXFP4 decoders."""
from dataclasses import dataclass

import torch

_MXFP4_BF16_BITS = (
    0x0000,
    0x3F00,
    0x3F80,
    0x3FC0,
    0x4000,
    0x4040,
    0x4080,
    0x40C0,
    0x8000,
    0xBF00,
    0xBF80,
    0xBFC0,
    0xC000,
    0xC040,
    0xC080,
    0xC0C0,
)

_MXFP4_FP8_E4M3_BITS = (
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x80,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
)

MXFP4_PREPARED_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class PreparedMxfp4Weight:
    """One immutable packed matrix in the Gaudi2 decoder layout."""

    q16: torch.Tensor
    s16: torch.Tensor
    original_shape: tuple[int, ...]
    scale_shape: tuple[int, ...]
    normal_scales: bool
    generation: int
    layout_version: int = MXFP4_PREPARED_LAYOUT_VERSION


def normal_e8m0_scales(*scales: torch.Tensor) -> bool:
    """Validate frozen weights once; never call this from compiled decode.

    Codes 0/1 can produce BF16 subnormals, and 255 denotes NaN. The full
    decoder handles them. Codes 2..254 use the shorter exact bit decoder,
    including signed zeros and overflow to infinity.
    """
    if not scales:
        return False
    for scale in scales:
        if scale.dtype != torch.uint8 or scale.numel() == 0 or scale.device.type == "meta":
            return False
        if int(scale.min().item()) < 2 or int(scale.max().item()) > 254:
            return False
    return True


def _prepare_mxfp4_q16_tensor(packed: torch.Tensor) -> torch.Tensor:
    prefix = packed.shape[:-2]
    rows, k_bytes = packed.shape[-2:]
    prefix_dims = len(prefix)
    byte_stream = packed.reshape(*prefix, rows // 128, 64, 2, k_bytes)
    byte_stream = byte_stream.permute(
        *range(prefix_dims),
        prefix_dims,
        prefix_dims + 3,
        prefix_dims + 1,
        prefix_dims + 2,
    ).contiguous()
    return byte_stream.view(torch.int16).reshape(*prefix, rows // 128, k_bytes * 64)


def prepare_mxfp4_q16(packed: torch.Tensor) -> torch.Tensor:
    """Pack two K values from two rows for hardware 4-to-8 unpacking.

    Input ``[...,N,K/2]`` becomes ``[...,N/128,(K/2)*64]``. Each 128-row
    block is a contiguous stream over K so a TPC does not stride through HBM.
    A word stores ``K0[row0], K1[row0], K0[row1], K1[row1]``; hardware unpack
    feeds one BF16 shuffle directly and one 16-bit shift exposes the other.
    The byte count is unchanged.
    """
    if packed.dtype != torch.uint8 or packed.ndim < 2 or packed.shape[-1] % 64:
        raise ValueError("packed MXFP4 must be uint8 [...,N,K/2] with K divisible by 128")
    if packed.shape[-2] % 128:
        raise ValueError("prepared MXFP4 output rows must be divisible by 128")
    # The low and high bytes are already the two adjacent output rows. A
    # permutation followed by a dtype view therefore builds Q16 without ever
    # expanding the FP4 codes. Gaudi's eager transpose retains a large
    # workspace for the full W13 tensor, so bound that workspace by preparing
    # four experts at a time into the one final destination allocation.
    if packed.device.type == "hpu" and packed.ndim == 3 and packed.shape[0] > 4:
        rows, k_bytes = packed.shape[-2:]
        prepared = torch.empty(
            packed.shape[0],
            rows // 128,
            k_bytes * 64,
            dtype=torch.int16,
            device=packed.device,
        )
        for start in range(0, packed.shape[0], 4):
            stop = min(start + 4, packed.shape[0])
            converted = _prepare_mxfp4_q16_tensor(packed[start:stop])
            prepared[start:stop].copy_(converted)
            torch.hpu.synchronize()
            del converted
        return prepared
    return _prepare_mxfp4_q16_tensor(packed)


def restore_mxfp4_u8(q16: torch.Tensor) -> torch.Tensor:
    """Restore the checkpoint byte layout from :func:`prepare_mxfp4_q16`."""
    if q16.dtype != torch.int16 or q16.ndim < 2 or q16.shape[-1] % 64:
        raise ValueError("prepared Q16 must be int16 [...,N/128,(K/2)*64]")
    prefix = q16.shape[:-2]
    blocks, stream = q16.shape[-2:]
    k_bytes = stream // 64
    prefix_dims = len(prefix)
    byte_stream = q16.view(torch.uint8).reshape(*prefix, blocks, k_bytes, 64, 2)
    byte_stream = byte_stream.permute(
        *range(prefix_dims),
        prefix_dims,
        prefix_dims + 2,
        prefix_dims + 3,
        prefix_dims + 1,
    )
    return byte_stream.reshape(*prefix, blocks * 128, k_bytes).contiguous()


def _prepare_mxfp4_s16_tensor(scales: torch.Tensor) -> torch.Tensor:
    prefix, rows, groups = scales.shape[:-2], scales.shape[-2], scales.shape[-1]
    values = (scales.to(torch.int16) << 7).view(torch.bfloat16)
    return values.reshape(*prefix, rows // 128, 128, groups).transpose(-2, -1).flatten(-2).contiguous()


def prepare_mxfp4_s16(scales: torch.Tensor) -> torch.Tensor:
    """Store exact E8M0 BF16 bits in block-major contiguous K streams."""
    if scales.dtype != torch.uint8 or scales.ndim < 2 or scales.shape[-2] % 128:
        raise ValueError("MXFP4 scales must be uint8 [...,N,K/32] with N divisible by 128")
    if scales.device.type == "hpu" and scales.ndim == 3 and scales.shape[0] > 4:
        rows, groups = scales.shape[-2:]
        prepared = torch.empty(
            scales.shape[0],
            rows // 128,
            groups * 128,
            dtype=torch.bfloat16,
            device=scales.device,
        )
        for start in range(0, scales.shape[0], 4):
            stop = min(start + 4, scales.shape[0])
            converted = _prepare_mxfp4_s16_tensor(scales[start:stop])
            prepared[start:stop].copy_(converted)
            torch.hpu.synchronize()
            del converted
        return prepared
    return _prepare_mxfp4_s16_tensor(scales)


def restore_mxfp4_scale_u8(s16: torch.Tensor) -> torch.Tensor:
    """Recover every E8M0 code, including 0, 1 and 255, from prepared S16."""
    if s16.dtype != torch.bfloat16 or s16.ndim < 2 or s16.shape[-1] % 128:
        raise ValueError("prepared S16 scales must be bfloat16 [...,N/128,(K/32)*128]")
    prefix, blocks, stream = s16.shape[:-2], s16.shape[-2], s16.shape[-1]
    groups = stream // 128
    values = s16.reshape(*prefix, blocks, groups, 128).transpose(-2, -1)
    values = values.reshape(*prefix, blocks * 128, groups).view(torch.int16)
    return ((values >> 7) & 255).to(torch.uint8)


def mxfp4_bf16_lut(device: torch.device | str) -> torch.Tensor:
    """Return the raw FP8 lookup bytes carried by a 128-value BF16 tensor."""
    values = torch.tensor(_MXFP4_FP8_E4M3_BITS * 16, dtype=torch.uint8)
    return values.view(torch.bfloat16).to(device)


def make_prepared_mxfp4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    generation: int,
    normal_scales: bool | None = None,
) -> PreparedMxfp4Weight:
    """Create one prepared matrix while retaining checkpoint metadata."""
    if packed.shape[:-1] != scales.shape[:-1] or packed.shape[-1] != scales.shape[-1] * 16:
        raise ValueError("packed MXFP4 weights and E8M0 scales have incompatible shapes")
    if packed.device != scales.device:
        raise ValueError("packed MXFP4 weights and E8M0 scales must share a device")
    if normal_scales is None:
        normal_scales = normal_e8m0_scales(scales)
    return PreparedMxfp4Weight(
        q16=prepare_mxfp4_q16(packed),
        s16=prepare_mxfp4_s16(scales),
        original_shape=tuple(packed.shape),
        scale_shape=tuple(scales.shape),
        normal_scales=normal_scales,
        generation=generation,
    )
