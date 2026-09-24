# SPDX-License-Identifier: Apache-2.0
"""Bounded sparse-MLA prefill for Gaudi's compiled Flash Attention backend.

This is a Gaudi extension, not a CUDA-kernel shim. It adopts FlashInfer's
selected-row prefill contract and runs Gaudi FusedSDPA with shared BF16 K/V.
Query heads form the matrix row axis because each query's heads share its
selected latent cache. An explicit zero-valued sink row preserves the
head-dependent softmax denominator without duplicating KV across heads.
"""
from __future__ import annotations

import math

import torch

from flashinfer_gaudi._dispatch import BackendUnavailableError


def _cache_with_zero_row(cache):
    """Keep invalid selections and the sink independent of cache contents."""
    zero = torch.zeros((1, cache.shape[1]), dtype=cache.dtype, device=cache.device)
    return torch.cat((cache, zero), 0)


def _final_tile_inputs(query, padded_cache, indices, sink, lengths):
    """Gather directly into the final KV layout, including a zero sink row."""
    tokens, heads, width = query.shape
    columns = indices.shape[1]
    zero_row = padded_cache.shape[0] - 1
    valid = (indices >= 0) & (indices < zero_row)
    valid = valid & (torch.arange(columns, device=indices.device)[None, :] < lengths[:, None])
    # Invalid rows must select actual zeros: masking alone cannot prevent
    # a NaN in an unrelated cache slot from contaminating the PV product.
    safe = torch.where(valid, indices, zero_row)
    sink_ids = torch.full((tokens, 1), zero_row, dtype=indices.dtype, device=indices.device)
    final_ids = torch.cat((safe, sink_ids), 1).long()
    kv = padded_cache.index_select(0, final_ids.flatten()).reshape(tokens, 1, columns + 1, width)
    mask = torch.where(valid, 0.0, -torch.inf).float()[:, None, None, :].expand(tokens, 1, heads, columns)
    mask = torch.cat((mask, sink.reshape(1, 1, heads, 1).expand(tokens, 1, heads, 1)), -1)
    return query.unsqueeze(1), kv, mask


def _tile_inputs(query, cache, indices, sink, lengths):
    """Build one bounded BF16 K/V tile and its FP32 additive mask."""
    return _final_tile_inputs(query, _cache_with_zero_row(cache), indices, sink, lengths)


def _native_tile_inputs(query, cache, indices, sink, lengths):
    """Write the final KV and sink-mask layouts without intermediate copies."""
    kv, mask = torch.ops.custom_op.custom_deepseek_v41_prefill_flash_inputs_gaudi2(cache.contiguous(),
                                                                                   indices.contiguous(),
                                                                                   lengths.contiguous(),
                                                                                   sink.contiguous())
    return query.unsqueeze(1), kv.unsqueeze(1), mask.unsqueeze(1)


def _full_valid_tile_inputs(query, cache, indices, sink, lengths):
    """Share a sink-only mask when the caller proves every selected ID valid.

    This experimental helper does not inspect device values or silently
    replace invalid rows. No production caller sets this contract yet.
    """
    tokens, heads, width = query.shape
    columns = indices.shape[1]
    selected = cache.index_select(0, indices.long().flatten()).reshape(tokens, columns, width)
    zero = torch.zeros((tokens, 1, width), dtype=query.dtype, device=query.device)
    kv = torch.cat((selected, zero), 1).unsqueeze(1)
    shared = torch.zeros((1, 1, heads, columns), dtype=torch.float32, device=query.device)
    mask = torch.cat((shared, sink.reshape(1, 1, heads, 1)), -1)
    return query.unsqueeze(1), kv, mask


def sparse_mla_prefill(query, cache, indices, sink, lengths, *, sm_scale=None, query_tile=512,
                       native_inputs=False, full_valid=False):
    """Return BF16 ``[T,H,512]`` sparse MLA output from a flat latent cache.

    ``indices`` is runtime I32 ``[T,K]``. Negative/out-of-range entries and
    columns beyond each runtime length are masked. ``sink`` contains H FP32
    logit biases contributing only to the denominator. This operation owns no
    KV/history state and is intended to be enclosed in a compiled region or
    the model's prepared native plan. Unsupported contracts fail explicitly.

    QK and softmax use the vendor FP32 mode. Shared K/V stays BF16; the vendor
    PV arithmetic may differ from the previous FP32 PV implementation and
    needs numerical qualification before production promotion.
    """
    if query.ndim != 3 or query.shape[2] != 512 or min(query.shape) < 1:
        raise ValueError("Sparse MLA query requires nonempty [T,H,512]")
    tokens, heads, _ = query.shape
    if (cache.ndim != 2 or cache.shape[0] < 1 or cache.shape[1] != 512 or indices.ndim != 2
            or indices.shape[0] != tokens or not 1 <= indices.shape[1] <= 640 or sink.shape != (heads, )
            or lengths.shape != (tokens, )):
        raise ValueError("Sparse MLA cache, indices, sink or lengths contract changed")
    if query_tile not in (16, 32, 64, 128, 256, 512):
        raise ValueError("Sparse MLA query tile must be 16,32,64,128,256 or 512")
    expected = (torch.bfloat16, torch.bfloat16, torch.int32, torch.float32, torch.int32)
    tensors = (query, cache, indices, sink, lengths)
    if any(t.dtype != dtype or t.device != query.device or t.requires_grad for t, dtype in zip(tensors, expected)):
        raise ValueError("Sparse MLA requires inference BF16 Q/KV, I32 IDs/lengths and FP32 sink")
    factor = 512**-0.5 if sm_scale is None else sm_scale
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Sparse MLA scale must be a finite positive static number")
    if query.device.type != "hpu":
        raise BackendUnavailableError("Sparse MLA native prefill requires Gaudi; no reference fallback")
    if full_valid and native_inputs:
        raise ValueError("Full-valid mask and native input producer are separate candidates")
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    # Prepare the zero row once for all inner tiles in this compiled call.
    # Including its index in the gather removes a full selected-KV masking
    # pass and the later sink concat while preserving the SDPA operands.
    tile_cache = cache if (full_valid or native_inputs) else _cache_with_zero_row(cache)
    prepare = _full_valid_tile_inputs if full_valid else (_native_tile_inputs if native_inputs else _final_tile_inputs)
    outputs = []
    for start in range(0, tokens, query_tile):
        stop = min(start + query_tile, tokens)
        q, kv, mask = prepare(query[start:stop], tile_cache, indices[start:stop], sink, lengths[start:stop])
        output = FusedSDPA.apply(q, kv, kv, mask, 0.0, False, factor, "fp32", True)
        outputs.append(output.squeeze(1))
    return torch.cat(outputs, 0)
