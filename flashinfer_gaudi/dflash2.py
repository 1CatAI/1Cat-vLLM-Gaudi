# SPDX-License-Identifier: Apache-2.0
"""Gaudi-friendly DFlash2 primitives with portable PyTorch fallbacks.

The public functions in this module deliberately keep the tensor contracts
used by the upstream DFlash2 implementation.  Native Gaudi kernels are loaded
when they are present, while CPU-only tests and unsupported shapes use the
same deterministic PyTorch implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from flashinfer_gaudi._native import (
    native_dflash2_grouped_conv_op,
    native_dflash2_score_select_op,
    native_dflash2_select_path_op,
    native_dflash2_top_k_op,
)
from flashinfer_gaudi._tactics import (
    dflash2_score_select_auto_promoted,
    dflash2_select_path_auto_promoted,
    dflash2_top_k_auto_promoted,
)

_QWEN38_DFLASH2_STEPS = 7
_QWEN38_DFLASH2_TOP_K = 16
_QWEN38_DFLASH2_MAX_BATCH = 16
_QWEN38_DFLASH2_VOCAB_SIZE = 248320
_QWEN38_DFLASH2_SELECTOR_RANK = 256


def _is_hpu(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "hpu"


def _require_integer(name: str, value: int, *, minimum: int = 1) -> int:
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return value


def build_attention_bias(
    query_starts: Sequence[int] | torch.Tensor,
    first_blocks: Sequence[int] | torch.Tensor,
    block_size: int,
    past_width: int,
    query_len: int,
    window: int | None,
    causal: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build DFlash's explicit context-plus-query attention bias on CPU.

    FlashAttention defines a window of size ``W`` as ``W - 1`` positions to
    either side of the query. Encoding causality in this bias lets the HPU
    prompt kernel represent both DFlash's bidirectional block and its causal
    checkpoint variants without relying on bottom-right causal alignment.
    """
    block_size = _require_integer("block_size", block_size)
    query_len = _require_integer("query_len", query_len)
    if len(query_starts) != len(first_blocks):
        raise ValueError("query_starts and first_blocks must have the same length.")
    if past_width < 0:
        raise ValueError(f"past_width must be non-negative, got {past_width}.")
    if window is not None:
        window = _require_integer("window", window)

    batch = len(query_starts)
    starts = torch.as_tensor(query_starts, dtype=torch.int64).reshape(batch, 1)
    device = starts.device
    frame_bases = (torch.as_tensor(first_blocks, dtype=torch.int64, device=device).reshape(batch, 1) * block_size)
    query_positions = starts + torch.arange(query_len, dtype=torch.int64, device=device)

    if past_width:
        past_positions = frame_bases + torch.arange(past_width, dtype=torch.int64, device=device)
        past_valid = (past_positions[:, None, :] < starts[:, :, None]).expand(-1, query_len, -1).clone()
        if window is not None:
            radius = window - 1
            past_valid &= past_positions[:, None, :] >= (query_positions[:, :, None] - radius)
    else:
        past_valid = torch.empty((batch, query_len, 0), dtype=torch.bool, device=device)

    query_keys = query_positions[:, None, :]
    query_valid = torch.ones((batch, query_len, query_len), dtype=torch.bool, device=device)
    if window is not None:
        query_valid &= (query_positions[:, :, None] - query_keys).abs() <= window - 1
    if causal:
        query_valid &= query_keys <= query_positions[:, :, None]

    valid = torch.cat((past_valid, query_valid), dim=-1).unsqueeze(1)
    return torch.zeros(valid.shape, dtype=dtype, device=device).masked_fill_(~valid, -torch.inf)


def prepare_device_inputs(
    sampled_token_ids: torch.Tensor,
    base_positions: torch.Tensor,
    block_table: torch.Tensor,
    active_mask: torch.Tensor,
    first_blocks: torch.Tensor,
    last_blocks: torch.Tensor,
    *,
    block_size: int,
    num_query_per_req: int,
    mask_token_id: int,
    pad_block_id: int,
    pad_slot_id: int,
    max_model_len: int,
    max_context_blocks: int,
    window: int | None,
    causal: bool,
    bias_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Prepare a fixed-width DFlash2 draft entirely on the accelerator.

    The target rejection sampler emits a request-major tensor whose valid
    prefix is followed by ``-1``. Keeping the full target width avoids making
    the host learn each request's accepted length before the draft can start.
    Rejected context rows and inactive graph-padding rows are redirected to
    distinct slots in the reserved padding block.
    """
    block_size = _require_integer("block_size", block_size)
    num_query_per_req = _require_integer("num_query_per_req", num_query_per_req)
    max_model_len = _require_integer("max_model_len", max_model_len)
    max_context_blocks = _require_integer("max_context_blocks", max_context_blocks)
    if sampled_token_ids.ndim != 2:
        raise ValueError("sampled_token_ids must have shape [batch, target_width], "
                         f"got {tuple(sampled_token_ids.shape)}.")
    batch_size, target_width = sampled_token_ids.shape
    if tuple(base_positions.shape) != (batch_size, ):
        raise ValueError(f"base_positions must have shape {(batch_size,)}, got {tuple(base_positions.shape)}.")
    if block_table.ndim != 2 or block_table.shape[0] != batch_size:
        raise ValueError("block_table must have shape [batch, max_blocks], "
                         f"got {tuple(block_table.shape)} for batch {batch_size}.")
    for name, tensor in (
        ("active_mask", active_mask),
        ("first_blocks", first_blocks),
        ("last_blocks", last_blocks),
    ):
        if tuple(tensor.shape) != (batch_size, ):
            raise ValueError(f"{name} must have shape {(batch_size,)}, got {tuple(tensor.shape)}.")
    tensors = (sampled_token_ids, base_positions, block_table, active_mask, first_blocks, last_blocks)
    if any(tensor.device != sampled_token_ids.device for tensor in tensors):
        raise ValueError("all DFlash2 device-prepare tensors must be on the same device.")
    if max_context_blocks > block_table.shape[1]:
        raise ValueError("max_context_blocks cannot exceed the block-table width, "
                         f"got {max_context_blocks} > {block_table.shape[1]}.")

    device = sampled_token_ids.device
    active = active_mask.to(torch.bool)
    valid_context = sampled_token_ids.ne(-1) & active[:, None]
    accepted_counts = valid_context.sum(dim=-1, dtype=torch.int32)
    safe_counts = accepted_counts.clamp(min=1)

    bonus_indices = (safe_counts - 1).to(torch.int64).unsqueeze(1)
    bonus_token_ids = sampled_token_ids.gather(1, bonus_indices).squeeze(1).to(torch.int32)
    bonus_token_ids = torch.where(active, bonus_token_ids, torch.full_like(bonus_token_ids, mask_token_id))

    context_offsets = torch.arange(target_width, dtype=torch.int64, device=device).unsqueeze(0)
    context_positions = base_positions.to(torch.int64).unsqueeze(1) + context_offsets
    context_positions = torch.where(valid_context, context_positions, torch.zeros_like(context_positions))

    query_starts = base_positions.to(torch.int64) + accepted_counts.to(torch.int64)
    query_starts = torch.where(active, query_starts, torch.zeros_like(query_starts))
    query_offsets = torch.arange(num_query_per_req, dtype=torch.int64, device=device).unsqueeze(0)
    query_positions = (query_starts.unsqueeze(1) + query_offsets).clamp(max=max_model_len - 1)

    mask_tokens = torch.full(
        (batch_size, num_query_per_req - 1),
        mask_token_id,
        dtype=torch.int32,
        device=device,
    )
    input_ids = torch.cat((bonus_token_ids.unsqueeze(1), mask_tokens), dim=1)

    table_width = block_table.shape[1]

    def make_slots(positions: torch.Tensor, valid: torch.Tensor, dummy_offset: int) -> torch.Tensor:
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        in_table = logical_blocks.ge(0) & logical_blocks.lt(table_width)
        safe_logical_blocks = logical_blocks.clamp(min=0, max=table_width - 1)
        physical_blocks = block_table.gather(1, safe_logical_blocks)
        resident = valid & in_table & physical_blocks.ge(0) & physical_blocks.ne(pad_block_id)
        physical_slots = physical_blocks.to(torch.int64) * block_size + positions.remainder(block_size)
        dummy_slots = pad_slot_id + (
            torch.arange(positions.numel(), dtype=torch.int64, device=device).reshape_as(positions) +
            dummy_offset).remainder(block_size)
        return torch.where(resident, physical_slots, dummy_slots)

    context_slots = make_slots(context_positions, valid_context, 0)
    query_valid = active[:, None].expand(-1, num_query_per_req)
    query_slots = make_slots(query_positions, query_valid, target_width * batch_size)

    block_offsets = torch.arange(max_context_blocks, dtype=torch.int64, device=device).unsqueeze(0)
    logical_context_blocks = first_blocks.to(torch.int64).unsqueeze(1) + block_offsets
    valid_context_blocks = (active[:, None]
                            & logical_context_blocks.le(last_blocks.to(torch.int64).unsqueeze(1))
                            & logical_context_blocks.ge(0)
                            & logical_context_blocks.lt(table_width))
    safe_context_blocks = logical_context_blocks.clamp(min=0, max=table_width - 1)
    context_block_list = block_table.gather(1, safe_context_blocks)
    context_block_list = torch.where(
        valid_context_blocks & context_block_list.ge(0) & context_block_list.ne(pad_block_id),
        context_block_list,
        torch.full_like(context_block_list, pad_block_id),
    )

    context_lens = query_starts - first_blocks.to(torch.int64) * block_size
    context_lens = torch.where(active, context_lens, torch.zeros_like(context_lens)).to(torch.int32)
    attention_bias = build_attention_bias(
        query_starts=query_starts,
        first_blocks=first_blocks,
        block_size=block_size,
        past_width=max_context_blocks * block_size,
        query_len=num_query_per_req,
        window=window,
        causal=causal,
        dtype=bias_dtype,
    )

    return (
        accepted_counts,
        context_positions.reshape(-1),
        context_slots.reshape(-1),
        input_ids,
        query_positions,
        query_slots,
        context_block_list.reshape(-1),
        context_lens,
        attention_bias,
    )


def grouped_conv_reference(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Reference grouped dynamic convolution used by DFlash2.

    ``hidden_states`` is request-major.  Convolution history is reset at every
    ``block_size`` boundary, so tokens from adjacent requests never interact.
    """
    block_size = _require_integer("block_size", block_size)
    num_groups = _require_integer("num_groups", num_groups)
    group_size = _require_integer("group_size", group_size)
    taps = _require_integer("taps", taps)
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size], "
                         f"got {tuple(hidden_states.shape)}.")
    tokens, hidden_size = hidden_states.shape
    if hidden_size != num_groups * group_size:
        raise ValueError("hidden_size must equal num_groups * group_size, "
                         f"got {hidden_size} != {num_groups} * {group_size}.")
    expected_delta = (tokens, taps, num_groups)
    if tuple(delta.shape) != expected_delta:
        raise ValueError(f"delta must have shape {expected_delta}, got {tuple(delta.shape)}.")
    expected_base = (taps, hidden_size)
    if tuple(base.shape) != expected_base:
        raise ValueError(f"base must have shape {expected_base}, got {tuple(base.shape)}.")

    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.reshape(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(tokens, device=hidden_states.device)
    position = (position & (block_size - 1) if block_size & (block_size - 1) == 0 else position % block_size)
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        valid = (position >= tap).reshape(-1, 1, 1)
        output = output + coefficients[:, tap] * shifted * valid
    return output.flatten(-2)


def grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Run DFlash2 grouped convolution, preferring a native Gaudi kernel."""
    native = native_dflash2_grouped_conv_op() if _is_hpu(hidden_states) else None
    if native is not None:
        return native(hidden_states, delta, base, block_size, num_groups, group_size, taps)
    return grouped_conv_reference(hidden_states, delta, base, block_size, num_groups, group_size, taps)


def score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Score the K-by-K transition lattice for every DFlash2 draft step."""
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, steps, top_k], "
                         f"got {tuple(candidate_ids.shape)}.")
    if unary_logits.shape != candidate_ids.shape:
        raise ValueError("unary_logits must have the same shape as candidate_ids.")
    batch, steps, top_k = candidate_ids.shape
    if hidden.ndim != 3 or hidden.shape[:2] != (batch, steps):
        raise ValueError("hidden must have shape [batch, steps, rank], "
                         f"got {tuple(hidden.shape)}.")
    if tuple(anchor_token_ids.shape) != (batch, ):
        raise ValueError(f"anchor_token_ids must have shape {(batch,)}, got {tuple(anchor_token_ids.shape)}.")

    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    if _is_hpu(candidate_ids) and top_k > 0 and hidden.shape[-1] > 0:
        # Keep the contraction explicitly three-dimensional. HPU's einsum
        # lowering can retain a singleton-batch reshape after a batch change
        # under force_static_compile. Preserve activation-dtype products and
        # the contraction result before adding the FP32 unary scores.
        rank = hidden.shape[-1]
        weighted = (predecessors * hidden[:, :, None]).reshape(-1, top_k, rank)
        edges = torch.bmm(weighted, successors.reshape(-1, top_k, rank).transpose(1, 2))
        return unary_logits[:, :, None] + edges.reshape(batch, steps, top_k, top_k)
    return unary_logits[:, :, None] + torch.einsum("blpr,blcr->blpc", predecessors * hidden[:, :, None], successors)


def score_and_select_path_reference(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    scores = score_edges(
        predecessor_table,
        successor_table,
        candidate_ids,
        unary_logits,
        hidden,
        anchor_token_ids,
    )
    return select_path_reference(candidate_ids, scores)


def _native_score_select_supported(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> bool:
    batch = candidate_ids.shape[0] if candidate_ids.ndim == 3 else 0
    device = candidate_ids.device
    tensors = (predecessor_table, successor_table, candidate_ids, unary_logits, hidden, anchor_token_ids)
    return (predecessor_table.dtype == torch.bfloat16 and successor_table.dtype == torch.bfloat16
            and candidate_ids.dtype in (torch.int32, torch.int64) and unary_logits.dtype == torch.float32
            and hidden.dtype == torch.bfloat16 and anchor_token_ids.dtype == torch.int32
            and tuple(predecessor_table.shape) == (_QWEN38_DFLASH2_VOCAB_SIZE, _QWEN38_DFLASH2_SELECTOR_RANK)
            and tuple(successor_table.shape) == tuple(predecessor_table.shape)
            and 0 < batch <= _QWEN38_DFLASH2_MAX_BATCH
            and tuple(candidate_ids.shape) == (batch, _QWEN38_DFLASH2_STEPS, _QWEN38_DFLASH2_TOP_K)
            and tuple(unary_logits.shape) == tuple(candidate_ids.shape)
            and tuple(hidden.shape) == (batch, _QWEN38_DFLASH2_STEPS, _QWEN38_DFLASH2_SELECTOR_RANK)
            and tuple(anchor_token_ids.shape) == (batch, ) and all(tensor.device == device for tensor in tensors)
            and all(tensor.is_contiguous() for tensor in tensors))


def maybe_score_and_select_path(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor | None:
    """Use the fused selected-row scorer only after independent promotion."""
    if (not _is_hpu(candidate_ids) or not dflash2_score_select_auto_promoted() or not _native_score_select_supported(
            predecessor_table,
            successor_table,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
    )):
        return None
    native = native_dflash2_score_select_op()
    if native is None:
        return None
    return native(
        predecessor_table,
        successor_table,
        candidate_ids,
        unary_logits,
        hidden,
        anchor_token_ids,
    )


def select_path_reference(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Greedily walk a DFlash2 candidate lattice.

    Ties follow ``torch.argmax`` semantics (the first candidate wins), matching
    the deterministic greedy contract required by the HPU integration.
    """
    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [batch, steps, top_k], "
                         f"got {tuple(candidate_ids.shape)}.")
    batch, steps, top_k = candidate_ids.shape
    expected_scores = (batch, steps, top_k, top_k)
    if tuple(scores.shape) != expected_scores:
        raise ValueError(f"scores must have shape {expected_scores}, got {tuple(scores.shape)}.")
    if steps < 1:
        return candidate_ids.new_empty((batch, 0))

    previous = torch.zeros(batch, dtype=torch.long, device=candidate_ids.device)
    batch_indices = torch.arange(batch, dtype=torch.long, device=candidate_ids.device)
    selected: list[torch.Tensor] = []
    for step in range(steps):
        transition_scores = scores[batch_indices, step, previous]
        previous = transition_scores.argmax(dim=-1)
        selected.append(candidate_ids[batch_indices, step, previous])
    return torch.stack(selected, dim=1)


def _native_select_path_supported(candidate_ids: torch.Tensor, scores: torch.Tensor) -> bool:
    return (candidate_ids.dtype in (torch.int32, torch.int64) and scores.dtype == torch.float32
            and candidate_ids.ndim == 3 and scores.ndim == 4 and 0 < candidate_ids.shape[0] <= _QWEN38_DFLASH2_MAX_BATCH
            and tuple(candidate_ids.shape[1:]) == (_QWEN38_DFLASH2_STEPS, _QWEN38_DFLASH2_TOP_K)
            and tuple(scores.shape) == (
                candidate_ids.shape[0],
                _QWEN38_DFLASH2_STEPS,
                _QWEN38_DFLASH2_TOP_K,
                _QWEN38_DFLASH2_TOP_K,
            ) and candidate_ids.device == scores.device and candidate_ids.is_contiguous() and scores.is_contiguous())


def select_path(candidate_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Select one greedy DFlash2 path with a qualification-gated TPC kernel."""
    native = None
    if (_is_hpu(candidate_ids) and dflash2_select_path_auto_promoted()
            and _native_select_path_supported(candidate_ids, scores)):
        native = native_dflash2_select_path_op()
    if native is not None:
        return native(candidate_ids, scores)
    return select_path_reference(candidate_ids, scores)


def top_k_reference(
    scores: torch.Tensor,
    k: int,
    *,
    sorted: bool = True,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Framework TopK with returned ties canonicalized by token id."""
    k = _require_integer("k", k)
    if scores.ndim < 1 or k > scores.shape[-1]:
        raise ValueError(f"k={k} is invalid for scores shape {tuple(scores.shape)}.")

    values, indices = torch.topk(scores, k, dim=-1, largest=True, sorted=sorted or deterministic)
    # Gaudi's eager gather cannot compile an int64 data tensor. TopK returns
    # int64 indices even though Qwen3.8's vocabulary fits safely in int32, so
    # narrow the candidate-id ABI before deterministic tie canonicalization.
    # Keep the framework-standard int64 dtype on CPU/CUDA.
    if _is_hpu(scores):
        indices = indices.to(torch.int32)
    if deterministic:
        # Stable sort by token id first, then by descending score.  The second
        # stable sort preserves ascending token ids for equal returned scores.
        by_id = torch.argsort(indices, dim=-1, stable=True)
        indices = indices.gather(-1, by_id)
        values = values.gather(-1, by_id)
        by_score = torch.argsort(values, dim=-1, descending=True, stable=True)
        indices = indices.gather(-1, by_score)
        values = values.gather(-1, by_score)
    elif sorted:
        order = torch.argsort(values, dim=-1, descending=True)
        indices = indices.gather(-1, order)
        values = values.gather(-1, order)
    return values, indices


def _top_k_vendor_cguid(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the Synapse TopK CGUID without the tie-canonicalization graph."""
    return torch.topk(scores, k, dim=-1, largest=True, sorted=True)


def _vendor_top_k_supported(scores: torch.Tensor, k: int, sorted: bool, deterministic: bool) -> bool:
    rows = scores.shape[0] if scores.ndim == 2 else 0
    return (scores.dtype == torch.bfloat16 and scores.ndim == 2 and scores.shape[1] == _QWEN38_DFLASH2_VOCAB_SIZE
            and rows % _QWEN38_DFLASH2_STEPS == 0 and 0 < rows <= _QWEN38_DFLASH2_MAX_BATCH * _QWEN38_DFLASH2_STEPS
            and k == _QWEN38_DFLASH2_TOP_K and sorted and deterministic and scores.is_contiguous())


def top_k(
    scores: torch.Tensor,
    k: int,
    *,
    sorted: bool = True,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact top-k values and indices for DFlash2 candidate selection.

    The production Gaudi path already lowers ``torch.topk`` to Intel's TopK
    CGUID. The qualification candidate calls it directly and removes four tiny
    stable-sort/gather ops; unsupported shapes preserve the canonical reference.
    """
    k = _require_integer("k", k)
    if scores.ndim < 1 or k > scores.shape[-1]:
        raise ValueError(f"k={k} is invalid for scores shape {tuple(scores.shape)}.")
    if (_is_hpu(scores) and dflash2_top_k_auto_promoted()
            and _vendor_top_k_supported(scores, k, sorted, deterministic)):
        native = native_dflash2_top_k_op()
        if native is not None:
            return native(scores, k, sorted, deterministic)
        return _top_k_vendor_cguid(scores, k)
    return top_k_reference(scores, k, sorted=sorted, deterministic=deterministic)


__all__ = [
    "build_attention_bias",
    "grouped_conv",
    "grouped_conv_reference",
    "maybe_score_and_select_path",
    "prepare_device_inputs",
    "score_and_select_path_reference",
    "score_edges",
    "select_path",
    "select_path_reference",
    "top_k",
    "top_k_reference",
]
