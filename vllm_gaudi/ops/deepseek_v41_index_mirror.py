# SPDX-License-Identifier: Apache-2.0
"""Bounded, derived index keys; scheduler-owned packed pages remain canonical."""

import torch

from vllm_gaudi.ops.deepseek_v41_prefill_kv_reuse import _compiled, _signature

INDEX_MIRROR_TOKENS = 32768


def index_mirror_execution_mode(program, search_length=None):
    """A captured writer/reader contract, separate from mutable request data."""
    shared = getattr(program, "shared", None)
    capacity = getattr(shared, "index_mirror_tokens", 0)
    if not capacity:
        return None
    search = getattr(program, "search_length", 512) if search_length is None else search_length
    return bool(shared.index_mirror_valid and search <= capacity)


def initialize_index_mirror(shared, tensor_parallel_size, device):
    capacity = getattr(torch.ops.custom_op, "deepseek_v41_decoded_index_capacity", None)
    shared.index_mirror_tokens = (min(shared.length, INDEX_MIRROR_TOKENS)
                                  if tensor_parallel_size > 1
                                  and capacity is not None and capacity() >= INDEX_MIRROR_TOKENS else 0)
    shared.index_mirror_valid = False
    shared.index_mirror_rebuilds = 0
    if not shared.index_mirror_tokens:
        return
    shared.register_buffer("index_mirror_visible", torch.zeros(1, dtype=torch.int32, device=device), False)
    for cache in shared.sources.values():
        if cache.ratio not in (1, 2):
            raise ValueError("Bounded index mirrors require ratio 1 or 2")
        rows = shared.index_mirror_tokens // cache.ratio
        cache.register_buffer("index_mirror", torch.zeros(rows, 128, dtype=torch.bfloat16, device=device), False)
        cache.register_buffer("index_mirror_rows", torch.arange(rows, dtype=torch.int32, device=device), False)


def _restore_index(packed, pages, logical, visible, destination, ratio):
    for start in range(0, logical.numel(), 2048):
        rows = logical[start:start + 2048]
        if packed.device.type == "hpu":
            keys = torch.ops.custom_op.custom_deepseek_v41_index_keys_gaudi2(
                packed, pages, rows.reshape(1, -1).contiguous(), ratio).reshape(-1, 128)
        else:
            from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
            width = 128 // ratio
            physical = pages.index_select(0, (rows // width).long()).long() * width + rows % width
            safe = physical.clamp(0, packed.shape[0] - 1).long()
            keys = unpack_fp4(packed.index_select(0, safe), 128, 32)
            keys = torch.where(((physical >= 0) & (physical < packed.shape[0]))[:, None], keys, 0)
        valid = rows < torch.div(visible, ratio, rounding_mode="floor")
        destination[start:start + rows.numel()].copy_(torch.where(valid[:, None], keys, 0))
    return destination


def prepare_index_mirror(shared, visible_tokens):
    if not shared.index_mirror_tokens or shared.index_mirror_valid:
        return
    if not 0 <= visible_tokens <= shared.index_mirror_tokens:
        raise ValueError("Index mirror prefix exceeds its bounded capacity")
    shared.index_mirror_visible.fill_(visible_tokens)
    for cache in shared.sources.values():
        args = (cache.index, shared.block_table, cache.index_mirror_rows,
                shared.index_mirror_visible, cache.index_mirror, cache.ratio)
        if cache.index.device.type == "hpu":
            _compiled(_restore_index, _signature(*args))(*args)
        else:
            _restore_index(*args)
    shared.index_mirror_valid = True
    shared.index_mirror_rebuilds += 1


def mirror_index_tile(query, weights, keys, positions, rows, ratio, local_heads):
    """Retain original K128 products, BF16 shard sums and logical masking."""
    columns = rows.shape[-1]
    logical = rows.reshape(-1).to(torch.int32).contiguous()
    safe = logical.clamp(0, keys.shape[0] - 1).long()
    selected = keys.index_select(0, safe)
    if columns % 128:
        padding = (-columns) % 128
        logical = torch.nn.functional.pad(logical, (0, padding), value=-1)
        selected = torch.nn.functional.pad(selected, (0, 0, 0, padding))
    result = torch.ops.custom_op.custom_deepseek_v41_prefill_index_scores_gaudi2(
        query.contiguous(), weights.contiguous(), selected.contiguous(),
        positions.to(torch.int32).contiguous(), logical, ratio, local_heads)
    return result[:, :columns]
