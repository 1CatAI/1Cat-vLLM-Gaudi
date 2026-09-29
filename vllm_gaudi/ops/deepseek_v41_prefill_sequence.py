# SPDX-License-Identifier: Apache-2.0
"""Four-rank sparse MLA query exchange with bounded temporary lifetimes.

The caller owns replicated KV/indices and immutable gathered sinks. This helper
does not update cache state and must use the caller's tensor-parallel group.
"""
from functools import lru_cache
from types import FunctionType

import torch
import torch.distributed as dist

from vllm_gaudi.ops.deepseek_v41_prefill_mla import sparse_prefill_mla

SEQUENCE_QUERY_CHUNK = 16384
SEQUENCE_FUSED_LAYOUT = True


def _retire_stream():
    torch.hpu.synchronize()


def _mla64(query, cache, indices, sink):
    lengths = torch.full((query.shape[0], ), indices.shape[1], dtype=torch.int32, device=query.device)
    return sparse_prefill_mla(query, cache, indices, sink, lengths, query_tile=256)


@lru_cache(maxsize=16)
def _compiled_mla64(signature):
    entry = FunctionType(_mla64.__code__.replace(co_name=f"sequence_mla_{signature}"), _mla64.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def _exchanged_mla64(query, cache, indices, sink):
    # Expose layout preparation/consumption to the same bounded compiled region
    # as MLA instead of materializing full-sequence head-major intermediates.
    rows = query.shape[1]
    full = query.permute(1, 0, 2, 3).reshape(rows, 64, 512)
    output = _mla64(full, cache, indices, sink)
    return output.reshape(rows, 4, 16, 512).permute(1, 0, 2, 3).contiguous()


@lru_cache(maxsize=16)
def _compiled_exchanged_mla64(signature):
    entry = FunctionType(_exchanged_mla64.__code__.replace(co_name=f"exchanged_mla_{signature}"),
                         _exchanged_mla64.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


def sequence_prefill_mla(query,
                         cache,
                         indices,
                         full_sink,
                         rank,
                         *,
                         group,
                         query_chunk=None,
                         fused_layout=None,
                         retire_chunks=True):
    """Exchange complete query rows, preserve each softmax and PV reduction.

    HCCL calls use the normal blocking PyTorch contract. All layout copies and
    compiled consumers retain their device dependencies before Python owners
    are retired; no raw pointers or manually reused communication buffers exist.
    """
    if query.ndim != 3 or query.shape[1:] != (16, 512) or query.shape[0] % 4 or query.shape[0] < 4:
        raise ValueError("Sequence MLA requires four equal query chunks with 16 local heads of width 512")
    if not 0 <= rank < 4 or dist.get_world_size(group) != 4 or dist.get_rank(group) != rank:
        raise ValueError("Sequence MLA requires the four-rank TP group and its local rank")
    if full_sink.shape != (64, ) or indices.shape[0] != query.shape[0]:
        raise ValueError("Sequence MLA requires 64 gathered sinks and a replicated index row per query")
    chunk = SEQUENCE_QUERY_CHUNK if query_chunk is None else query_chunk
    fused_layout = SEQUENCE_FUSED_LAYOUT if fused_layout is None else fused_layout
    if not isinstance(chunk, int) or chunk < 4 or chunk % 4:
        raise ValueError("Sequence MLA exchange chunk must contain a positive multiple of four queries")
    if not retire_chunks and query.shape[0] > chunk:
        raise ValueError("Unretired sequence MLA is bounded to one exchange chunk")
    # Each reverse exchange writes its disjoint contiguous final-output view.
    # The full owner survives all consumers, eliminating the final full-query
    # concatenation and its second output allocation without reusing storage.
    output = query.new_empty(query.shape)
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        _sequence_tile(query[start:end],
                       cache,
                       indices[start:end],
                       full_sink,
                       rank,
                       group,
                       output[start:end],
                       fused_layout=fused_layout)
        # Eager HCCL/compiled submissions can retain tensor owners after Python
        # references are gone. Retire the bounded chunk before admitting another
        # exchange working set; keep this cost inside the caller's chain timer.
        if retire_chunks:
            _retire_stream()
    return output


def _sequence_tile(query, cache, indices, full_sink, rank, group, returned, *, fused_layout=False):
    # Chunk only the independent query dimension. Full selected-row softmax and
    # PV reductions remain unchanged, and every output returns to its head owner.
    tokens, heads, width = query.shape
    rows = tokens // 4
    received = torch.empty_like(query)
    dist.all_to_all_single(received, query.contiguous(), group=group)
    if fused_layout:
        received = received.reshape(4, rows, heads, width)
        packed = torch.empty_like(received)
        for begin in range(0, rows, 512):
            stop = min(begin + 512, rows)
            signature = (stop - begin, rows, tuple(cache.shape), indices.shape[1])
            selected = indices[rank * rows + begin:rank * rows + stop].contiguous()
            block = _compiled_exchanged_mla64(signature)(received[:, begin:stop], cache, selected, full_sink)
            packed[:, begin:stop].copy_(block)
            del block, selected
        del received
        dist.all_to_all_single(returned, packed.reshape_as(query), group=group)
        return
    full = received.reshape(4, rows, heads, width).permute(1, 0, 2, 3).reshape(rows, 64, width)
    del received
    selected = indices[rank * rows:(rank + 1) * rows].contiguous()
    outputs = []
    for begin in range(0, rows, 512):
        stop = min(begin + 512, rows)
        signature = (stop - begin, tuple(cache.shape), selected.shape[1])
        outputs.append(
            _compiled_mla64(signature)(full[begin:stop].contiguous(), cache, selected[begin:stop].contiguous(),
                                       full_sink))
    del full, selected
    output = torch.cat(outputs, 0)
    del outputs
    packed = output.reshape(rows, 4, heads, width).permute(1, 0, 2, 3).contiguous().reshape_as(query)
    del output
    dist.all_to_all_single(returned, packed, group=group)
