# SPDX-License-Identifier: Apache-2.0
"""Explicit compiled tensor regions around bounded prefill operators.

The decoder's ordinary prompt path deliberately does not capture a whole
layer group: doing so retains every expert workspace until the group ends.
These regions own only their local intermediates. Communication and bounded
expert/attention execution remain explicit consumers outside the region.
"""
from collections import OrderedDict
from functools import wraps
from itertools import count
from types import FunctionType, MethodType

import torch

from vllm_gaudi import envs

_entries = count()
_function_regions = OrderedDict()
# Compile only explicit tensor contracts. Broader projection regions need
# separate qualification because compiler reduction choices can affect routing.
_qualified_regions = frozenset(
    ("_prefill_swa_workspace", "_prefill_main_workspace", "_prefill_engram_unpack", "_prefill_combine"))
_qualified_function_regions = frozenset(("_prefill_hc_post", "_prefill_hc_collapse", "_prefill_hc_control_bf16",
                                         "_prefill_combine", "prefill_main_workspace", "prefill_hc_input",
                                         "prefill_q_projection", "prefill_output_projection", "prefill_swa_workspace"))


def validate_prefill_region_config():
    """Reject an incomplete input-region contract before loading model state."""
    if envs.VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP:
        requirements = {
            "Full-index SRAM scores": envs.VLLM_HPU_DSV41_PREFILL_INDEX_SRAM,
            "shared decoded Reindex keys": envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE,
            "Reindex SRAM scores": envs.VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM,
        }
        missing = [name for name, enabled in requirements.items() if not enabled]
        if missing:
            raise ValueError("TP-partitioned prefill index requires: " + ", ".join(missing))
    if envs.VLLM_HPU_DSV41_PREFILL_INDEX_VISIBLE and not (envs.VLLM_HPU_DSV41_PREFILL_INDEX_SRAM
                                                          and envs.VLLM_HPU_DSV41_PREFILL_INDEX_QUERY_TP):
        raise ValueError("Visible prefill index tiles require TP query partition and Full-index SRAM scores")
    if envs.VLLM_HPU_DSV41_PREFILL_INDEX_SRAM and not envs.VLLM_HPU_DSV41_PREFILL_INDEX_MME:
        raise ValueError("Full prefill SRAM index scoring requires the explicit prefill MME dispatch")
    if envs.VLLM_HPU_DSV41_PREFILL_REINDEX_SRAM:
        requirements = {
            "shared decoded Reindex keys": envs.VLLM_HPU_DSV41_PREFILL_REINDEX_REUSE,
            "native candidate gather": envs.VLLM_HPU_DSV41_PREFILL_CANDIDATE_GATHER,
        }
        missing = [name for name, enabled in requirements.items() if not enabled]
        if missing:
            raise ValueError("SRAM Reindex scoring requires: " + ", ".join(missing))
    if envs.VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION:
        requirements = {
            "compiled prefill regions": envs.VLLM_HPU_DSV41_PREFILL_REGIONS,
            "wo_a FP8 weights": envs.VLLM_HPU_DSV41_WO_A_FP8,
            "wo_a output roundtrip": envs.VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP,
            "vector group codec": envs.VLLM_HPU_DSV41_PREFILL_VECTOR_QUANT,
            "prepared FP8 attention weights": envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8,
        }
        missing = [name for name, enabled in requirements.items() if not enabled]
        if missing:
            raise ValueError("Compiled prefill output projection requires: " + ", ".join(missing))
    if envs.VLLM_HPU_DSV41_PREFILL_Q_PROJECTION:
        requirements = {
            "compiled prefill regions": envs.VLLM_HPU_DSV41_PREFILL_REGIONS,
            "native RoPE": envs.VLLM_HPU_DSV41_NATIVE_ROPE,
            "prefill RoPE": envs.VLLM_HPU_DSV41_PREFILL_ROPE,
            "prepared FP8 attention weights": envs.VLLM_HPU_DSV41_ATTN_DENSE_FP8,
        }
        missing = [name for name, enabled in requirements.items() if not enabled]
        if missing:
            raise ValueError("Compiled prefill Q projection requires: " + ", ".join(missing))
    if not envs.VLLM_HPU_DSV41_PREFILL_MHC_INPUT:
        return
    requirements = {
        "compiled prefill regions": envs.VLLM_HPU_DSV41_PREFILL_REGIONS,
        "native BF16 mHC boundary": envs.VLLM_HPU_DSV41_PREFILL_MHC_POST,
    }
    missing = [name for name, enabled in requirements.items() if not enabled]
    if missing:
        raise ValueError("Compiled prefill mHC input requires: " + ", ".join(missing))


def bf16_boundary(value):
    """Retain a producer's BF16 rounding when regions fuse its consumers."""
    if torch.compiler.is_compiling() and value.device.type == "hpu" and value.dtype == torch.bfloat16:
        return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(value.reshape(1, -1)).reshape(value.shape)
    return value


def _signature(value):
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), value.stride(), value.storage_offset(), value.dtype, value.device)
    if isinstance(value, (list, tuple)):
        return tuple(_signature(item) for item in value)
    return value


def clear_prefill_regions(owner):
    """The execution owner must drain users before invalidating weights/state."""
    owner.__dict__.pop("_prefill_tensor_regions", None)


def clear_prefill_function_regions():
    """Drop process-owned pure-function executors after their users drain.

    Function regions take every tensor, including weights and mutable state,
    as an explicit argument.  This lets equal shapes across decoder layers use
    one recipe instead of capturing forty layer owners.  The model generation
    still clears the executors on reload so a runtime/library generation never
    inherits an old compiled program.
    """
    _function_regions.clear()
    from vllm_gaudi.ops.deepseek_v41_prefill_index_scores import compiled_native_prefill_index_scores
    compiled_native_prefill_index_scores.cache_clear()


def prefill_function_region(function):
    """Compile a pure tensor function once per static contract.

    Unlike :func:`prefill_region`, the cache is intentionally shared across
    layer instances.  Callers must pass all data and weights explicitly; the
    decorated function must not close over request or layer state.
    """

    @wraps(function)
    def invoke(*args, **kwargs):
        if (torch.compiler.is_compiling() or not envs.VLLM_HPU_DSV41_PREFILL_REGIONS
                or (_qualified_function_regions is not None and function.__name__ not in _qualified_function_regions)):
            return function(*args, **kwargs)
        # The mHC function bodies select their exact native contract before
        # Dynamo tracing.  Keep diagnostic A/B executors disjoint when that
        # process-local switch changes; production processes keep one value.
        compile_mode = (envs.VLLM_HPU_DSV41_PREFILL_MHC_POST, envs.VLLM_HPU_DSV41_PREFILL_MHC_CONTROL_BF16,
                        envs.VLLM_HPU_DSV41_PREFILL_MHC_POST_PREPARE)
        key = (function.__module__, function.__qualname__, compile_mode, _signature(args),
               tuple((k, _signature(v)) for k, v in sorted(kwargs.items())))
        entry = _function_regions.get(key)
        if entry is None:
            name = f"v41_prefill_function_{function.__name__}_{next(_entries)}"
            cloned = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name,
                                  function.__defaults__, function.__closure__)
            cloned.__kwdefaults__ = function.__kwdefaults__
            entry = torch.compile(cloned, backend="hpu_backend", fullgraph=True, dynamic=False)
            _function_regions[key] = entry
            if len(_function_regions) > 64:
                _function_regions.popitem(last=False)
        else:
            _function_regions.move_to_end(key)
        try:
            return entry(*args, **kwargs)
        except RuntimeError as error:
            raise RuntimeError(f"Prefill region {function.__name__} failed for {key}") from error

    return invoke


def prefill_region(function):
    """Compile an explicitly selected method, with no process-wide cache change.

    Calls from an already compiled region inline the original tensor program.
    The caller, rather than a token-count heuristic, chooses this prefill entry.
    Separate code objects prevent unrelated layers/buckets exhausting Dynamo's
    shared guard limit. Values and expert IDs never enter the signature.
    """

    @wraps(function)
    def invoke(owner, *args, **kwargs):
        if (torch.compiler.is_compiling() or not envs.VLLM_HPU_DSV41_PREFILL_REGIONS
                or (_qualified_regions is not None and function.__name__ not in _qualified_regions)):
            return function(owner, *args, **kwargs)
        cache = owner.__dict__.setdefault("_prefill_tensor_regions", OrderedDict())
        key = (function.__name__, _signature(args), tuple((k, _signature(v)) for k, v in sorted(kwargs.items())))
        entry = cache.get(key)
        if entry is None:
            name = f"v41_prefill_{function.__name__}_{next(_entries)}"
            cloned = FunctionType(function.__code__.replace(co_name=name), function.__globals__, name,
                                  function.__defaults__, function.__closure__)
            cloned.__kwdefaults__ = function.__kwdefaults__
            entry = torch.compile(MethodType(cloned, owner), backend="hpu_backend", fullgraph=True, dynamic=False)
            cache[key] = entry
            # This bounds Python/recipe ownership; the framework retains any
            # executor still referenced by an in-flight launch.
            if len(cache) > 64:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return entry(*args, **kwargs)

    return invoke


@prefill_function_region
def prefill_output_projection(value, wo_a, scale_a, wo_b, scale_b):
    """Reuse one explicit wo_a/codec/wo_b recipe across prompt layer owners."""
    projected = torch.ops.custom_op.custom_deepseek_v41_woa_fp8_roundtrip_wide_gaudi2(value, wo_a, scale_a)
    return torch.ops.custom_op.custom_deepseek_v41_dense_fp8_gaudi2(projected, wo_b, scale_b)


@prefill_function_region
def prefill_q_projection(value, weight, channel_scale, positions, table):
    """Keep FP8 Q projection and the BF16/RoPE epilogue in one recipe.

    All layer weights and runtime positions are explicit bindings. The native
    epilogue preserves the BF16 projection boundary and separate FP32 rotary
    products used by large prefill, independently of the C1 decoder contract.
    """
    return torch.ops.custom_op.custom_deepseek_v41_prefill_q_projection_rope_gaudi2(value, weight, channel_scale,
                                                                                    positions, table).reshape(
                                                                                        value.shape[0], 32, 512)


@prefill_function_region
def prefill_main_workspace(packed, table, cache, indices, selected, logical, ratio):
    """Bind paged state explicitly, sharing one cache-assembly recipe per shape."""
    from vllm_gaudi.ops.deepseek_v41_math import unpack_fp4
    width = 128 // ratio
    shift = width.bit_length() - 1
    blocks = table.index_select(0, torch.bitwise_right_shift(logical, shift).long())
    physical = blocks * width + torch.bitwise_and(logical, width - 1)
    main = unpack_fp4(packed.index_select(0, physical.long()))
    offset = cache.shape[0]
    return (torch.cat((cache, main), 0), torch.cat((indices, torch.where(selected >= 0, selected + offset, -1)),
                                                   -1).int())


@prefill_function_region
def prefill_swa_workspace(kv, positions, swa, window_offsets, decoded_swa, decoded_offset):
    """Build the current prompt window while committing its unique ring tail.

    Cache tensors are explicit inputs so the compiled program does not retain
    a layer owner's mutable request bindings. Read the prefix before updating
    the ring; both packed and decoded state keep the same last-row ownership.
    """
    from vllm_gaudi.ops.deepseek_v41_math import pack_swa, unpack_swa
    window, ring_rows = window_offsets.numel(), swa.shape[0]
    prefix_positions = positions[:1].to(torch.int32) - (window - 1) + window_offsets[:-1]
    prefix_packed = swa.index_select(0, prefix_positions.remainder(ring_rows).long())
    cache = torch.cat((unpack_swa(prefix_packed), kv), 0)
    local = (torch.arange(positions.numel(), device=positions.device, dtype=torch.int32).unsqueeze(-1) +
             window_offsets.unsqueeze(0))
    logical = positions.unsqueeze(-1) - window + 1 + window_offsets.unsqueeze(0)
    indices = torch.where(logical >= 0, local, -1).int()
    tail = min(positions.numel(), ring_rows)
    tail_positions = positions[-tail:].remainder(ring_rows).long()
    packed_tail = pack_swa(kv[-tail:])
    swa.index_copy_(0, tail_positions, packed_tail)
    if decoded_swa is not None:
        decoded_swa.index_copy_(0, tail_positions + decoded_offset, unpack_swa(packed_tail))
    return cache, indices
