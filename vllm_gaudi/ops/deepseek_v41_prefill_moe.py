# SPDX-License-Identifier: Apache-2.0
"""Large-M V4.1 MXFP4 MoE using the resident decode weight allocation.

The V4.1 rank files store a lossless Q16/S16 lane permutation for the C1
decoder.  Habana's stock MXFP4 FusedMoE expects checkpoint-order packed bytes.
For prefill, restore bounded expert ranges immediately before their fused MoE
consumer.  The restored tensors are recipe temporaries, so the model keeps one
compressed expert-weight allocation and the compiler can reuse the range
workspace between calls.
"""

from __future__ import annotations

import functools
import torch

from vllm_gaudi.extension.ops import _mxfp4_fused_fwd
from vllm_gaudi.ops.deepseek_v4_mxfp4 import (
    restore_mxfp4_scale_u8,
    restore_mxfp4_u8,
)


PREFILL_EXPERT_CHUNK = 64
PREFILL_BF16_EXPERT_CHUNK = 24


def _views(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(value[index] for index in range(value.shape[0]))


def _i16_little_endian_bytes(value: torch.Tensor) -> torch.Tensor:
    """Expand I16 storage bytes without a dtype reinterpret-cast node.

    Synapse can execute ``Tensor.view(torch.uint8)`` in an isolated recipe,
    but a large prefill followed by the native C1 recipe leaves that eager
    reinterpret node with an invalid internal dtype (524288).  Integer masks
    express the same little-endian bytes as normal arithmetic and keep the
    produced U8 tensor explicit in the graph.
    """
    if value.dtype != torch.int16:
        raise ValueError("byte expansion requires int16 storage")
    low = torch.bitwise_and(value, 255).to(torch.uint8)
    high = torch.bitwise_right_shift(value, 8).to(torch.uint8)
    return torch.stack((low, high), dim=-1).reshape(*value.shape[:-1], value.shape[-1] * 2)


def restore_n256_mxfp4_u8(q16: torch.Tensor) -> torch.Tensor:
    """Reverse the resident N256 nibble permutation using graph-visible ops."""
    if q16.dtype != torch.int16 or q16.ndim != 3 or q16.shape[-1] % 8192:
        raise ValueError("N256 weights must be I16 [E,N/256,K*64]")
    experts, blocks, stream = q16.shape
    k = stream // 64
    packed = _i16_little_endian_bytes(q16).reshape(experts, blocks, k, 128)
    even, odd = packed[:, :, 0::2], packed[:, :, 1::2]
    halves = []
    for half in range(2):
        lane = slice(half * 64, (half + 1) * 64)
        low = (even[..., lane] & 15) | ((odd[..., lane] & 15) << 4)
        high = (even[..., lane] >> 4) | (odd[..., lane] & 240)
        halves.append(torch.stack((low, high), dim=-1).reshape(experts, blocks, k // 2, 128))
    # Interleave the two original N128 row blocks, then undo their K-major
    # Q16 permutation to recover checkpoint-order [E,N,K/2] bytes.
    raw = torch.stack(halves, dim=2).reshape(experts, blocks * 2, k // 2, 128)
    # ``raw`` is the byte view of the old N128 Q16 tensor.  Apply the inverse
    # Q16 permutation directly in U8 instead of packing it back to I16 only to
    # reinterpret it again in ``restore_mxfp4_u8``.
    byte_stream = raw.reshape(experts, blocks * 2, k // 2, 64, 2)
    return byte_stream.permute(0, 1, 3, 4, 2).reshape(experts, blocks * 256, k // 2).contiguous()


def restore_n256_scale_u8(planes: torch.Tensor) -> torch.Tensor:
    """Extract the original E8M0 plane retained beside N256 FP8 offsets."""
    if planes.dtype != torch.int16 or planes.ndim != 3 or planes.shape[-1] % 1024:
        raise ValueError("N256 scale planes must be I16 [E,N/256,K*8]")
    experts, blocks, stream = planes.shape
    k = stream // 8
    original = _i16_little_endian_bytes(planes).reshape(experts, blocks, k // 32, 512)[..., :256]
    return (original.reshape(experts, blocks, k // 32, 2, 128)
            .permute(0, 1, 3, 4, 2).reshape(experts, blocks * 256, k // 32).contiguous())


def q16_chunked_mxfp4_moe(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w13_q16: torch.Tensor,
    w2_q16: torch.Tensor,
    w13_s16: torch.Tensor,
    w2_s16: torch.Tensor,
    *,
    expert_chunk: int = PREFILL_EXPERT_CHUNK,
) -> torch.Tensor:
    """Run a large-M fused MoE without a persistent second weight layout."""
    experts = w13_q16.shape[0]
    if (experts != w2_q16.shape[0] or experts != w13_s16.shape[0]
            or experts != w2_s16.shape[0] or experts % expert_chunk):
        raise ValueError("V4.1 prefill expert tensors require equal, exactly chunked expert axes")
    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 6:
        raise ValueError("The large-M V4.1 prefill path requires more than six tokens")
    if expert_ids.shape != (hidden_states.shape[0], 6) or router_weights.shape != expert_ids.shape:
        raise ValueError("V4.1 prefill routing must be [tokens,6]")

    # The stock kernel owns expert filtering through experts_min/max.  Keeping
    # global IDs avoids a routing remap and preserves their order.  Each call
    # sees only the matching resident expert range.
    output_f32 = None
    routing_bf16 = router_weights.to(torch.bfloat16)
    ids_i32 = expert_ids.to(torch.int32)
    for start in range(0, experts, expert_chunk):
        stop = start + expert_chunk
        n256 = w13_s16.dtype == torch.int16
        restore_q = restore_n256_mxfp4_u8 if n256 else restore_mxfp4_u8
        restore_s = restore_n256_scale_u8 if n256 else restore_mxfp4_scale_u8
        w13 = restore_q(w13_q16[start:stop])
        w2 = restore_q(w2_q16[start:stop])
        s13 = restore_s(w13_s16[start:stop])
        s2 = restore_s(w2_s16[start:stop])
        partial = _mxfp4_fused_fwd(
            hidden_states,
            ids_i32,
            routing_bf16,
            _views(w13),
            _views(w2),
            _views(s13),
            _views(s2),
            32,
            "silu",
            start,
            stop - 1,
            0,
            0,
        )
        # Keep inter-range accumulation in FP32.  Each stock invocation has
        # already applied the matching routing weights; one final BF16 round
        # avoids adding a BF16 boundary for every expert range.
        partial_f32 = partial.float()
        output_f32 = partial_f32 if output_f32 is None else output_f32 + partial_f32
    assert output_f32 is not None
    return output_f32.to(hidden_states.dtype)


def q16_chunked_bf16_moe(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w13_q16: torch.Tensor,
    w2_q16: torch.Tensor,
    w13_s16: torch.Tensor,
    w2_s16: torch.Tensor,
    lookup: torch.Tensor,
    *,
    expert_chunk: int = PREFILL_BF16_EXPERT_CHUNK,
    normal_scales: bool = False,
) -> torch.Tensor:
    """Bridge resident N256 weights to the ordinary BF16 fused-MoE.

    The version-locked Synapse used by native decode replay predates the
    packed-MXFP4 graph dtype.  Its standard BF16 graph types and the V4.1 TPC
    decoder are both supported, however.  Decode each expert range exactly
    once, transpose it to the ordinary fused-MoE weight contract, consume it
    immediately, and retain only the resident Q16/S16 allocation between
    calls.  This is a prompt-only compatibility path; C1 stays on N256.
    """
    experts = w13_q16.shape[0]
    if (experts != w2_q16.shape[0] or experts != w13_s16.shape[0]
            or experts != w2_s16.shape[0] or experts % expert_chunk):
        raise ValueError("V4.1 BF16 prefill bridge requires equal, exactly chunked expert axes")
    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 6:
        raise ValueError("The BF16 prefill bridge requires more than six tokens")
    if expert_ids.shape != (hidden_states.shape[0], 6) or router_weights.shape != expert_ids.shape:
        raise ValueError("V4.1 prefill routing must be [tokens,6]")
    if w13_s16.dtype != torch.int16 or w2_s16.dtype != torch.int16:
        raise ValueError("The BF16 prefill bridge requires resident N256 scale planes")

    output_f32 = None
    routing_bf16 = router_weights.to(torch.bfloat16)
    # The ordinary BF16 fused-MoE contract uses I64 routing tables (the
    # MXFP4 overload above is the one that requires I32).
    ids_i64 = expert_ids.to(torch.int64)
    # Local IDs address each sliced Q16/S16 range.  The fused-MoE keeps global
    # routing IDs and receives the corresponding experts_min/max interval.
    local_decode_ids = torch.arange(expert_chunk, device=hidden_states.device,
                                    dtype=torch.int32).reshape(1, expert_chunk)
    for start in range(0, experts, expert_chunk):
        stop = start + expert_chunk
        w13_kn = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
            local_decode_ids, w13_q16[start:stop], w13_s16[start:stop], lookup,
            normal_scales)
        w2_kn = torch.ops.custom_op.custom_deepseek_v41_expert_n256_bf16_gaudi2(
            local_decode_ids, w2_q16[start:stop], w2_s16[start:stop], lookup,
            normal_scales)
        # Standard fused-MoE with permuted_weights=True consumes [N,K].
        w13_nk = w13_kn.transpose(1, 2).contiguous()
        w2_nk = w2_kn.transpose(1, 2).contiguous()
        partial = torch.ops.hpu.mixture_of_experts(
            hidden_states=hidden_states,
            expert_routing_table=ids_i64,
            router_weights=routing_bf16,
            w12=_views(w13_nk),
            w3=_views(w2_nk),
            permuted_weights=True,
            activation="silu",
            experts_min=start,
            experts_max=stop - 1,
        )
        partial_f32 = partial.float()
        output_f32 = partial_f32 if output_f32 is None else output_f32 + partial_f32
    assert output_f32 is not None
    return output_f32.to(hidden_states.dtype)


@functools.lru_cache(maxsize=1)
def _compiled_q16_prefill_moe():
    from vllm_gaudi.extension.ops import _mxfp4_hpu_backend

    return torch.compile(
        q16_chunked_mxfp4_moe,
        backend=_mxfp4_hpu_backend,
        fullgraph=True,
        dynamic=False,
    )


def run_q16_prefill_moe(*args):
    """Inline into an outer prefill graph, or own one reusable inner recipe."""
    if torch.compiler.is_compiling():
        return q16_chunked_mxfp4_moe(*args)
    return _compiled_q16_prefill_moe()(*args)
