# SPDX-License-Identifier: Apache-2.0
"""Concurrent MoE candidate using the existing N256 FP8 numerical contract.

Grouping, quantization, matrices and ordered reduction form one compiled chain.
This is deliberately not selected by serving dispatch until real-chain gates
pass. Empty descriptors are explicit: they do not imply that MME was skipped.
"""
import torch

from vllm_gaudi.ops.deepseek_v41_route_blocks import device_route_blocks


def ordered_route_rows(ids):
    """Stable device permutation with one row per real route, including duplicates."""
    if ids.ndim != 2 or ids.shape[1] != 6 or not 1 <= ids.shape[0] <= 64:
        raise ValueError("Ordered routes require B1..64 top6")
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Route IDs must be integers")
    size = ids.numel()
    # IDs come from the validated top6 producer. Unique low bits make ties
    # deterministic without depending on backend stable-sort support.
    flat = ids.reshape(-1).int()
    original = torch.arange(size, device=ids.device, dtype=torch.int32)
    order = (flat * size + original).sort().values.remainder(size).long()
    inverse = torch.empty(size, dtype=torch.int64, device=ids.device)
    inverse.scatter_(0, order, original.long())
    return flat.index_select(0, order).reshape(1, -1), order[:, None], inverse


def grouped_decode(value,
                   ids,
                   routing,
                   q13,
                   q2,
                   s13,
                   s2,
                   lookup,
                   channel13,
                   channel2,
                   *,
                   rows,
                   reuse_weights=False,
                   native_pack=False):
    if (rows not in (1, 4, 8, 16) or value.ndim != 2 or not 1 <= value.shape[0] <= 64
            or ids.shape != (value.shape[0], 6) or routing.shape != ids.shape):
        raise ValueError("Grouped decode requires B1..64 top6 and 1/4/8/16 rows")
    if native_pack:
        if rows != 1 or not reuse_weights:
            raise ValueError("Native route packing requires the unpadded sorted-route consumer")
        quant, scales = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(value)
        flat_ids = ids.reshape(1, -1).int()
        inverse = torch.ops.custom_op.custom_deepseek_v41_route_order_gaudi2(flat_ids)
        activation, sx, experts, routes = torch.ops.custom_op.custom_deepseek_v41_route_pack_gaudi2(
            quant.view(torch.uint8), scales, flat_ids, routing.reshape(1, -1), inverse)
        result = torch.ops.custom_op.custom_deepseek_v41_reused_n256_fp8_gaudi2(activation.view(torch.float8_e4m3fn),
                                                                                sx, experts, routes, q13, q2, s13, s2,
                                                                                lookup, channel13, channel2)
        return torch.ops.custom_op.custom_deepseek_v41_route_reduce_gaudi2(result, inverse)
    if reuse_weights:
        if rows != 1:
            raise ValueError("Register reuse requires unpadded route rows")
        experts, slots, inverse = ordered_route_rows(ids)
    elif rows == 1:
        # Direct-route control: the same bounded SRAM graph, without grouping
        # or padded matrix rows. It isolates placement from expert reuse.
        experts = ids.reshape(1, -1).int()
        inverse = torch.arange(ids.numel(), dtype=torch.int64, device=ids.device)
        slots = inverse[:, None]
    else:
        experts, slots, inverse, _ = device_route_blocks(ids, q13.shape[0], rows)
    # Quantize once per original token using exactly the parent MoE kernel.
    quant, scales = torch.ops.custom_op.custom_deepseek_v41_dynamic_quant_bf16_gaudi2(value)
    groups = slots.shape[0]
    safe = slots.clamp(min=0).flatten()
    token = safe // 6
    # Byte gather preserves native Gaudi FP8 encodings, with no FP8 arithmetic
    # or conversion in this routing operation.
    activation = quant.view(torch.uint8).index_select(0, token).reshape(groups, rows, -1)
    activation = activation.view(torch.float8_e4m3fn).contiguous()
    sx = scales.index_select(0, token).reshape(groups, rows, 1).contiguous()
    routes = routing.flatten().index_select(0, safe).reshape(groups, rows).masked_fill(slots < 0, 0).contiguous()
    experts = torch.where(slots[:, :1].T >= 0, experts, -1).contiguous()
    operation = (torch.ops.custom_op.custom_deepseek_v41_reused_n256_fp8_gaudi2
                 if reuse_weights else torch.ops.custom_op.custom_deepseek_v41_grouped_n256_fp8_gaudi2)
    result = operation(activation, sx, experts, routes, q13, q2, s13, s2, lookup, channel13, channel2)
    ordered = result.reshape(-1, value.shape[1]).index_select(0, inverse).reshape(value.shape[0], 6, -1)
    total = ordered[:, 0].float()
    for index in range(1, 6):
        total = total + ordered[:, index].float()
    return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(total.bfloat16().reshape(1,
                                                                                                 -1)).reshape_as(value)
