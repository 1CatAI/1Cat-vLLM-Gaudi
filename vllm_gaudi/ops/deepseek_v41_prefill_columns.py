# SPDX-License-Identifier: Apache-2.0
"""Restore decoder column order on activations and after ordered reduction."""

import functools
from types import FunctionType

import torch
import torch.nn.functional as F


def restore_expert_columns(value):
    """Undo even/odd column separation independently within every N256 tile."""
    return value.reshape(-1, 2, 128).transpose(1, 2).reshape(value.shape)


def permuted_body_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales, routed, indices):
    if not normal_scales:
        raise ValueError("Interleaved expert columns require qualified normal exponent codes")
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    groups, rows = local_slots.shape
    safe = local_slots.clamp(min=0)
    selected = value.index_select(0, (safe // 6).reshape(-1)).reshape(groups, rows, value.shape[-1])
    weight13 = torch.ops.custom_op.custom_deepseek_v41_prefill_permuted_bf16_gaudi2(local_experts, q13, s13, lookup)
    projected = restore_expert_columns(torch.bmm(selected, weight13))
    width = projected.shape[-1] // 2
    gate = projected[..., :width].float().clamp(max=10.0)
    up = projected[..., width:].float().clamp(-10.0, 10.0)
    route = routing.reshape(-1).index_select(0, safe.reshape(-1))
    route = route.reshape(groups, rows).masked_fill(local_slots < 0, 0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, rows, width)
    weight2 = torch.ops.custom_op.custom_deepseek_v41_prefill_permuted_bf16_gaudi2(local_experts, q2, s2, lookup)
    output = torch.bmm(middle, weight2).reshape(-1, value.shape[-1])
    # The private route workspace retains interleaved columns until all six
    # routes have been reduced in their original order.
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(routed, output,
                                                                              local_slots.flatten().int())


def permuted_body_single_fp8_write(value, routing, slots, expert_ids, q13, q2, s13, s2, lookup, normal_scales, routed,
                                   channel13, high, high_scale, indices):
    """Use one pre-quantized W13 GEMM within the occupancy-bucket route plan."""
    if not normal_scales:
        raise ValueError("Interleaved expert columns require qualified normal exponent codes")
    local_slots = slots.index_select(0, indices.long())
    local_experts = expert_ids.index_select(1, indices.long())
    groups, rows = local_slots.shape
    safe = local_slots.clamp(min=0)
    token_rows = (safe.reshape(-1) // 6).long()
    selected = high.index_select(0, token_rows).reshape(groups, rows, value.shape[-1])
    selected_scale = high_scale.index_select(0, token_rows).reshape(groups, rows, 1)
    safe_experts = local_experts.clamp(min=0)
    weight13 = torch.ops.custom_op.custom_deepseek_v41_expert_n256_fp8_gaudi2(safe_experts, q13, s13, lookup, True)
    weight_scale = channel13.index_select(0,
                                          safe_experts.reshape(-1).long()).reshape(groups, 1,
                                                                                   weight13.shape[-1]).float()
    projected = torch.ops.hpu.fp8_gemm_v2(selected, False, weight13, False, None, torch.bfloat16, selected_scale,
                                          weight_scale, None, False)
    width = projected.shape[-1] // 2
    gate = projected[..., :width].float().clamp(max=10.0)
    up = projected[..., width:].float().clamp(-10.0, 10.0)
    route = routing.reshape(-1).index_select(0, safe.reshape(-1))
    route = route.reshape(groups, rows).masked_fill(local_slots < 0, 0)
    middle = (F.silu(gate) * up * route.unsqueeze(-1)).to(torch.bfloat16)
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(
        groups, rows, width)
    weight2 = torch.ops.custom_op.custom_deepseek_v41_prefill_permuted_bf16_gaudi2(local_experts, q2, s2, lookup)
    output = torch.bmm(middle, weight2).reshape(-1, value.shape[-1])
    return torch.ops.custom_op.custom_deepseek_v41_prefill_route_write_gaudi2(routed, output,
                                                                              local_slots.flatten().int())


def permuted_ordered_reduce(value):
    from vllm_gaudi.ops.deepseek_v41_grouped_prefill import ordered_reduce

    return restore_expert_columns(ordered_reduce(value))


@functools.lru_cache(maxsize=32)
def compiled_permuted_write_body(signature):
    entry = FunctionType(permuted_body_write.__code__.replace(co_name=f"permuted_expert_write_{signature}"),
                         permuted_body_write.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_permuted_single_fp8_write_body(signature):
    entry = FunctionType(permuted_body_single_fp8_write.__code__.replace(co_name=f"permuted_single_fp8_{signature}"),
                         permuted_body_single_fp8_write.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)


@functools.lru_cache(maxsize=32)
def compiled_permuted_reduce(signature):
    entry = FunctionType(permuted_ordered_reduce.__code__.replace(co_name=f"permuted_expert_reduce_{signature}"),
                         permuted_ordered_reduce.__globals__)
    return torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
