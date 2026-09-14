# SPDX-License-Identifier: Apache-2.0
"""C2–C6 expert reuse without changing token-local routing or rounding.

An expert selected by several tokens owns one decoded matrix. Repeated IDs
within a token retain separate occurrence numbers, so their independently
rounded W2 inputs never collide. All shapes remain static during replay.
"""

import torch
import torch.nn.functional as F


def expert_owners(ids):
    tokens, experts = ids.shape
    flat = ids.reshape(-1)
    slots = torch.arange(flat.numel(), dtype=torch.int32, device=ids.device)
    equal = flat[:, None] == flat[None, :]
    earlier = slots[None, :] < slots[:, None]
    same_token = slots[:, None] // experts == slots[None, :] // experts
    occurrence = (equal & earlier & same_token).to(torch.int32).sum(-1, dtype=torch.int32)
    matches = equal & (occurrence[:, None] == occurrence[None, :])
    owner = torch.where(matches, slots[None, :], flat.numel()).amin(-1)
    active = (owner == slots).to(torch.int32).reshape(1, -1).contiguous()
    token = slots // experts
    rows = (owner * tokens + token).to(torch.int64)
    return active, rows


def shared_expert_moe(value, ids, routing, weights, lookup, normal):
    tokens, experts = ids.shape
    batches = tokens * experts
    active, rows = expert_owners(ids)
    ids = ids.to(torch.int32).reshape(1, -1).contiguous()
    op = torch.ops.custom_op.custom_deepseek_v41_mxfp4_shared_linear_bf16_gaudi2
    first = op(value.unsqueeze(0), ids, active, weights.w13_q16, weights.w13_s16, lookup, normal)
    # Inactive batches are unspecified internal storage. Every selected row
    # belongs to an active owner; no reduction may consume inactive batches.
    gate_up = first.flatten(0, 1).index_select(0, rows).reshape(tokens, experts, -1)
    width = gate_up.shape[-1] // 2
    gate = gate_up[..., :width].float().clamp(max=10.0)
    up = gate_up[..., width:].float().clamp(-10.0, 10.0)
    middle = (F.silu(gate) * up * routing.unsqueeze(-1)).to(torch.bfloat16)
    # Keep the explicit BF16 boundary used by the original compound op.
    middle = torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(middle.reshape(1, -1)).reshape(batches, width)
    packed = torch.zeros((batches * tokens, width), dtype=value.dtype, device=value.device)
    packed = packed.index_copy(0, rows, middle).reshape(batches, tokens, width)
    down = op(packed, ids, active, weights.w2_q16, weights.w2_s16, lookup, normal)
    down = down.flatten(0, 1).index_select(0, rows).reshape(tokens, experts, -1).float()
    result = down[:, 0]
    for expert in range(1, experts):
        result = result + down[:, expert]
    return torch.ops.custom_op.custom_deepseek_v41_bf16_identity_gaudi2(result.to(value.dtype).reshape(
        1, -1)).reshape_as(value)
