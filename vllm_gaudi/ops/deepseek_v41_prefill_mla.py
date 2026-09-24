# SPDX-License-Identifier: Apache-2.0
"""BF16 sparse MLA with an FP32 softmax sink and bounded gathered operands."""

import math

import torch


def _bmm_f32(a, b, transpose_b):
    return torch.ops.custom_op.custom_deepseek_v41_prefill_bmm_f32_gaudi2(a.contiguous(), b.contiguous(), transpose_b)


def sparse_prefill_mla(query, cache, indices, sink, lengths, *, sm_scale=None, query_tile=1024):
    """Keep the sink in FP32 logits without adding a synthetic K/V row.

    The vendor BF16 path rounds the scaling constant, scales Q, and rounds Q
    to BF16 before the FP32 QK product. Preserve those boundaries, along with
    BF16 probabilities and PV operands. Reducing over the selected K/V rows
    can change the last bits of accumulation compared with appending zeros.
    """
    if query.ndim != 3 or query.shape[2] != 512 or min(query.shape) < 1 or query.shape[1] > 128:
        raise ValueError("Sparse MLA query requires nonempty [tokens,heads<=128,512]")
    tokens, heads, width = query.shape
    if (cache.ndim != 2 or cache.shape[0] < 1 or cache.shape[1] != width or indices.ndim != 2
            or indices.shape[0] != tokens or not 1 <= indices.shape[1] <= 640 or sink.shape != (heads, )
            or lengths.shape != (tokens, )):
        raise ValueError("Sparse MLA cache, indices, sink or lengths contract changed")
    expected = (torch.bfloat16, torch.bfloat16, torch.int32, torch.float32, torch.int32)
    tensors = (query, cache, indices, sink, lengths)
    if any(t.dtype != dtype or t.device != query.device or t.requires_grad for t, dtype in zip(tensors, expected)):
        raise ValueError("Sparse MLA requires inference BF16 Q/KV, I32 IDs/lengths and FP32 sink")
    factor = width**-0.5 if sm_scale is None else sm_scale
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Sparse MLA scale must be finite and positive")
    if not isinstance(query_tile, int) or not 1 <= query_tile <= 2048:
        raise ValueError("Sparse MLA query tile must be bounded by the native BMM batch capacity")
    if query.device.type != "hpu":
        raise ValueError("Sparse prefill MLA requires HPU operands")

    columns, zero_row = indices.shape[1], cache.shape[0]
    padded = torch.cat((cache, cache.new_zeros((1, width))), 0)
    # The separate V IDs keep the compiler from retaining and spilling K
    # between QK and PV. Each consumer instead gathers its own bounded tile.
    value_ids = indices.clone()
    scale = query.new_full((), factor).float()
    offsets = torch.arange(columns, device=indices.device)[None, :]
    outputs = []
    for start in range(0, tokens, query_tile):
        stop = min(start + query_tile, tokens)
        rows = stop - start
        ids, lens = indices[start:stop], lengths[start:stop]
        valid = (ids >= 0) & (ids < zero_row) & (offsets < lens[:, None])
        safe = torch.where(valid, ids, zero_row).long()
        keys = padded.index_select(0, safe.flatten()).reshape(rows, columns, width)
        scaled_query = (query[start:stop].float() * scale).to(torch.bfloat16)
        logits = _bmm_f32(scaled_query, keys, True).masked_fill(~valid[:, None, :], -torch.inf)
        logits = torch.cat((logits, sink.reshape(1, heads, 1).expand(rows, heads, 1)), -1)
        probability = torch.softmax(logits, -1)[..., :columns].to(torch.bfloat16)

        ids = value_ids[start:stop]
        valid = (ids >= 0) & (ids < zero_row) & (offsets < lens[:, None])
        safe = torch.where(valid, ids, zero_row).long()
        values = padded.index_select(0, safe.flatten()).reshape(rows, columns, width)
        outputs.append(_bmm_f32(probability, values, False).to(torch.bfloat16))
    return torch.cat(outputs, 0)
