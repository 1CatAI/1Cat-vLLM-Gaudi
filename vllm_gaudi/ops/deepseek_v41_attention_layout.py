# SPDX-License-Identifier: Apache-2.0
"""Offline head interleaving for a transposed MME attention chain.

Within each existing eight-head output group, the physical order is
``[dimension, head]``. QK consumes it with transpose-A; PV exchanges its
operands and uses transpose-A/B to produce exactly the output consumer's
order. The checkpoint's per-head group-32 activation codec is unchanged.
"""

import torch


HEAD_GROUP = 8
HEAD_DIM = 512


def interleave_query_weight(weight, heads):
    if weight.ndim != 2 or weight.shape[0] != heads * HEAD_DIM or heads % HEAD_GROUP:
        raise ValueError("Query layout requires whole groups of eight 512-dimensional heads")
    groups = heads // HEAD_GROUP
    return weight.reshape(groups, HEAD_GROUP, HEAD_DIM, -1).permute(0, 2, 1, 3).reshape_as(weight).contiguous()


def interleave_output_weight(weight):
    if weight.ndim != 3 or weight.shape[-1] != HEAD_GROUP * HEAD_DIM:
        raise ValueError("Output layout requires [groups, output channels, 8*512]")
    return weight.reshape(*weight.shape[:2], HEAD_GROUP, HEAD_DIM).permute(0, 1, 3, 2).reshape_as(weight).contiguous()


def pack_heads(value):
    if value.ndim != 3 or value.shape[-1] != HEAD_DIM or value.shape[1] % HEAD_GROUP:
        raise ValueError("Attention layout requires [tokens, 8*groups, 512]")
    return value.reshape(value.shape[0], -1, HEAD_GROUP, HEAD_DIM).transpose(-1, -2).contiguous()


def unpack_heads(value):
    if value.ndim != 4 or value.shape[-2:] != (HEAD_DIM, HEAD_GROUP):
        raise ValueError("Interleaved attention layout requires [tokens, groups, 512, 8]")
    return value.transpose(-1, -2).reshape(value.shape[0], -1, HEAD_DIM).contiguous()


def group32_membership(heads):
    """Return original codec group IDs in the interleaved physical order."""
    ids = torch.arange(heads * HEAD_DIM).reshape(1, heads, HEAD_DIM) // 32
    return pack_heads(ids).reshape(heads // HEAD_GROUP, HEAD_DIM // 32, 32, HEAD_GROUP)
