# SPDX-License-Identifier: Apache-2.0
"""Lossless load-time BF16 weight layout for a gated native dense consumer."""
from functools import lru_cache
import json
from pathlib import Path

import torch


@lru_cache(maxsize=4)
def lane_map(path):
    import numpy as np

    proof = json.loads(Path(path).read_text())
    values = np.asarray(proof.get('lane_map'), dtype=np.int64)
    if (proof.get('status') != 'WEIGHT_BYTES_FUNCTIONAL_ONLY'
            or not proof.get('device_lane_permutation_qualified')
            or len(proof.get('actual_matrices', [])) != 4
            or not all(row.get('bf16_bytes_exact') for row in proof['actual_matrices'])
            or values.shape != (256,) or not np.array_equal(np.sort(values), np.arange(256))):
        raise ValueError('Lossless dense packing requires the qualified native lane/actual-weight proof')
    return values


def prepare(owner, weight, prefix):
    """Retain the reference BF16 allocation until complete-chain admission.

    Immutable planes are prepared once, before capture, for either TP size.
    Extra residency is a component-only tradeoff; full-serving memory
    admission must be resolved before selecting this storage by default.
    """
    import numpy as np
    from vllm_gaudi import envs

    if (weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[0] % 32
            or weight.shape[1] % 256 or not envs.VLLM_HPU_DSV41_DSPARK_DENSE_BITS12_MAP):
        raise ValueError('Lossless dense planes require BF16 N32/K256 weights and a lane proof')
    if prefix+'_high' in owner._buffers or prefix+'_low' in owner._buffers:
        raise ValueError('Prepare a new weight owner after packing/generation changes')
    mapping = lane_map(envs.VLLM_HPU_DSV41_DSPARK_DENSE_BITS12_MAP)
    bits = weight.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    if np.any(bits & 15):
        raise ValueError('The prepared source weights are not losslessly representable by the 12-bit layout')
    high = (bits >> 8).astype(np.uint8)
    wanted = ((bits >> 4) & 15).astype(np.uint8).reshape(-1, 256)
    shuffled = np.empty_like(wanted)
    shuffled[:, mapping] = wanted
    low = (shuffled[:, ::2] | (shuffled[:, 1::2] << 4)).reshape(bits.shape[0], -1)
    owner.register_buffer(prefix+'_high', torch.from_numpy(high.view(np.int8)).to(weight.device), False)
    owner.register_buffer(prefix+'_low', torch.from_numpy(low.view(np.int8)).to(weight.device), False)


def project(value, owner, prefix):
    """One native compound: decoded private BF16 operand and its MME consumer."""
    return torch.ops.custom_op.custom_deepseek_v41_dense_bits12_projection_bf16_gaudi2(
        value.contiguous(), getattr(owner, prefix+'_high'), getattr(owner, prefix+'_low'))
