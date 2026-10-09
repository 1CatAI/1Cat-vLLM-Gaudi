# SPDX-License-Identifier: Apache-2.0
"""Prepare draft shared projections for the maintained C1 FP8 shared body.

Original BF16 modules remain owned by the draft. Both native comparison arms
can therefore capture from the same resident checkpoint without reloading it.
"""
import copy
import hashlib


def prepare(moe):
    import numpy as np
    import torch
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import covering_scale, encode_gaudi2

    if moe.topk != 3 or moe.weights.gate.weight.shape != (128, 5120):
        raise ValueError('Draft shared FP8 requires the checkpoint top3/E128/H5120 contract')

    def views(module):
        result = copy.copy(module)
        result._buffers = module._buffers.copy()
        result._parameters = module._parameters.copy()
        result._modules = {name: views(value) if value is not None else None
                           for name, value in module._modules.items()}
        return result

    candidate = views(moe)
    candidate.shared_gate_up = True
    records = {}
    for name in ('w1', 'w3', 'w2'):
        projection = getattr(candidate.weights.shared_experts, name)
        weight = projection.weight
        if weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError('Draft shared preparation requires decoded BF16 checkpoint weights')
        source = weight.detach().cpu().float().numpy()
        channels = covering_scale(np.max(np.abs(source), axis=1, keepdims=True))
        encoded = encode_gaudi2(source / channels)
        records[name] = dict(shape=list(source.shape), source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                             encoded_sha256=hashlib.sha256(encoded.tobytes()).hexdigest())
        projection.weight = torch.from_numpy(encoded).view(torch.float8_e4m3fn).to(weight.device)
        projection.register_buffer('channel_scale', torch.from_numpy(channels.T.copy()).to(weight.device), False)
        projection.dense_fp8 = projection.dense_fp8_direct_input = True

    candidate.prepare_shared_gate_up_weight()
    for name in ('shared_gate_up_weight', 'shared_gate_up_channel', 'shared_down_weight'):
        moe.register_buffer('draft_' + name, getattr(candidate, name), False)
    moe.register_buffer('draft_shared_down_channel', candidate.weights.shared_experts.w2.channel_scale, False)
    moe.draft_shared_preparation = dict(quantization='C1 channel power-of-two E4M3 RNE/FTZ',
                                        original_weights_retained=True, projections=records)
