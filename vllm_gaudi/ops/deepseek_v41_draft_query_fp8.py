# SPDX-License-Identifier: Apache-2.0
"""Prepare draft Q weights for the maintained C1 FP8 projection/RoPE op."""
import hashlib


def prepare(attention):
    import numpy as np
    import torch
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import covering_scale, encode_gaudi2

    weight = attention.weights.wq_b.weight
    if (attention.layer < 40 or weight.dtype != torch.bfloat16
            or weight.shape != (32768 // attention.tensor_parallel_size, 1280)
            or not attention.native_rope):
        raise ValueError('Draft Q FP8 requires the checkpoint BF16 TP weight and common native RoPE table')
    source = weight.detach().cpu().float().numpy()
    channels = covering_scale(np.max(np.abs(source), axis=1, keepdims=True))
    codes = encode_gaudi2(source / channels)
    attention.register_buffer('draft_query_weight',
                              torch.from_numpy(codes).view(torch.float8_e4m3fn).to(weight.device), False)
    attention.register_buffer('draft_query_channel', torch.from_numpy(channels.T.copy()).to(weight.device), False)
    attention.draft_query_preparation = dict(source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                                             encoded_sha256=hashlib.sha256(codes.tobytes()).hexdigest(),
                                             original_weights_retained=True,
                                             quantization='C1 channel power-of-two E4M3 RNE/FTZ')
