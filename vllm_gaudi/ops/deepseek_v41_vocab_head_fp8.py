# SPDX-License-Identifier: Apache-2.0
"""Bounded preparation of an optional DSpark vocabulary matrix.

BF16 checkpoint weights remain the reference and the prefill operand. The
additional bank uses the same finite Gaudi2 channel encoding as dense C1
projections. This numerical candidate requires conditional p/q qualification.
"""
import numpy as np
import torch

from vllm_gaudi.ops.deepseek_v41_woa_fp8 import covering_scale, encode_gaudi2


def prepare_vocab_head_fp8(module):
    if (getattr(module, 'dspark_vocab_fp8_weight', None) is not None
            and getattr(module, 'dspark_vocab_fp8_source_weight', None) is module.weight):
        return
    for name in ('dspark_vocab_fp8_weight', 'dspark_vocab_fp8_scale'):
        module._buffers.pop(name, None)
    weight = module.weight
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[1] != 5120:
        raise ValueError('Vocabulary FP8 requires the unchanged BF16 checkpoint head')
    source = weight.detach().cpu()
    packed = np.empty(tuple(weight.shape), dtype=np.uint8)
    channel = np.empty((weight.shape[0], 1), dtype=np.float32)
    changed, error_energy, reference_energy, maximum_abs = 0, 0., 0., 0.
    from vllm_gaudi.ops.deepseek_v41_woa_fp8 import decode_gaudi2

    # Only256rows are expanded to FP32 at a time; no full FP32 bank is kept.
    for first in range(0, weight.shape[0], 256):
        last = min(first + 256, weight.shape[0])
        values = source[first:last].float().numpy()
        scales = covering_scale(np.max(np.abs(values), axis=1, keepdims=True))
        codes = encode_gaudi2(values / scales)
        packed[first:last], channel[first:last] = codes, scales
        difference = decode_gaudi2(codes) * scales - values
        changed += int(np.count_nonzero(difference))
        error_energy += float(np.sum(difference.astype(np.float64) ** 2))
        reference_energy += float(np.sum(values.astype(np.float64) ** 2))
        maximum_abs = max(maximum_abs, float(np.abs(difference).max()))
    module.register_buffer('dspark_vocab_fp8_weight',
                           torch.from_numpy(packed).view(torch.float8_e4m3fn).to(weight.device), False)
    module.register_buffer('dspark_vocab_fp8_scale', torch.from_numpy(channel.T.copy()).to(weight.device), False)
    module.dspark_vocab_fp8_source_weight = weight
    module.dspark_vocab_fp8_preparation = dict(
        format='finite Gaudi2 e4m3 bias7, channel power-of-two scales',
        source_dtype='bfloat16', shape=list(weight.shape), maximum_absolute_weight_error=maximum_abs,
        relative_weight_l2=(error_energy / reference_energy) ** .5 if reference_energy else 0.,
        changed_weights=changed, additional_device_bytes=packed.nbytes + channel.nbytes,
        full_model_quality_qualified=False)
