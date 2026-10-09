# SPDX-License-Identifier: Apache-2.0
"""Two normal-FP8 channel planes from the original FP32 controller."""
import numpy as np

from vllm_gaudi.ops.deepseek_v41_woa_fp8 import covering_scale, decode_gaudi2, encode_gaudi2


def prepare_weight(source):
    source = np.asarray(source, dtype=np.float32)
    if source.shape != (24, 20480) or not np.isfinite(source).all():
        raise ValueError("Paired control preparation requires finite checkpoint FP32 [24,20480]")
    high_scale = covering_scale(np.max(np.abs(source), axis=1, keepdims=True))
    high = encode_gaudi2(source / high_scale)
    residual = source - decode_gaudi2(high) * high_scale
    low_scale = covering_scale(np.max(np.abs(residual), axis=1, keepdims=True))
    low = encode_gaudi2(residual / low_scale)
    return np.concatenate((high, low)), np.concatenate((high_scale, low_scale)).T.copy()
