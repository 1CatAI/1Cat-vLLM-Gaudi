# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert
from vllm_gaudi.ops.deepseek_v41_prefill_moe import (
    _i16_little_endian_bytes,
    restore_n256_mxfp4_u8,
    restore_n256_scale_u8,
)
from vllm_gaudi.ops.deepseek_v41_weights import prepare_q16, prepare_s16


@pytest.mark.parametrize("n,k", ((512, 1152), (2304, 5120), (5120, 1152)))
def test_n256_resident_layout_restores_stock_prefill_bytes(n, k):
    generator = np.random.default_rng(n + k)
    packed = generator.integers(0, 256, (n, k // 2), dtype=np.uint8)
    # A fixed finite scale keeps this layout test within the N256 FP8
    # candidate's qualified row-range contract.
    scales = np.full((n, k // 32), 127, dtype=np.uint8)
    q16, shape = prepare_q16(packed)
    s16, _ = prepare_s16(scales, shape)
    n256, planes, _, _ = prepare_expert(q16, s16)
    actual_packed = restore_n256_mxfp4_u8(torch.from_numpy(n256).unsqueeze(0))[0].numpy()
    actual_scales = restore_n256_scale_u8(torch.from_numpy(planes).unsqueeze(0))[0].numpy()
    assert np.array_equal(actual_packed, packed)
    assert np.array_equal(actual_scales, scales)


def test_i16_byte_expansion_covers_every_encoding():
    words = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    actual = _i16_little_endian_bytes(words.reshape(1, -1))[0].numpy()
    expected = words.numpy().view(np.uint8)
    assert np.array_equal(actual, expected)
