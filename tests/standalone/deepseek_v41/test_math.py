# SPDX-License-Identifier: Apache-2.0
"""Byte-format checks guard against Gaudi2/checkpoint FP8 range confusion."""

import torch

from vllm_gaudi.ops.deepseek_v41_math import (
    e4m3_decode, e4m3_encode, fp4_decode, fp4_encode, pack_fp4, pack_swa, unpack_fp4, unpack_swa,
)


def test_e4m3_all_codes_and_tie_rounding_match_checkpoint_format():
    codes = torch.arange(256, dtype=torch.uint8)
    reference = codes.view(torch.float8_e4m3fn).float()
    actual = e4m3_decode(codes)
    assert torch.equal(torch.isnan(actual), torch.isnan(reference))
    finite = ~torch.isnan(reference)
    assert torch.equal(actual[finite].view(torch.int32), reference[finite].view(torch.int32))
    assert torch.equal(e4m3_encode(reference)[finite], codes[finite])
    positive = reference[:127]
    midpoint = (positive[1:] + positive[:-1]) * .5
    values = torch.cat((midpoint, -midpoint, torch.tensor([-0., 0., 240., 448., -448., 1000.])))
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(e4m3_encode(values), expected)


def test_fp4_codes_and_ties_preserve_even_rounding_and_signed_zero():
    reference = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    codes = torch.arange(16, dtype=torch.uint8)
    assert torch.equal(fp4_decode(codes).view(torch.int32), reference.view(torch.int32))
    assert torch.equal(fp4_encode(reference), codes)
    values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8)
    assert torch.equal(fp4_encode(values), expected)
    assert torch.equal(fp4_encode(-values), expected | 8)


def test_packed_cache_shapes_and_known_values_include_rope_tail():
    value = torch.zeros(2, 512, dtype=torch.bfloat16)
    value[0] = 3
    value[1, -64:] = -3
    for packed, restored, expected_shape in (
        (pack_swa(value), unpack_swa, (2, 528)),
        (pack_fp4(value), unpack_fp4, (2, 288)),
    ):
        assert packed.shape == expected_shape and packed.dtype == torch.uint8
        assert torch.equal(restored(packed), value)
    index = value[:, :128]
    packed = pack_fp4(index, 32)
    assert packed.shape == (2, 68)
    assert torch.equal(unpack_fp4(packed, 128, 32), index)
