# SPDX-License-Identifier: Apache-2.0
"""CPU codec checks need no HPU initialization or model weights."""

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_woa_fp8 import (
    covering_scale,
    decode_e4m3fn,
    decode_gaudi2,
    encode_gaudi2,
    prepare_rows,
)


def test_all_finite_normal_gaudi_encodings():
    codes = np.array([x for x in range(256) if 0 < (x & 0x78) < 0x78] + [0], np.uint8)
    np.testing.assert_array_equal(encode_gaudi2(decode_gaudi2(codes)), codes)
    with pytest.raises(ValueError):
        decode_gaudi2(np.array([0x78], np.uint8))
    assert decode_e4m3fn(np.array([0x7e], np.uint8))[0] == 448


def test_rne_ftz_and_zero():
    x = np.array([0., -0., 1.0625, 1.1875, 232., 248., 2**-7], np.float32)
    with pytest.raises(ValueError):
        encode_gaudi2(x)
    q = encode_gaudi2(x[[0, 1, 2, 3, 4, 6]])
    np.testing.assert_array_equal(decode_gaudi2(q), [0., 0., 1., 1.25, 224., 0.])
    assert q[0] == q[1] == 0


def test_covering_thresholds():
    for exponent in range(-30, 31):
        threshold = np.float32(240 * 2.**exponent)
        values = np.array([
            np.nextafter(threshold, -np.inf, dtype=np.float32), threshold,
            np.nextafter(threshold, np.inf, dtype=np.float32)
        ], np.float32)
        np.testing.assert_array_equal(covering_scale(values), [2.**exponent, 2.**exponent, 2.**(exponent + 1)])
    assert covering_scale(np.array([0], np.float32))[0] == 1


def test_block_scale_is_consumed_before_channel_scale():
    codes = np.full((32, 4096), 0x7e, np.uint8)
    scales = np.tile(np.array([120, 121, 122, 123], np.uint8), (1, 32))
    q, s, record = prepare_rows(codes, scales)
    expected = np.repeat(np.ldexp(np.full((1, 128), 448., np.float32), scales.astype(np.int32) - 127), 32, 1)
    np.testing.assert_array_equal(decode_gaudi2(q) * s, np.repeat(expected, 32, 0))
    assert record["error_energy"] == 0
    assert record["temporary_upper_bound_bytes"] < 2 * 2**30


def test_source_specials_rejected():
    with pytest.raises(ValueError):
        decode_e4m3fn(np.array([0x7f, 0xff], np.uint8))
    with pytest.raises(ValueError):
        prepare_rows(np.zeros((32, 4096), np.uint8), np.full((1, 128), 255, np.uint8))
