// SPDX-License-Identifier: Apache-2.0
// Four independent group-32 codecs share a BF16 vector. Positive BF16 bit
// patterns preserve magnitude order, with NaNs sorting above infinity.
static inline bfloat128 v41_group32_roundtrip_bf16(bfloat128 value) {
    const ushort128 original = *((ushort128*)&value);
    const ushort128 magnitude = original & 0x7fff;
    ushort128 maximum = v_u16_max_b(magnitude, v_u16_mov_group_b(magnitude, 0xffffffff, 63, 0));
    const uchar256 lane = read_lane_id_1b_b();
    #pragma loop_unroll(4)
    for (int offset = 2; offset <= 16; offset *= 2) {
        const uchar256 selector = ((lane ^ offset) & 31) | 0x80;
        const uchar256 shuffled = v_u8_shuffle_b(*((uchar256*)&maximum), selector, 0, (uchar256)0);
        maximum = v_u16_max_b(maximum, *((ushort128*)&shuffled));
    }
    // The BF16 value below 1e-4 selects the same scale exponent as the FP32
    // floor. A group containing NaN retains the original minimum-scale rule.
    maximum = v_u16_max_b(maximum, 0x38d1);
    maximum = v_u16_sel_grt_u16_b(maximum, 0x7f80, 0x38d1, maximum);
    const short128 exponent = convert_ushort128_to_short128(maximum >> 7, 0) - 135
        + v_i16_sel_grt_u16_b(maximum & 127, 96, 1, 0);
    const short128 input_exponent = convert_ushort128_to_short128(magnitude >> 7, 0);
    // Subnormal FP8 codes are integers on the scale/512 grid. Clamp the
    // shift for lanes outside this branch and for values rounding to zero.
    const short128 shift = v_i16_max_b(v_i16_min_b(exponent + 125 - input_exponent, 9), 1);
    const ushort128 significand = (magnitude & 127) | 128;
    const ushort128 code = (significand + ((ushort128)1 << (shift - 1)) - 1
        + ((significand >> shift) & 1)) >> shift;
    const bfloat128 small_code = convert_ushort128_to_bfloat128(code, SW_RHNE);
    const ushort128 grid_bits = (ushort128)((exponent + 118) << 7);
    const bfloat128 tiny = small_code * *((bfloat128*)&grid_bits);
    // Keep three mantissa bits with ties to even, then saturate at 448*scale.
    const ushort128 rounded = (magnitude + 7 + ((magnitude >> 4) & 1)) & 0xfff0;
    ushort128 result = v_u16_sel_less_i16_b(input_exponent, exponent + 121,
                                            *((ushort128*)&tiny), rounded);
    const ushort128 cap = v_u16_min_b((ushort128)(((exponent + 135) << 7) + 96), 0x7f80);
    result = v_u16_min_b(result, cap) | (original & 0x8000);
    result = v_u16_sel_eq_u16_b(result & 0x7fff, 0, 0, result);
    result = v_u16_sel_grt_u16_b(magnitude, 0x7f80, 0x7fff, result);
    return *((bfloat128*)&result);
}
