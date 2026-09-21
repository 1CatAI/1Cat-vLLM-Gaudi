// SPDX-License-Identifier: Apache-2.0
// Four independent checkpoint group-32 codecs share one BF16 vector load.
// The linear conversion preserves contiguous groups in the two FP32 vectors.
static inline float64 v41_group32_max(float64 value) {
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(
        value, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_group_b(value, 0xffffffff, 63, 0));
    float64 maximum = 0;
    #pragma loop_unroll(8)
    for (int lane = 0; lane < 8; ++lane)
        maximum = v_f32_max_b(maximum,
            v_f32_shuffle_b(value, (uchar256)(0x80 | lane), 0, value));
    return maximum;
}

static inline float64 v41_group32_roundtrip(float64 value) {
    const float64 absolute = v_f32_abs_b(value);
    const uint64 bits = as_uint64(absolute);
    const float64 any_nan = v41_group32_max(v_f32_sel_grt_u32_b(bits, 0x7f800000, 1.0f, 0.0f));
    float64 maximum = v_f32_max_b(v41_group32_max(absolute), 1.0e-4f);
    maximum = v_f32_sel_grt_f32_b(any_nan, 0.0f, 1.0e-4f, maximum);
    const uint64 maximum_bits = as_uint64(maximum);
    const int64 exponent = convert_uint64_to_int64(maximum_bits >> 23, 0) - 135;
    const int64 adjustment = v_i32_sel_grt_u32_b(maximum_bits & 0x7fffff, 0x600000, 1, 0);
    const int64 scale_exponent = exponent + adjustment;
    const float64 scale = as_float64((scale_exponent + 127) << 23);
    const float64 reciprocal = as_float64((127 - scale_exponent) << 23);
    const float64 normalized = absolute * reciprocal;
    const uint64 rounded = (bits + 0x7ffff + ((bits >> 20) & 1)) & 0xfff00000;
    const int64 tiny_code = v_convert_f32_to_i32_b(normalized * 512.0f, 0, SW_RHNE);
    const float64 tiny = convert_int64_to_float64(tiny_code, 0) * (scale * 0.001953125f);
    float64 result = v_f32_sel_less_f32_b(normalized, 0.015625f, tiny, as_float64(rounded));
    result = v_f32_min_b(result, scale * 448.0f);
    result = as_float64(as_uint64(result) | (as_uint64(value) & 0x80000000));
    result = v_f32_sel_eq_f32_b(result, 0.0f, 0.0f, result);
    return v_f32_sel_grt_u32_b(bits, 0x7f800000, as_float64((uint64)0x7fffffff), result);
}
