// SPDX-License-Identifier: Apache-2.0
// Reuses FlashInfer-Gaudi's row statistics and FP8 packing, with V4.1
// FP32 clamp/SwiGLU/routing and explicit BF16 projection/activation boundaries.
static inline float64 round_bf16(float64 value) {
    float64_pair_t pair;
    pair.v1 = value;
    pair.v2 = value;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pair)).v1;
}

static inline float64 reciprocal_without_lookup(float64 value) {
    // Positive normal scales are bounded below by epsilon/240 and above
    // by BF16_MAX/240. Three Newton steps preserve the BF16-rounded result.
    float64 estimate = as_float64((int64)0x7ef311c3 - as_int64(value));
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    return v_f32_sel_eq_f32_b(value, as_float64((int64)0x7f800000), 0.0f, estimate);
}

static inline float64 row_max_without_lookup(float64 value) {
    // Fold the four dual groups and their two groups. Unlike the generic
    // reduction helper, explicit lane broadcasts do not reserve the LUT cache.
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(value, 0xffffffff, 1, 0, 3, 2,
                                                      MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(value, 0xffffffff, 2, 3, 0, 1,
                                                      MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_group_b(value, 0xffffffff, 63, 0));
    float64 result = 0;
    #pragma loop_unroll(8)
    for (int lane = 0; lane < 8; ++lane) {
        const uchar256 broadcast = 0x80 | lane;
        result = v_f32_max_b(result, v_f32_shuffle_b(value, broadcast, 0, value));
    }
    return result;
}

static inline float64 projected(tensor product, tensor channel, int n, int row,
                                int expert, float sx, bool valid) {
    const float64 acc = v_f32_ld_tnsr_b((int5){n, 0, row}, product);
    const uint64 bits = v_u32_ld_tnsr_b((int5){n % 256, n / 256, expert}, channel,
        SW_UNPACK | SW_UNPCK_16_TO_32, (uint64){0}, valid) << 16;
    return round_bf16(v_f32_mul_b(v_f32_mul_b(acc, *((float64*)&bits)), sx));
}

static inline float64 activated_half(tensor product, tensor channel, int n, int row,
                                     int expert, int width, float sx, float route, bool valid) {
    float64 gate = projected(product, channel, n, row, expert, sx, valid);
    float64 up = projected(product, channel, n + width, row, expert, sx, valid);
    gate = v_f32_min_b(gate, 10.0f);
    up = v_f32_max_b(v_f32_min_b(up, 10.0f), -10.0f);
    const float64 silu = v_f32_mul_b(gate, v_sigmoid_f32(gate));
    return v_f32_mul_b(v_f32_mul_b(silu, up), route);
}

void main(tensor product, tensor ids, tensor activation_scale, tensor channel, tensor router,
          tensor output, tensor scales) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(output, 0);
    const int experts = get_dim_size(channel, 2);
    const float sx = s_f32_ld_g(gen_addr((int5){0}, activation_scale));
    bfloat128 activated[20];
    for (int row = start[0]; row < end[0]; ++row) {
        const int expert = s_i32_ld_g(gen_addr((int5){row}, ids));
        const bool valid = expert >= 0 && expert < experts;
        const float route = s_f32_ld_g(gen_addr((int5){row}, router));
        float64 maximum = 0;
        #pragma loop_unroll(2)
        for (int tile = 0; tile < width / 128; ++tile) {
            float64_pair_t pair;
            pair.v1 = activated_half(product, channel, tile * 128, row, expert, width, sx, route, valid);
            pair.v2 = activated_half(product, channel, tile * 128 + 64, row, expert, width, sx, route, valid);
            // Projection loads are two contiguous 64-element halves, unlike
            // the interleaved pair produced by convert_bf16_to_f32_all.
            const bfloat128 value = convert_float128_to_bfloat128(pair, SW_RHNE | SW_LINEAR);
            activated[tile] = value;
            const float64_pair_t rounded = v_convert_bf16_to_f32_all_b(value);
            maximum = v_f32_max_b(maximum, v_f32_abs_b(rounded.v1));
            maximum = v_f32_max_b(maximum, v_f32_abs_b(rounded.v2));
        }
        maximum = row_max_without_lookup(maximum);
        const float64 raw_scale = round_bf16(maximum * (float)(bf16)(1.0f / 240.0f));
        const float64 scale = round_bf16(raw_scale + (float)(bf16)(1.0e-8f / 240.0f));
        const float64 inverse = round_bf16(reciprocal_without_lookup(scale));
        v_f32_st_tnsr((int5){0, 0, row}, scales, scale);
        #pragma loop_unroll(2)
        for (int tile = 0; tile < width / 128; ++tile) {
            const float64_pair_t fp32 = v_convert_bf16_to_f32_all_b(activated[tile]);
            minifloat256 packed = 0;
            packed = v_convert_f32_to_f8_b(round_bf16(fp32.v1 * inverse), 0, SW_CLIP_FP, packed);
            packed = v_convert_f32_to_f8_b(round_bf16(fp32.v2 * inverse), 2, SW_CLIP_FP, packed);
            const minifloat256 sparse = packed;
            packed = v_f8_pack_b(sparse, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            packed = v_f8_pack_b(sparse, SW_GROUP_1 | SW_STRIDE_2, packed);
            packed = v_f8_mov_dual_group_pack_b(packed, SW_PACK21, (minifloat256)0);
            v_f8_st_tnsr_partial((int5){tile * 128, 0, row}, output, packed, 127, 0);
        }
    }
}
