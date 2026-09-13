// SPDX-License-Identifier: Apache-2.0
// Decode selected SWA/CSA2 rows once, preserving selection order and duplicates.
static inline float64 e4m3fn(uint64 code) {
    const uint64 exponent = (code >> 3) & 15;
    const uint64 mantissa = code & 7;
    const uint64 normal_bits = ((exponent + 120) << 23) | (mantissa << 20);
    const float64 tiny = convert_uint64_to_float64(mantissa, 0) * 0.001953125f;
    float64 value = v_f32_sel_eq_u32_b(exponent, 0, tiny, as_float64(normal_bits));
    const uint64 sign = (code & 128) << 24;
    value = as_float64(as_uint64(value) | sign);
    const uint64 nan_bits = 0x7fffffff;
    value = v_f32_sel_eq_u32_b(code & 127, 127, as_float64(nan_bits), value);
    return v_f32_sel_eq_f32_b(value, 0.0f, 0.0f, value);
}

static inline float64 ue8m0(uint64 code) {
    uint64 bits = code << 23;
    bits = v_u32_sel_eq_u32_b(code, 0, 0x00400000, bits);
    bits = v_u32_sel_eq_u32_b(code, 255, 0x7fffffff, bits);
    return as_float64(bits);
}

void main(tensor swa, tensor main_cache, tensor indices,
#ifdef DSV41_KV_WRITE_DEPENDENCY
          tensor completion,
#endif
#ifdef DSV41_COMPRESS_WRITE_DEPENDENCY
          tensor compressed_completion,
#endif
          tensor rows, tensor local_indices) {
#ifdef DSV41_KV_WRITE_DEPENDENCY
    const bool swa_ready = s_i32_ld_g(gen_addr((int5){0}, completion)) >= 0;
#else
    const bool swa_ready = 1;
#endif
#ifdef DSV41_COMPRESS_WRITE_DEPENDENCY
    const bool ready = swa_ready && s_i32_ld_g(gen_addr((int5){0}, compressed_completion)) >= 0;
#else
    const bool ready = swa_ready;
#endif
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int swa_length = get_dim_size(swa, 1);
    const int main_length = get_dim_size(main_cache, 0) == 288 ? get_dim_size(main_cache, 1) : 0;
    const uint64 lanes = V_LANE_ID_32;
    uint256 wide_directions = {0};
    wide_directions.v1 = (lanes >> 1) | 0x80;
    const uchar256 directions = convert_uint256_to_uchar256(wide_directions, SW_LINEAR);
    for (int slot = begin[0]; slot < end[0]; ++slot) {
        const int index = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, indices));
        const bool valid = ready && index >= 0 && index < swa_length + main_length;
        s_i32_st_g(gen_addr((int5){slot, 0, 0, 0, 0}, local_indices), valid ? slot : -1);
#ifdef DSV41_SELECTED_VALID_ONLY
        // The only consumer rejects -1 before loading a row. Invalid rows are
        // internal unspecified storage; the public gather still writes zeros.
        if (!valid) continue;
#endif
        for (int chunk = 0; chunk < 8; ++chunk) {
            float64 value = 0;
            if (valid && index < swa_length) {
                const uchar256 bytes = v_u8_ld_tnsr_partial_b((int5){chunk * 64, index, 0, 0, 0}, swa, 63, 0);
                const uint64 code = convert_uchar256_to_uint256(bytes, SW_LINEAR).v1;
                const unsigned s0 = s_u8_ld_g(gen_addr((int5){512 + chunk * 2, index, 0, 0, 0}, swa));
                const unsigned s1 = s_u8_ld_g(gen_addr((int5){513 + chunk * 2, index, 0, 0, 0}, swa));
                const uint64 scales = v_u32_sel_less_u32_b(lanes, 32, s0, s1);
                value = e4m3fn(code) * ue8m0(scales);
            } else if (valid) {
                const int row = index - swa_length;
                const uchar256 bytes = v_u8_ld_tnsr_partial_b((int5){chunk * 32, row, 0, 0, 0}, main_cache, 31, 0);
                const uchar256 expanded = v_u8_shuffle_b(bytes, directions, 0, (uchar256){0});
                const uint64 code = (convert_uchar256_to_uint256(expanded, SW_LINEAR).v1 >> ((lanes & 1) * 4)) & 15;
                const uint64 magnitude = code & 7;
                const float64 number = convert_uint64_to_float64(magnitude, 0);
                value = v_f32_sel_less_u32_b(magnitude, 6, number - 2.0f, number * 2.0f - 8.0f);
                value = v_f32_sel_less_u32_b(magnitude, 4, number * 0.5f, value);
                value = as_float64(as_uint64(value) | ((code & 8) << 28));
                const unsigned s0 = s_u8_ld_g(gen_addr((int5){256 + chunk * 4, row, 0, 0, 0}, main_cache));
                const unsigned s1 = s_u8_ld_g(gen_addr((int5){257 + chunk * 4, row, 0, 0, 0}, main_cache));
                const unsigned s2 = s_u8_ld_g(gen_addr((int5){258 + chunk * 4, row, 0, 0, 0}, main_cache));
                const unsigned s3 = s_u8_ld_g(gen_addr((int5){259 + chunk * 4, row, 0, 0, 0}, main_cache));
                uint64 scales = v_u32_sel_less_u32_b(lanes, 16, s0, s1);
                scales = v_u32_sel_geq_u32_b(lanes, 32, s2, scales);
                scales = v_u32_sel_geq_u32_b(lanes, 48, s3, scales);
                value *= e4m3fn(scales);
                value = v_f32_sel_eq_f32_b(value, 0.0f, 0.0f, value);
            }
            float128 wide = {0};
            wide.v1 = value;
            const bfloat128 output = convert_float128_to_bfloat128(wide, SW_RHNE | SW_LINEAR);
            v_bf16_st_tnsr_partial((int5){chunk * 64, slot, 0, 0, 0}, rows, output, 63, 0);
        }
    }
}
