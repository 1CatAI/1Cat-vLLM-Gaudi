// SPDX-License-Identifier: Apache-2.0
// Decode each selected row once, using a full BF16 vector and row scale reuse.
static inline bfloat128 selected_e4m3fn(ushort128 code) {
    const ushort128 exponent = (code >> 3) & 15;
    const ushort128 mantissa = code & 7;
    ushort128 tiny = (mantissa << 5) + 0x3b80;
    tiny = v_u16_sel_less_u16_b(mantissa, 4, (mantissa << 6) + 0x3b00, tiny);
    tiny = v_u16_sel_eq_u16_b(mantissa, 1, 0x3b00, tiny);
    tiny = v_u16_sel_eq_u16_b(mantissa, 0, 0, tiny);
    ushort128 bits = ((exponent + 120) << 7) | (mantissa << 4);
    bits = v_u16_sel_eq_u16_b(exponent, 0, tiny, bits);
    bits |= (code & 128) << 8;
    bits = v_u16_sel_eq_u16_b(code & 127, 0, 0, bits);
    bits = v_u16_sel_eq_u16_b(code & 127, 127, 0x7fff, bits);
    return *((bfloat128*)&bits);
}

static inline bfloat128 selected_ue8m0(ushort128 code) {
    ushort128 bits = code << 7;
    // The reference F32 multiply flushes the UE8M0 zero-code subnormal input.
    bits = v_u16_sel_eq_u16_b(code, 0, 0, bits);
    bits = v_u16_sel_eq_u16_b(code, 255, 0x7fff, bits);
    return *((bfloat128*)&bits);
}

static inline bfloat128 selected_fp4(ushort128 code) {
    const ushort128 magnitude = code & 7;
    ushort128 bits = (magnitude << 6) + 0x3f00;
    bits = v_u16_sel_eq_u16_b(magnitude, 1, 0x3f00, bits);
    bits = v_u16_sel_eq_u16_b(magnitude, 0, 0, bits);
    bits |= (code & 8) << 12;
    return *((bfloat128*)&bits);
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
    const int slots = get_dim_size(indices, 0);
    const int swa_length = get_dim_size(swa, 1);
    const int main_length = get_dim_size(main_cache, 0) == 288 ? get_dim_size(main_cache, 1) : 0;
    const uchar256 lanes = V_LANE_ID_8;
    const uchar256 fp4_directions = (lanes >> 1) | 0x80;
    const ushort128 nibble_shifts = ((ushort128)V_LANE_ID_16 & 1) << 2;
    for (int point = begin[0]; point < end[0]; ++point) {
        for (int slot = point; slot < slots; slot += 128) {
            const int index = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, indices));
            const bool valid = ready && index >= 0 && index < swa_length + main_length;
            s_i32_st_g(gen_addr((int5){slot, 0, 0, 0, 0}, local_indices), valid ? slot : -1);
#ifdef DSV41_SELECTED_VALID_ONLY
            if (!valid) continue;
#endif
            if (!valid) {
                for (int chunk = 0; chunk < 4; ++chunk)
                    v_bf16_st_tnsr((int5){chunk * 128, slot, 0, 0, 0}, rows, (bfloat128){0});
            } else if (index < swa_length) {
                const uchar256 raw_scales = v_u8_ld_tnsr_partial_b((int5){512, index, 0, 0, 0}, swa, 15, 0);
                const uchar256 row_scales = v_u8_mov_dual_group_all_b(
                    raw_scales, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                for (int chunk = 0; chunk < 4; ++chunk) {
                    const uchar256 bytes = v_u8_ld_tnsr_partial_b((int5){chunk * 128, index, 0, 0, 0}, swa, 127, 0);
                    const ushort128 code = convert_uchar256_to_ushort256(bytes, SW_LINEAR).v1;
                    const uchar256 directions = ((lanes >> 5) + chunk * 4) | 0x80;
                    const uchar256 scale_bytes = v_u8_shuffle_b(row_scales, directions, 0, (uchar256){0});
                    const ushort128 scales = convert_uchar256_to_ushort256(scale_bytes, SW_LINEAR).v1;
                    const bfloat128 value = v_bf16_mul_b(selected_e4m3fn(code), selected_ue8m0(scales));
                    v_bf16_st_tnsr((int5){chunk * 128, slot, 0, 0, 0}, rows, value);
                }
            } else {
                const int row = index - swa_length;
                const uchar256 raw_scales = v_u8_ld_tnsr_partial_b((int5){256, row, 0, 0, 0}, main_cache, 31, 0);
                const uchar256 row_scales = v_u8_mov_dual_group_all_b(
                    raw_scales, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                for (int chunk = 0; chunk < 4; ++chunk) {
                    const uchar256 raw_bytes = v_u8_ld_tnsr_partial_b((int5){chunk * 64, row, 0, 0, 0}, main_cache, 63, 0);
                    const uchar256 bytes = v_u8_mov_dual_group_all_b(
                        raw_bytes, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), (uchar256){0});
                    const uchar256 expanded = v_u8_shuffle_b(bytes, fp4_directions, 0, (uchar256){0});
                    const ushort128 code = (convert_uchar256_to_ushort256(expanded, SW_LINEAR).v1 >> nibble_shifts) & 15;
                    const uchar256 directions = ((lanes >> 4) + chunk * 8) | 0x80;
                    const uchar256 scale_bytes = v_u8_shuffle_b(row_scales, directions, 0, (uchar256){0});
                    const ushort128 scales = convert_uchar256_to_ushort256(scale_bytes, SW_LINEAR).v1;
                    bfloat128 value = v_bf16_mul_b(selected_fp4(code), selected_e4m3fn(scales));
                    value = v_bf16_sel_eq_bf16_b(value, (bfloat)0, (bfloat)0, value);
                    v_bf16_st_tnsr((int5){chunk * 128, slot, 0, 0, 0}, rows, value);
                }
            }
        }
    }
}
