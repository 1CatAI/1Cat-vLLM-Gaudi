// SPDX-License-Identifier: Apache-2.0
// Reuse V4's exact adjacent-pair shuffle convention with the V4.1
// interleaved [cos, sin] table. Prefix values retain their original bits.
#define DSV4_QNORM_HELPERS_ONLY
#include "deepseek_v4_qnorm_rope_kv_pack_bf16.h"

void main(tensor value, tensor positions, tensor phase, tensor output) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int width = get_dim_size(value, 0);
    const uint64 lanes = V_LANE_ID_32;
    const uint64 group = ((lanes >> 3) & 1) << 5;
    const uchar256 even = dsv4_qkv_shuffle_directions((lanes & 6) + group + 0x80);
    const uchar256 odd = dsv4_qkv_shuffle_directions((lanes & 6) + group + 0x81);
    for (int token = begin[2]; token < end[2]; ++token) {
        const int position = s_i32_ld_g(gen_addr((int5){token,0,0,0,0}, positions));
        const bool valid = position >= 0 && position < get_dim_size(phase, 1);
        const float64 trig = v_f32_ld_tnsr_b((int5){0,position,0,0,0}, phase, 0, 0.0f, valid);
        const float64 cosine = v_f32_shuffle_b(trig, even, 0, 0.0f);
        float64 sine = v_f32_shuffle_b(trig, odd, 0, 0.0f);
#ifdef DSV41_ROPE_INVERSE
        sine = -sine;
#endif
        for (int head = begin[1]; head < end[1]; ++head) {
            for (int offset = 0; offset < width - 128; offset += 128) {
                const int5 at = {offset,head,token,0,0};
                v_bf16_st_tnsr(at, output, v_bf16_ld_tnsr_b(at, value));
            }
            const int5 at = {width-128,head,token,0,0};
            const bfloat128 tail = v_bf16_ld_tnsr_b(at, value);
            float128 wide = convert_bfloat128_to_float128(tail, SW_LINEAR);
            const float64 real = v_f32_shuffle_b(wide.v2, even, 0, 0.0f);
            const float64 imag = v_f32_shuffle_b(wide.v2, odd, 0, 0.0f);
            // The production graph contracts the second product into the
            // add/subtract. Preserve its FP32 multiply then FMA rounding.
            const float64 rc = v_f32_mul_b(real, cosine);
            const float64 ic = v_f32_mul_b(imag, cosine);
            const float64 rotated_real = v_f32_mac_b(-imag, sine, rc);
            const float64 rotated_imag = v_f32_mac_b(real, sine, ic);
            wide.v2 = v_f32_sel_eq_u32_b(lanes & 1, 1, rotated_imag, rotated_real);
            const bfloat128 rotated = convert_float128_to_bfloat128(wide, SW_RHNE | SW_LINEAR);
            // Store only the rotated half through FP32. Even subnormal and
            // signed-zero values in the non-RoPE prefix are copied verbatim.
            const ushort128 bits = v_u16_sel_less_u16_b(
                (ushort128)V_LANE_ID_16, 64, as_ushort128(tail), as_ushort128(rotated));
            v_bf16_st_tnsr(at, output, as_bfloat128(bits));
        }
    }
}
