// SPDX-License-Identifier: Apache-2.0
// Consume a complete head. Preserve the BF16 boundary before adjacent-pair RoPE.
#pragma clang fp contract(off)
#define DSV4_QNORM_HELPERS_ONLY 1
#define DSV4_ROPE_SECOND_TERM_FMA 1
#include "deepseek_v4_qnorm_rope_kv_pack_bf16.h"
void main(tensor product, tensor weight_scale, tensor activation_scale,
          tensor positions, tensor table, tensor output) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const float sx = s_f32_ld_g(gen_addr((int5){0}, activation_scale));
    const int position = s_i32_ld_g(gen_addr((int5){0}, positions));
    for (int head = start[0]; head < end[0]; ++head) {
        for (int block = 0; block < 3; ++block) {
            int5 at = {head * 512 + block * 128, 0, 0, 0, 0};
            float128 scaled;
            scaled.v1 = v_f32_ld_tnsr_b(at, product) * v_f32_ld_tnsr_b(at, weight_scale) * sx;
            at[0] += 64;
            scaled.v2 = v_f32_ld_tnsr_b(at, product) * v_f32_ld_tnsr_b(at, weight_scale) * sx;
            at[0] -= 64;
            v_bf16_st_tnsr(at, output, convert_float128_to_bfloat128(scaled, SW_RHNE | SW_LINEAR));
        }
        int5 at = {head * 512 + 384, 0, 0, 0, 0};
        float128 scaled = {0};
        scaled.v1 = v_f32_ld_tnsr_b(at, product) * v_f32_ld_tnsr_b(at, weight_scale) * sx;
        v_bf16_st_tnsr_partial(at, output, convert_float128_to_bfloat128(scaled, SW_RHNE | SW_LINEAR), 63, 0);
        at[0] += 64;
        scaled.v1 = v_f32_ld_tnsr_b(at, product) * v_f32_ld_tnsr_b(at, weight_scale) * sx;
        const bfloat128 rounded = convert_float128_to_bfloat128(scaled, SW_RHNE | SW_LINEAR);
        const float64 expanded = convert_bfloat128_to_float128(rounded, SW_LINEAR).v1;
        scaled.v1 = dsv4_qkv_apply_pairwise_rope_f32(expanded, table, position);
        v_bf16_st_tnsr_partial(at, output, convert_float128_to_bfloat128(scaled, SW_RHNE | SW_LINEAR), 63, 0);
    }
}
