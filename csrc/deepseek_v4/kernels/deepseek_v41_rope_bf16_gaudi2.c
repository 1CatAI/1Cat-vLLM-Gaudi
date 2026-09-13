// SPDX-License-Identifier: Apache-2.0
// Reuse V4 adjacent-pair RoPE with its prepared [cos32, sin32] table.
#pragma clang fp contract(off)
#define DSV4_QNORM_HELPERS_ONLY 1
#define DSV4_ROPE_SECOND_TERM_FMA 1
#include "deepseek_v4_qnorm_rope_kv_pack_bf16.h"
void main(tensor value, tensor positions, tensor table, tensor output) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int width = get_dim_size(value, 0);
    const int position = s_i32_ld_g(gen_addr((int5){0}, positions));
    for (int head = begin[0]; head < end[0]; ++head) {
        for (int offset = 0; offset < width - 128; offset += 128) {
            const int5 coords = {offset, head, 0, 0, 0};
            v_bf16_st_tnsr(coords, output, v_bf16_ld_tnsr_b(coords, value));
        }
        int5 coords = {width - 128, head, 0, 0, 0};
        const bfloat128 prefix = v_bf16_ld_tnsr_partial_b(coords, value, 63, 0);
        v_bf16_st_tnsr_partial(coords, output, prefix, 63, 0);
        coords[0] = width - 64;
        const bfloat128 tail = v_bf16_ld_tnsr_partial_b(coords, value, 63, 0);
        const float64 expanded = convert_bfloat128_to_float128(tail, SW_LINEAR).v1;
        float128 rotated = {0};
        rotated.v1 = dsv4_qkv_apply_pairwise_rope_f32(expanded, table, position);
        v_bf16_st_tnsr_partial(coords, output, convert_float128_to_bfloat128(rotated, SW_LINEAR), 63, 0);
    }
}
