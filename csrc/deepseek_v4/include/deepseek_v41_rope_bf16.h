// SPDX-License-Identifier: Apache-2.0
// Reuse V4's exact adjacent-pair shuffle convention with the V4.1
// concatenated [cos32, sin32] table. Prefix values retain their original bits.
#pragma clang fp contract(off)
#ifdef DSV41_ROPE_INVERSE
#define DSV4_ROPE_INVERSE 1
#endif
#define DSV4_QNORM_HELPERS_ONLY 1
#ifndef DSV41_ROPE_SEPARATE_PRODUCTS
#define DSV4_ROPE_SECOND_TERM_FMA 1
#endif
#include "deepseek_v4_qnorm_rope_kv_pack_bf16.h"

void main(tensor value, tensor positions, tensor phase, tensor output) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int width = get_dim_size(value, 0);
    for (int token = begin[2]; token < end[2]; ++token) {
        const int position = s_i32_ld_g(gen_addr((int5){token,0,0,0,0}, positions));
        for (int head = begin[1]; head < end[1]; ++head) {
            for (int offset = 0; offset < width - 128; offset += 128) {
                const int5 at = {offset,head,token,0,0};
                v_bf16_st_tnsr(at, output, v_bf16_ld_tnsr_b(at, value));
            }
            int5 at = {width-128,head,token,0,0};
            const bfloat128 prefix = v_bf16_ld_tnsr_partial_b(at, value, 63, 0);
            v_bf16_st_tnsr_partial(at, output, prefix, 63, 0);
            at[0] = width - 64;
            const bfloat128 tail = v_bf16_ld_tnsr_partial_b(at, value, 63, 0);
            const float64 expanded = convert_bfloat128_to_float128(tail, SW_LINEAR).v1;
            float128 rotated = {0};
            rotated.v1 = dsv4_qkv_apply_pairwise_rope_f32(expanded, phase, position);
            v_bf16_st_tnsr_partial(
                at, output, convert_float128_to_bfloat128(rotated, SW_LINEAR), 63, 0);
        }
    }
}
