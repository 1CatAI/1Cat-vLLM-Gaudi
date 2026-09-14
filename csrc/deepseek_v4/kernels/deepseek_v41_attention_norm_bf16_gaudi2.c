// SPDX-License-Identifier: Apache-2.0
// Reuse FlashInfer-Gaudi row reduction/rsqrt, preserving V4.1's FP32 weight
// product and final BF16 boundary rather than the vendor BF16 product.
#pragma clang fp contract(off)
#define FLASHINFER_NORM_USE_LOOKUP_RSQRT
#include "../../flashinfer_gaudi/kernels/norm_math_gaudi2.h"
void main(tensor input, tensor weight, tensor output, float epsilon, float inverse_width) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int tiles = get_dim_size(input, 0) / 128;
    bfloat128 cached[10];
    for (int row = begin[0]; row < end[0]; ++row) {
        float128 squares = {0};
        for (int tile = 0; tile < tiles; ++tile) {
            const int5 at = {tile * 128, row, 0, 0, 0};
            const bfloat128 value = v_bf16_ld_tnsr_b(at, input);
            cached[tile] = value;
            squares = v_bf16_mac_acc32_b(value, value, squares, (e_no_negation) << 1);
        }
        const float64 rrms = positive_rsqrt(row_sum(squares.v1 + squares.v2) * inverse_width + epsilon);
        for (int tile = 0; tile < tiles; ++tile) {
            const int5 at = {tile * 128, row, 0, 0, 0};
            const int5 wt = {tile * 128, 0, 0, 0, 0};
            const float128 w = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wt, weight));
            float128 value = v_convert_bf16_to_f32_all_b(cached[tile]);
            value.v1 = (value.v1 * rrms) * w.v1;
            value.v2 = (value.v2 * rrms) * w.v2;
            v_bf16_st_tnsr(at, output, v_convert_f32_to_bf16_all_b(value));
        }
    }
}
