// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_group32_roundtrip.h"

void main(tensor input, tensor output) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int width = get_dim_size(input, 0);
    for (int row = begin[1]; row < end[1]; ++row) {
        for (int tile = begin[0]; tile < end[0]; ++tile) {
            const int offset = tile * 128;
            const int last = s_i32_min(width - offset, 128) - 1;
            const int5 at = {offset, row, 0, 0, 0};
            const bfloat128 packed = v_bf16_ld_tnsr_partial_b(at, input, last, 0);
            float128 values = convert_bfloat128_to_float128(packed, SW_LINEAR);
            values.v1 = v41_group32_roundtrip(values.v1);
            values.v2 = v41_group32_roundtrip(values.v2);
            const bfloat128 result = convert_float128_to_bfloat128(values, SW_RHNE | SW_LINEAR);
            v_bf16_st_tnsr_partial(at, output, result, last, 0);
        }
    }
}
