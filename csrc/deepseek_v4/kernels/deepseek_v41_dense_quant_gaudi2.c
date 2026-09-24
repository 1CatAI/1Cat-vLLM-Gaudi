// SPDX-License-Identifier: Apache-2.0
// A complete K row shares one power-of-two scale. Reload the small activation
// instead of retaining dynamically indexed vector arrays in local memory.
void main(tensor input, tensor output, tensor scales) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int tiles = get_dim_size(input, 0) / 128;
    for (int row = start[0]; row < end[0]; ++row) {
        float64 maximum = 0;
        #pragma loop_unroll(4) pipelined taken
        for (int tile = 0; tile < tiles; ++tile) {
            const int5 at = {tile * 128, row, 0, 0, 0};
            const float128 wide = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(at, input));
            maximum = v_f32_max_b(maximum, v_f32_abs_b(wide.v1));
            maximum = v_f32_max_b(maximum, v_f32_abs_b(wide.v2));
        }
        maximum = v_f32_reduce_max(maximum);
        const uint64 bits = as_uint64(maximum);
        int64 power = convert_uint64_to_int64(bits >> 23, 0) - 134;
        power += v_i32_sel_grt_u32_b(bits & 0x7fffff, 0x700000, 1, 0);
        power = v_i32_sel_eq_f32_b(maximum, 0.0f, 0, power);
        const float64 scale = as_float64((power + 127) << 23);
        const float64 inverse = as_float64((127 - power) << 23);
        // A power-of-two reciprocal is exactly representable in BF16. Its
        // product has the same FP8 rounding in the representable FP8 range.
        const float128 inverse_pair = {inverse, inverse};
        const bfloat128 inverse_bf16 = v_convert_f32_to_bf16_all_b(inverse_pair);
        const int5 scale_at = {0, row, 0, 0, 0};
        v_f32_st_tnsr_partial(scale_at, scales, scale, 0, 0);
        #pragma loop_unroll(4) pipelined taken
        for (int tile = 0; tile < tiles; ++tile) {
            const int5 at = {tile * 128, row, 0, 0, 0};
            const bfloat128 value = v_bf16_ld_tnsr_b(at, input) * inverse_bf16;
            minifloat256 q = v_convert_bf16_to_f8_b(value, 0, SW_RHNE | SW_CLIP_FP, (minifloat256)0);
            const minifloat256 sparse = q;
            q = v_f8_pack_b(sparse, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            q = v_f8_pack_b(sparse, SW_GROUP_1 | SW_STRIDE_2, q);
            q = v_f8_mov_dual_group_pack_b(q, SW_PACK21, (minifloat256)0);
            uchar256 raw = *((uchar256*)&q);
            raw = v_u8_sel_eq_u8_b(raw & 0x78, 0, 0, raw);
            q = *((minifloat256*)&raw);
            v_f8_st_tnsr_partial(at, output, q, 127, 0);
        }
    }
}
