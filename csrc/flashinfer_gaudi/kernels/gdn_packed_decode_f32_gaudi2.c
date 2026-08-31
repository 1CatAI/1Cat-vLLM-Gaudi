/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

// Single-token Qwen GDN decode for Gaudi2. Each index-space point owns 32
// value rows for one value head and one request. Q/K remain grouped instead
// of being materialized once per value head.
void main(
    tensor state_cache,
    tensor packed_qkv,
    tensor decay,
    tensor beta,
    tensor state_indices,
    tensor new_state,
    tensor output)
{
    const uchar256 broadcast_lane_zero = 0x80;
    const int5 index_start = get_index_space_offset();
    const int5 index_end = get_index_space_size() + index_start;
    const int key_dim = 128;
    const int value_dim = 128;
    const int value_heads = get_dim_size(state_cache, 2);
    const int state_slots = get_dim_size(state_cache, 3);
    const int packed_width = get_dim_size(packed_qkv, 0);
    const int key_heads = (packed_width - value_heads * value_dim) / (2 * key_dim);
    const int head_repeat = value_heads / key_heads;
    const int key_offset = key_heads * key_dim;
    const int value_offset = 2 * key_offset;
    const float q_scale = 0.08838834764831845f;

    for (int batch = index_start[3]; batch < index_end[3]; ++batch) {
        int5 index_coords = {batch, 0, 0, 0, 0};
        const int state_slot = s_i32_ld_g(gen_addr(index_coords, state_indices));
        const bool valid_slot = state_slot >= 0 && state_slot < state_slots;

        for (int value_head = index_start[2]; value_head < index_end[2]; ++value_head) {
            const int value_start = index_start[1] * 32;
            const int value_end = index_end[1] * 32;
            if (!valid_slot) {
                for (int value_row = value_start; value_row < value_end; ++value_row) {
                    int5 output_coords = {value_row, value_head, batch, 0, 0};
                    const float64 zero = 0.0f;
                    v_f32_st_tnsr_partial(output_coords, output, zero, 0, 0);
                }
                continue;
            }

            const int key_head = value_head / head_repeat;
            int5 packed_coords = {key_head * key_dim, batch, 0, 0, 0};
            float64 q_lo = v_f32_ld_tnsr_b(packed_coords, packed_qkv);
            packed_coords[0] += 64;
            float64 q_hi = v_f32_ld_tnsr_b(packed_coords, packed_qkv);
            packed_coords[0] = key_offset + key_head * key_dim;
            float64 k_lo = v_f32_ld_tnsr_b(packed_coords, packed_qkv);
            packed_coords[0] += 64;
            float64 k_hi = v_f32_ld_tnsr_b(packed_coords, packed_qkv);

            float64 q_norm = v_f32_mac_b(q_lo, q_lo, q_hi * q_hi);
            q_norm = v_f32_reduce_add(q_norm);
            q_norm = v_f32_shuffle_b(q_norm, broadcast_lane_zero, 0, q_norm);
            q_norm = v_rsqrt_f32(q_norm + 1e-6f) * q_scale;
            q_lo = q_lo * q_norm;
            q_hi = q_hi * q_norm;

            float64 k_norm = v_f32_mac_b(k_lo, k_lo, k_hi * k_hi);
            k_norm = v_f32_reduce_add(k_norm);
            k_norm = v_f32_shuffle_b(k_norm, broadcast_lane_zero, 0, k_norm);
            k_norm = v_rsqrt_f32(k_norm + 1e-6f);
            k_lo = k_lo * k_norm;
            k_hi = k_hi * k_norm;

            int5 scalar_coords = {value_head, batch, 0, 0, 0};
            const float decay_value = s_f32_ld_g(gen_addr(scalar_coords, decay));
            const float beta_value = s_f32_ld_g(gen_addr(scalar_coords, beta));

            for (int value_row = value_start; value_row < value_end; ++value_row) {
                int5 state_coords = {0, value_row, value_head, state_slot, 0};
                float64 state_lo = v_f32_ld_tnsr_b(state_coords, state_cache);
                state_coords[0] = 64;
                float64 state_hi = v_f32_ld_tnsr_b(state_coords, state_cache);

                int5 value_coords = {value_offset + value_head * value_dim + value_row, batch, 0, 0, 0};
                const float v_value = s_f32_ld_g(gen_addr(value_coords, packed_qkv));

                state_lo = state_lo * decay_value;
                state_hi = state_hi * decay_value;

                float64 projection = state_lo * k_lo;
                projection = v_f32_mac_b(state_hi, k_hi, projection);
                projection = v_f32_reduce_add(projection);
                projection = v_f32_shuffle_b(projection, broadcast_lane_zero, 0, projection);
                const float64 delta = (v_value - projection) * beta_value;

                state_lo = v_f32_mac_b(k_lo, delta, state_lo);
                state_hi = v_f32_mac_b(k_hi, delta, state_hi);

                float64 out_value = state_lo * q_lo;
                out_value = v_f32_mac_b(state_hi, q_hi, out_value);
                out_value = v_f32_reduce_add(out_value);

                // Public CustomOp metadata cannot express input/output aliasing.
                // Update the cache directly; new_state is a dependency output.
                (void)new_state;
                state_coords[0] = 0;
                v_f32_st_tnsr(state_coords, state_cache, state_lo);
                state_coords[0] = 64;
                v_f32_st_tnsr(state_coords, state_cache, state_hi);

                int5 output_coords = {value_row, value_head, batch, 0, 0};
                v_f32_st_tnsr_partial(output_coords, output, out_value, 0, 0);
            }
        }
    }
}

