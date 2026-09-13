// SPDX-License-Identifier: Apache-2.0
// Repeated-max selection follows the upstream DeepSeek top-6 contract.
// Input scores retain the model's FP32 softplus/sqrt computation.
void main(tensor input, tensor text_bias, tensor image_bias, tensor image_mask,
          tensor output_ids, tensor output_weights) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int64 lanes = (int64)V_LANE_ID_32;
    for (int token = start[0]; token < end[0]; ++token) {
        const int5 image_at = {token, 0, 0, 0, 0};
        const char is_image = s_i8_ld_g(gen_addr(image_at, image_mask));
        float64 scores[6], choices[6], weights[6];
        bool64 eligible[6];
        int64 ids[6];
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            const int5 at = {c * 64, token, 0, 0, 0};
            const int5 bias_at = {c * 64, 0, 0, 0, 0};
            scores[c] = v_f32_ld_tnsr_b(at, input);
            const float64 bias = is_image ? v_f32_ld_tnsr_b(bias_at, image_bias) : v_f32_ld_tnsr_b(bias_at, text_bias);
            choices[c] = scores[c] + bias;
            eligible[c] = (bool64)1;
        }
        float64 sum = 0;
        #pragma unroll
        for (int rank = 0; rank < 6; ++rank) {
            float64 best = choices[0];
            #pragma unroll
            for (int c = 1; c < 6; ++c) best = v_f32_max_b(best, choices[c]);
            best = v_f32_reduce_max(best);
            int64 winner = 2147483647;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                const bool64 match = v_f32_cmp_eq_b(choices[c], best) & eligible[c];
                const int64 candidate = v_i32_mov_vb(lanes + c * 64, 0, (int64)2147483647, match, 0);
                winner = v_i32_min_b(winner, candidate);
            }
            winner = v_i32_reduce_min(winner);
            float64 selected = 0;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                const bool64 match = v_i32_cmp_eq_b(lanes + c * 64, winner);
                selected += v_f32_mov_vb(scores[c], 0, 0, match, 0);
                choices[c] = v_f32_mov_vb(as_float64((int64)0xff800000), 0, choices[c], match, 0);
                eligible[c] = eligible[c] & v_i32_cmp_neq_b(lanes + c * 64, winner);
            }
            weights[rank] = v_f32_reduce_add(selected);
            ids[rank] = winner;
            sum += weights[rank];
        }
        const float64 denominator = sum + 1.0e-20f;
        float64 inverse = v_reciprocal_f32(denominator);
        inverse = inverse * (2.0f - denominator * inverse);
        #pragma unroll
        for (int rank = 0; rank < 6; ++rank) {
            const int5 at = {rank, token, 0, 0, 0};
            v_i32_st_tnsr_partial(at, output_ids, ids[rank], 0, 0);
            v_f32_st_tnsr_partial(at, output_weights, weights[rank] * inverse * 1.5f, 0, 0);
        }
    }
}
