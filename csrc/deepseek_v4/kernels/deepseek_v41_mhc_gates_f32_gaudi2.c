// SPDX-License-Identifier: Apache-2.0
// Keep each 4x4 mHC matrix in SIMD lanes. The scalar-per-entry implementation
// replicated each matrix entry across a full vector and spilled local vectors.
// Row and column shuffle reductions retain its FP32 addition order.
#define HC_EPS 1.0e-6f
#define SINKHORN_ITERS 20

static inline float64 sigmoid_f32(float64 value)
{
    return v_reciprocal_f32(1.0f + v_exp_cephes_f32(-value));
}

static inline uchar256 matrix_direction(uint64 index)
{
    const uint64 selected = (index & 7) | ((index & 8) << 2) | 0x80;
    uint256 packed;
    packed.v1 = selected;
    packed.v2 = selected;
    packed.v3 = selected;
    packed.v4 = selected;
    return v_convert_u32_to_u8_all_b(packed);
}

void main(tensor raw_mixes, tensor rrms, tensor hc_scale, tensor hc_base,
          tensor gates_out)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const uint64 lane = read_lane_id_4b_b();
    const uchar256 r0 = matrix_direction(lane & 0xfffffffc);
    const uchar256 r1 = matrix_direction((lane & 0xfffffffc) | 1);
    const uchar256 r2 = matrix_direction((lane & 0xfffffffc) | 2);
    const uchar256 r3 = matrix_direction((lane & 0xfffffffc) | 3);
    const uchar256 c0 = matrix_direction(lane & 3);
    const uchar256 c1 = matrix_direction((lane & 3) | 4);
    const uchar256 c2 = matrix_direction((lane & 3) | 8);
    const uchar256 c3 = matrix_direction((lane & 3) | 12);
    for (int token = start[0]; token < end[0]; ++token) {
        const float64 inverse_rms = s_f32_ld_g(
            gen_addr((int5){0, token, 0, 0, 0}, rrms));
        const float64 pre_scale = s_f32_ld_g(gen_addr((int5){0}, hc_scale));
        const float64 post_scale = s_f32_ld_g(gen_addr((int5){1, 0, 0, 0, 0}, hc_scale));
        const float64 comb_scale = s_f32_ld_g(gen_addr((int5){2, 0, 0, 0, 0}, hc_scale));

        const float64 pre_raw = v_f32_ld_tnsr_partial_b(
            (int5){0, token, 0, 0, 0}, raw_mixes, 3, 0);
        const float64 pre_base = v_f32_ld_tnsr_partial_b((int5){0}, hc_base, 3, 0);
        const float64 pre = sigmoid_f32(pre_raw * inverse_rms * pre_scale + pre_base) + HC_EPS;
        v_f32_st_tnsr_partial((int5){0, token, 0, 0, 0}, gates_out, pre, 3, 0);

        const float64 post_raw = v_f32_ld_tnsr_partial_b(
            (int5){4, token, 0, 0, 0}, raw_mixes, 3, 0);
        const float64 post_base = v_f32_ld_tnsr_partial_b(
            (int5){4, 0, 0, 0, 0}, hc_base, 3, 0);
        const float64 post = sigmoid_f32(post_raw * inverse_rms * post_scale + post_base) * 2.0f;
        v_f32_st_tnsr_partial((int5){4, token, 0, 0, 0}, gates_out, post, 3, 0);

        const float64 raw = v_f32_ld_tnsr_partial_b(
            (int5){8, token, 0, 0, 0}, raw_mixes, 15, 0);
        const float64 base = v_f32_ld_tnsr_partial_b(
            (int5){8, 0, 0, 0, 0}, hc_base, 15, 0);
        float64 values = raw * inverse_rms * comb_scale + base;
        float64 maximum = -3.402823466e+38f;
        maximum = v_f32_max_b(v_f32_shuffle_b(values, r0, 0, 0.0f), maximum);
        maximum = v_f32_max_b(v_f32_shuffle_b(values, r1, 0, 0.0f), maximum);
        maximum = v_f32_max_b(v_f32_shuffle_b(values, r2, 0, 0.0f), maximum);
        maximum = v_f32_max_b(v_f32_shuffle_b(values, r3, 0, 0.0f), maximum);
        values = v_exp_cephes_f32(values - maximum);
        float64 sum = 0.0f;
        sum += v_f32_shuffle_b(values, r0, 0, 0.0f);
        sum += v_f32_shuffle_b(values, r1, 0, 0.0f);
        sum += v_f32_shuffle_b(values, r2, 0, 0.0f);
        sum += v_f32_shuffle_b(values, r3, 0, 0.0f);
        values = values / sum + HC_EPS;

        for (int iteration = 0; iteration < SINKHORN_ITERS; ++iteration) {
            if (iteration != 0) {
                const float64 a = v_f32_shuffle_b(values, r0, 0, 0.0f)
                                + v_f32_shuffle_b(values, r1, 0, 0.0f);
                const float64 b = v_f32_shuffle_b(values, r2, 0, 0.0f)
                                + v_f32_shuffle_b(values, r3, 0, 0.0f);
                values /= (a + b) + HC_EPS;
            }
            float64 column_sum = 0.0f;
            column_sum += v_f32_shuffle_b(values, c0, 0, 0.0f);
            column_sum += v_f32_shuffle_b(values, c1, 0, 0.0f);
            column_sum += v_f32_shuffle_b(values, c2, 0, 0.0f);
            column_sum += v_f32_shuffle_b(values, c3, 0, 0.0f);
            values /= column_sum + HC_EPS;
        }
        v_f32_st_tnsr_partial((int5){8, token, 0, 0, 0}, gates_out, values, 15, 0);
    }
}
