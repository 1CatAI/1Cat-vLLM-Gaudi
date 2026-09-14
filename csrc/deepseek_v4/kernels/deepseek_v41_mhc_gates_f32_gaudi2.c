// SPDX-License-Identifier: Apache-2.0
// Decode-specialized V4.1 mHC gate preparation. The FP32 control projection
// and residual RMS reduction remain independent producers; this kernel fuses
// their small consumers so C1 pays one TPC launch instead of the sigmoid,
// softmax, epsilon and Sinkhorn launch chain. The single packed output keeps
// the production Gaudi graph compiler from extracting a tuple placeholder.
#define STREAMS 4
#define HC_EPS 1.0e-6f
#define SINKHORN_ITERS 20

static inline float64 sigmoid_f32(float64 value)
{
    return v_reciprocal_f32(1.0f + v_exp_cephes_f32(-value));
}

void main(
    tensor raw_mixes,
    tensor rrms,
    tensor hc_scale,
    tensor hc_base,
    tensor gates_out)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int token = start[0]; token < end[0]; ++token) {
        const float64 inverse_rms = s_f32_ld_g(
            gen_addr((int5){0, token, 0, 0, 0}, rrms));
        float64 scales[3];
        for (int group = 0; group < 3; ++group) {
            scales[group] = s_f32_ld_g(
                gen_addr((int5){group, 0, 0, 0, 0}, hc_scale));
        }

        for (int stream = 0; stream < STREAMS; ++stream) {
            const float64 pre_logit =
                s_f32_ld_g(gen_addr((int5){stream, token, 0, 0, 0}, raw_mixes))
                * inverse_rms * scales[0]
                + s_f32_ld_g(gen_addr((int5){stream, 0, 0, 0, 0}, hc_base));
            const int post_index = STREAMS + stream;
            const float64 post_logit =
                s_f32_ld_g(gen_addr((int5){post_index, token, 0, 0, 0}, raw_mixes))
                * inverse_rms * scales[1]
                + s_f32_ld_g(gen_addr((int5){post_index, 0, 0, 0, 0}, hc_base));
            const float64 pre = sigmoid_f32(pre_logit) + HC_EPS;
            const float64 post = sigmoid_f32(post_logit) * 2.0f;
            v_f32_st_tnsr_partial((int5){stream, token, 0, 0, 0}, gates_out, pre, 0, 0);
            v_f32_st_tnsr_partial((int5){STREAMS + stream, token, 0, 0, 0}, gates_out, post, 0, 0);
        }

        float64 values[STREAMS][STREAMS];
        for (int row = 0; row < STREAMS; ++row) {
            float64 maximum = -3.402823466e+38f;
            for (int col = 0; col < STREAMS; ++col) {
                const int index = 2 * STREAMS + row * STREAMS + col;
                const float64 value =
                    s_f32_ld_g(gen_addr((int5){index, token, 0, 0, 0}, raw_mixes))
                    * inverse_rms * scales[2]
                    + s_f32_ld_g(gen_addr((int5){index, 0, 0, 0, 0}, hc_base));
                values[row][col] = value;
                maximum = v_f32_max_b(value, maximum);
            }
            float64 sum = 0.0f;
            for (int col = 0; col < STREAMS; ++col) {
                values[row][col] = v_exp_cephes_f32(values[row][col] - maximum);
                sum += values[row][col];
            }
            for (int col = 0; col < STREAMS; ++col) {
                values[row][col] = values[row][col] / sum + HC_EPS;
            }
        }

        // The initial row normalization above is the first Sinkhorn row step,
        // matching the existing softmax + epsilon producer. The remaining
        // sequence is identical to deepseek_v4_sinkhorn4_gaudi2.
        for (int iteration = 0; iteration < SINKHORN_ITERS; ++iteration) {
            if (iteration != 0) {
                for (int row = 0; row < STREAMS; ++row) {
                    float64 sum = (values[row][0] + values[row][1])
                                + (values[row][2] + values[row][3]);
                    sum += HC_EPS;
                    for (int col = 0; col < STREAMS; ++col) values[row][col] /= sum;
                }
            }
            for (int col = 0; col < STREAMS; ++col) {
                float64 sum = 0.0f;
                for (int row = 0; row < STREAMS; ++row) sum += values[row][col];
                sum += HC_EPS;
                for (int row = 0; row < STREAMS; ++row) values[row][col] /= sum;
            }
        }
        for (int row = 0; row < STREAMS; ++row) {
            for (int col = 0; col < STREAMS; ++col) {
                const int index = 2 * STREAMS + row * STREAMS + col;
                v_f32_st_tnsr_partial((int5){index, token, 0, 0, 0}, gates_out,
                                      values[row][col], 0, 0);
            }
        }
    }
}
