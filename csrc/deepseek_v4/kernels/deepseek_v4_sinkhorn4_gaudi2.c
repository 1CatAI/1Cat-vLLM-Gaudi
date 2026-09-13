// SPDX-License-Identifier: Apache-2.0
// The input is the existing FP32 softmax + epsilon result. Keep all
// surrounding mHC projections, activations and BF16 boundaries outside.
void main(tensor input, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int token = start[0]; token < end[0]; ++token) {
        float64 values[4][4];
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                const int5 coord = {col, row, token, 0, 0};
                values[row][col] = s_f32_ld_g(gen_addr(coord, input));
            }
        }
        for (int iteration = 0; iteration < 20; ++iteration) {
            // The first row normalization is the preceding softmax.
            if (iteration != 0) {
                for (int row = 0; row < 4; ++row) {
                    // Production contiguous-axis reduction uses an adjacent
                    // pair tree. The noncontiguous column axis below is
                    // sequential. Preserve both FP32 addition orders.
                    float64 sum = (values[row][0] + values[row][1])
                                + (values[row][2] + values[row][3]);
                    sum += 1.0e-6f;
                    for (int col = 0; col < 4; ++col) values[row][col] /= sum;
                }
            }
            for (int col = 0; col < 4; ++col) {
                float64 sum = 0.0f;
                for (int row = 0; row < 4; ++row) sum += values[row][col];
                sum += 1.0e-6f;
                for (int row = 0; row < 4; ++row) values[row][col] /= sum;
            }
        }
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                const int5 coord = {col, row, token, 0, 0};
                v_f32_st_tnsr_partial(coord, output, values[row][col], 0, 0);
            }
        }
    }
}
