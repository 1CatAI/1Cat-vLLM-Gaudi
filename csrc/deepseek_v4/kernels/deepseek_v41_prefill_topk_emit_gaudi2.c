// SPDX-License-Identifier: Apache-2.0
static inline unsigned score_key(float value) {
    unsigned bits = *((unsigned*)&value) >> 16;
    const unsigned magnitude = bits & 32767;
    if (magnitude > 32640) return 65535;
    if (!magnitude) bits = 0;
    return bits > 32767 ? (~bits) & 65535 : bits ^ 32768;
}
void main(tensor scores, tensor metadata, tensor values, tensor indices, int columns, int width) {
    const int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
    for (int row = begin[1]; row < end[1]; ++row) {
        if (width == columns) {
            for (int worker = begin[0]; worker < end[0]; ++worker)
                for (int col = worker * 64; col < columns; col += 8 * 64) {
                    const int5 at = {col,row};
                    v_f32_st_tnsr(at, values, v_f32_ld_tnsr_b(at,scores));
                    v_i32_st_tnsr(at, indices, (int64)V_LANE_ID_32 + col);
                }
            continue;
        }
        const unsigned threshold = s_u32_ld_g(gen_addr((int5){0,row},metadata));
        const int greater_total = s_i32_ld_g(gen_addr((int5){1,row},metadata));
        const int budget = width - greater_total;
        for (int worker = begin[0]; worker < end[0]; ++worker) {
            const int greater = s_i32_ld_g(gen_addr((int5){2 + worker * 2,row},metadata));
            int equal = s_i32_ld_g(gen_addr((int5){3 + worker * 2,row},metadata));
            int written = greater + (equal < budget ? equal : budget);
            const int first = columns * worker / 8, last = columns * (worker + 1) / 8;
            for (int col = first; col < last; ++col) {
                const float score = s_f32_ld_g(gen_addr((int5){col,row},scores));
                const unsigned key = score_key(score);
                if (key > threshold || (key == threshold && equal++ < budget)) {
                    const int5 at = {written++,row};
                    s_f32_st_g(gen_addr(at,values),score);
                    s_i32_st_g(gen_addr(at,indices),col);
                }
            }
        }
    }
}
