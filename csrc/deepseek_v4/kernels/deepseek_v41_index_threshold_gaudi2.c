// SPDX-License-Identifier: Apache-2.0
// Scores have a BF16 boundary. Exact ordered-bit selection needs 16 passes,
// with each pass bounded by current device length, not allocation capacity.
static inline uint64 ordered(float64 values) {
    const uint64 bits = as_uint64(values) >> 16;
    return v_u32_sel_grt_u32_b(bits, 32767, (~bits)&65535, bits^32768);
}
static inline int64 sum_broadcast(int64 value) {
    value=v_i32_reduce_add(value);
    return v_i32_shuffle_b(value,(uchar256)0x80,0,value);
}
void main(tensor scores, tensor positions, tensor metadata, int ratio, int reindex, int blocks_mode) {
    int count=(s_i32_ld_g(gen_addr((int5){0},positions))+1)/ratio;
    if (count <= 512) return;
    count = reindex ? 16384 : count;
    if (blocks_mode) count=(count+7)/8;
    const int width=blocks_mode ? 2048 : 512;
    const int64 lanes=(int64)V_LANE_ID_32;
    uint64 threshold=0;
    for (int bit=15;bit>=0;--bit) {
        const uint64 trial=threshold | (1u<<bit);
        int64 total=0;
        for(int offset=0;offset<count;offset+=64) {
            const float64 score=v_f32_ld_tnsr_b((int5){offset},scores);
            const bool64 valid=v_i32_cmp_less_b(lanes+offset,count) & v_f32_cmp_grt_b(score,as_float64((int64)0xff800000));
            const bool64 keep=valid & v_u32_cmp_geq_b(ordered(score),trial);
            total += v_i32_mov_vb((int64)1,0,(int64)0,keep,0);
        }
        total=sum_broadcast(total);
        threshold=v_u32_sel_geq_i32_b(total,width,trial,threshold);
    }
    // Prefix counts permit disjoint, ordered output emission by 24 workers.
    int64 greater=0,equal=0;
    for(int worker=0;worker<24;++worker) {
        v_i32_st_tnsr_partial((int5){2+worker*2},metadata,greater,0,0);
        v_i32_st_tnsr_partial((int5){3+worker*2},metadata,equal,0,0);
        const int first=count*worker/24,last=count*(worker+1)/24;
        int64 local_greater=0,local_equal=0;
        for(int offset=first;offset<last;offset+=64) {
            const float64 score=v_f32_ld_tnsr_b((int5){offset},scores);
            const bool64 valid=v_i32_cmp_less_b(lanes+offset,last) & v_f32_cmp_grt_b(score,as_float64((int64)0xff800000));
            const uint64 key=ordered(score);
            local_greater+=v_i32_mov_vb((int64)1,0,(int64)0,valid & v_u32_cmp_grt_b(key,threshold),0);
            local_equal+=v_i32_mov_vb((int64)1,0,(int64)0,valid & v_u32_cmp_eq_b(key,threshold),0);
        }
        greater+=sum_broadcast(local_greater);equal+=sum_broadcast(local_equal);
    }
    v_u32_st_tnsr_partial((int5){0},metadata,threshold,0,0);
    v_i32_st_tnsr_partial((int5){1},metadata,greater,0,0);
}
