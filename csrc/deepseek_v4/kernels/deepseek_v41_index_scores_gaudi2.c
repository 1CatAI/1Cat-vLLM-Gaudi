// SPDX-License-Identifier: Apache-2.0
// Runtime-sized C1 index scan. Capacity does not determine executed row count.
// Query/key FP4 roundtrip is performed by the existing producer.
static inline bfloat128 index_key(tensor cache, int row) {
    const uchar256 lanes = V_LANE_ID_8;
    uchar256 raw = v_u8_ld_tnsr_partial_b((int5){0,row}, cache, 63, 0);
    raw = v_u8_mov_dual_group_all_b(raw, 0xffffffff, 0,0,0,0, MkWrA(3,3,3,3), (uchar256){0});
    const uchar256 expanded = v_u8_shuffle_b(raw, (lanes >> 1) | 0x80, 0, (uchar256){0});
    const ushort128 codes = (convert_uchar256_to_ushort256(expanded, SW_LINEAR).v1 >>
                             (((ushort128)V_LANE_ID_16 & 1) << 2)) & 15;
    const ushort128 magnitude = codes & 7;
    ushort128 bits = (magnitude << 6) + 0x3f00;
    bits = v_u16_sel_eq_u16_b(magnitude, 1, 0x3f00, bits);
    bits = v_u16_sel_eq_u16_b(magnitude, 0, 0, bits) | ((codes & 8) << 12);
    uchar256 scales = v_u8_ld_tnsr_partial_b((int5){64,row}, cache, 3, 0);
    scales = v_u8_mov_dual_group_all_b(scales, 0xffffffff, 0,0,0,0, MkWrA(3,3,3,3), (uchar256){0});
    scales = v_u8_shuffle_b(scales, (lanes >> 5) | 0x80, 0, (uchar256){0});
    ushort128 scale = convert_uchar256_to_ushort256(scales, SW_LINEAR).v1 << 7;
    scale = v_u16_sel_eq_u16_b(scale, 0, 0, scale);
    bfloat128 result = v_bf16_mul_b(*((bfloat128*)&bits), *((bfloat128*)&scale));
    return v_bf16_sel_eq_bf16_b(result, (bfloat)0, (bfloat)0, result);
}

static inline float64 round_bf16(float64 x) {
    const bfloat128 b = v_convert_f32_to_bf16_all_b((float128){x,x}, SW_RHNE);
    return v_convert_bf16_to_f32_all_b(b).v1;
}

void main(tensor q, tensor weights, tensor cache, tensor pages, tensor positions,
          tensor candidates, tensor scores, tensor block_scores, int ratio, int reindex) {
    const int5 begin = get_index_space_offset(), end = begin + get_index_space_size();
    const int visible = (s_i32_ld_g(gen_addr((int5){0}, positions)) + 1) / ratio;
    // The selection emitter publishes the causal prefix directly in this case.
    if (visible <= 512) return;
    const int blocks = reindex ? 2048 : (visible + 7) / 8;
    const int page_rows = 128 / ratio;
    for (int worker = begin[0]; worker < end[0]; ++worker) {
        const int first = (blocks * worker / 24);
        const int last = (blocks * (worker + 1) / 24);
        for (int block = first; block < last; ++block) {
            const int logical_block = reindex ? s_i32_ld_g(gen_addr((int5){block,0},candidates)) : block;
            float64 maximum = as_float64((int64)0xff800000);
            for (int offset = 0; offset < 8; ++offset) {
                const int logical = logical_block * 8 + offset;
                float64 score = as_float64((int64)0xff800000);
                if (logical_block >= 0 && logical < visible) {
                    const int page = s_i32_ld_g(gen_addr((int5){logical / page_rows},pages));
                    const int row = page * page_rows + logical % page_rows;
                    const bfloat128 key = index_key(cache, row);
                    float64 partial[2] = {0,0};
                    for (int shard = 0; shard < 2; ++shard) {
                        for (int head = 0; head < 16; ++head) {
                            const int h = shard * 16 + head;
                            const bfloat128 query = v_bf16_ld_tnsr_b((int5){0,h,0}, q);
                            const float128 product = v_bf16_mac_acc32_b(query,key,(float128){0},0);
                            float64 sum = v_f32_reduce_add(product.v1 + product.v2);
                            const float64 dot = round_bf16(v_f32_shuffle_b(sum,(uchar256)0x80,0,sum));
                            const bfloat w = s_bf16_ld_g(gen_addr((int5){h,0},weights));
                            const float wf = s_convert_bf16_to_f32(w,0);
                            partial[shard] += round_bf16(v_f32_max_b(dot,0.0f) * wf);
                        }
                        partial[shard] = round_bf16(partial[shard]);
                    }
                    score = round_bf16(partial[0] + partial[1]);
                }
                v_f32_st_tnsr_partial((int5){block*8+offset},scores,score,0,0);
                maximum = v_f32_max_b(maximum,score);
            }
            if (!reindex && block == (visible - 1) / 8) maximum = as_float64((int64)0x7f800000);
            v_f32_st_tnsr_partial((int5){block},block_scores,maximum,0,0);
        }
    }
}
