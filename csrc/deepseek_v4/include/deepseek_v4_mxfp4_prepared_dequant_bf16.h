// SPDX-License-Identifier: Apache-2.0
// Decode a load-time Q16/S16 layout directly into BF16 SRAM tiles.

#ifndef DSV4_MXFP4_PREPARED_NORMAL
#define DSV4_MXFP4_PREPARED_NORMAL 0
#endif

#ifndef DSV4_MXFP4_PREPARED_STORE_DIAGNOSTIC
#define DSV4_MXFP4_PREPARED_STORE_DIAGNOSTIC 0
#endif

#ifndef DSV4_MXFP4_PREPARED_N_BLOCK_OFFSET
#define DSV4_MXFP4_PREPARED_N_BLOCK_OFFSET 0
#endif

#define DSV4_MXFP4_PREPARED_N_TILE 128

#ifndef DSV41_MXFP4_PREPARED_K128
#define DSV41_MXFP4_PREPARED_K128 0
#endif

static inline ushort128 decode_prepared_bits(ushort128 nibble, ushort128 scale)
{
    const ushort128 magnitude = v_u16_and_b(nibble, 7);
    const ushort128 sign = v_u16_shl_b(v_u16_and_b(nibble, 8), 12);
    ushort128 bits = scale + v_u16_shl_b(magnitude, 6) - 128;
    bits = v_u16_sel_eq_u16_b(magnitude, 1, bits - 64, bits);
    bits = v_u16_min_b(bits, 0x7f80);
#if !DSV4_MXFP4_PREPARED_NORMAL
    const ushort128 scale_code = v_u16_shr_b(scale, 7);
    const ushort128 subnormal = v_u16_sel_less_u16_b(
        magnitude, 4, v_u16_shl_b(magnitude, 5), bits);
    bits = v_u16_sel_eq_u16_b(scale_code, 0, subnormal, bits);
    const ushort128 half_subnormal = v_u16_sel_eq_u16_b(
        magnitude, 1, 0x0040, bits);
    bits = v_u16_sel_eq_u16_b(scale_code, 1, half_subnormal, bits);
#endif
    bits = v_u16_sel_eq_u16_b(magnitude, 0, 0, bits);
#if !DSV4_MXFP4_PREPARED_NORMAL
    bits = v_u16_sel_eq_u16_b(scale_code, 255, 0x7fc0, bits);
#endif
    return v_u16_or_b(bits, sign);
}

static inline bfloat256 lookup_pair_and_scale(
    uchar256 unpacked, bfloat128 scale, uchar256 table)
{
    const uchar256 directions = v_u8_or_b(unpacked, 0x80);
    const uchar256 fp8Bits = v_u8_shuffle_b(table, directions, 0, (uchar256){0});
    const bfloat256 values = v_convert_f8_to_bf16_all_b(*((minifloat256*)&fp8Bits));
    const ushort128 negativeZeroBits = 0x8000;
    const bfloat128 negativeZero = *((bfloat128*)&negativeZeroBits);
    bfloat256 scaled;
    scaled.v1 = v_bf16_madd_b(values.v1, scale, negativeZero);
    scaled.v2 = v_bf16_madd_b(values.v2, scale, negativeZero);
    return scaled;
}

void main(tensor expert_ids, tensor q16, tensor s16, tensor lookup, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(q16, 2);
#if DSV41_MXFP4_PREPARED_K128
    const int pairBegin = start[2] * 64;
    const int pairEnd = end[2] * 64;
#else
    const int pairBegin = 0;
    const int pairEnd = get_dim_size(q16, 0) / 64;
#endif
#if !DSV4_MXFP4_PREPARED_STORE_DIAGNOSTIC
    const int groupBegin = pairBegin / 16;
    const int groupEnd = pairEnd / 16;
#endif
#if DSV4_MXFP4_PREPARED_NORMAL && !DSV4_MXFP4_PREPARED_STORE_DIAGNOSTIC
    const uchar256 table = v_u8_ld_tnsr_b((int5){0, 0, 0, 0, 0}, lookup);
#endif
    for (int slot = start[1]; slot < end[1]; ++slot) {
        const int expert = s_i32_ld_g(gen_addr((int5){slot, 0, 0, 0, 0}, expert_ids));
        for (int nBlock = start[0]; nBlock < end[0]; ++nBlock) {
            const int row = nBlock * 128;
            const int sourceNBlock = nBlock + DSV4_MXFP4_PREPARED_N_BLOCK_OFFSET;
            if (expert < 0 || expert >= experts) {
                int5 dst = {row, pairBegin * 2, slot, 0, 0};
                for (int pair = pairBegin; pair < pairEnd; ++pair) {
                    v_bf16_st_tnsr(dst, output, (bfloat128){0});
                    dst[1] += 1;
                    v_bf16_st_tnsr(dst, output, (bfloat128){0});
                    dst[1] += 1;
                }
                continue;
            }
#if DSV4_MXFP4_PREPARED_STORE_DIAGNOSTIC
            int5 dst = {row, pairBegin * 2, slot, 0, 0};
            for (int pair = pairBegin; pair < pairEnd; ++pair) {
                v_bf16_st_tnsr(dst, output, (bfloat128){0});
                dst[1] += 1;
                v_bf16_st_tnsr(dst, output, (bfloat128){0});
                dst[1] += 1;
            }
#else
            for (int group = groupBegin; group < groupEnd; ++group) {
                const bfloat128 scale = v_bf16_ld_tnsr_b(
                    (int5){group * 128, sourceNBlock, expert, 0, 0}, s16);
#if DSV4_MXFP4_PREPARED_NORMAL
                #pragma unroll (16)
#else
                #pragma unroll (8)
#endif
                for (int part = 0; part < 16; ++part) {
                    const int pair = group * 16 + part;
                    const uchar256 unpacked = v_u8_ld_tnsr_b(
                        (int5){pair * 64, sourceNBlock, expert, 0, 0}, q16,
                        SW_UNPACK | SW_UNPCK_4_TO_8);
#if DSV4_MXFP4_PREPARED_NORMAL
                    const bfloat256 decoded = lookup_pair_and_scale(unpacked, scale, table);
                    const bfloat128 decoded0 = decoded.v1;
                    const bfloat128 decoded1 = decoded.v2;
#else
                    const ushort128 unpackedWords = *((ushort128*)&unpacked);
                    const ushort128 shiftedWords = v_u16_shr_b(unpackedWords, 8);
                    const ushort128 nibble0 = v_u16_and_b(unpackedWords, 15);
                    const ushort128 nibble1 = v_u16_and_b(shiftedWords, 15);
                    const ushort128 bits0 = decode_prepared_bits(nibble0, *((ushort128*)&scale));
                    const ushort128 bits1 = decode_prepared_bits(nibble1, *((ushort128*)&scale));
                    const bfloat128 decoded0 = *((bfloat128*)&bits0);
                    const bfloat128 decoded1 = *((bfloat128*)&bits1);
#endif
                    const int k = pair * 2;
                    v_bf16_st_tnsr((int5){row, k, slot, 0, 0}, output, decoded0);
                    v_bf16_st_tnsr((int5){row, k + 1, slot, 0, 0}, output, decoded1);
                }
            }
#endif
        }
    }
}

#undef DSV4_MXFP4_PREPARED_N_TILE
#undef DSV4_MXFP4_PREPARED_N_BLOCK_OFFSET
#undef DSV4_MXFP4_PREPARED_STORE_DIAGNOSTIC
