// SPDX-License-Identifier: Apache-2.0
// N256 weights: adjacent N nibbles at each K, one full FP8 vector store.
#ifndef DSV41_N256_FP8
#define DSV41_N256_FP8 1
#endif
#ifndef DSV41_N256_NORMAL_BF16
#define DSV41_N256_NORMAL_BF16 0
#endif
#ifndef DSV41_N256_PREFETCH
#define DSV41_N256_PREFETCH 8
#endif
#ifndef DSV41_N256_FUSE_SLOTS
#define DSV41_N256_FUSE_SLOTS 0
#endif
#ifndef DSV41_N256_HORIZONTAL
#define DSV41_N256_HORIZONTAL 0
#endif
#ifndef DSV41_N256_REUSE_SLOTS
#define DSV41_N256_REUSE_SLOTS 0
#endif

#if DSV41_N256_REUSE_SLOTS
#define DSV41_N256_WRITE(ENCODED) do { \
    destination[2] = slot; \
    v_f8_st_tnsr(destination, output, *((minifloat256*)&(ENCODED))); \
    int5 duplicate = destination; \
    duplicate[2] += 1; \
    v_f8_st_tnsr(duplicate, output, *((minifloat256*)&(ENCODED)), 0, repeat_end > slot + 1); \
} while (0)
#else
#define DSV41_N256_WRITE(ENCODED) \
    v_f8_st_tnsr(destination, output, *((minifloat256*)&(ENCODED)))
#endif

static inline ushort128 exact_bf16(ushort128 nibble, ushort128 code)
{
    const ushort128 magnitude = nibble & 7;
    const ushort128 sign = (nibble & 8) << 12;
    ushort128 bits = (code << 7) + (magnitude << 6) - 128;
    bits = v_u16_sel_eq_u16_b(magnitude, 1, bits - 64, bits);
    bits = v_u16_min_b(bits, 0x7f80);
    const ushort128 tiny = v_u16_sel_less_u16_b(magnitude, 4, magnitude << 5, bits);
    bits = v_u16_sel_eq_u16_b(code, 0, tiny, bits);
    const ushort128 half_value = v_u16_sel_eq_u16_b(magnitude, 1, 0x40, bits);
    bits = v_u16_sel_eq_u16_b(code, 1, half_value, bits);
    bits = v_u16_sel_eq_u16_b(magnitude, 0, 0, bits);
    bits = v_u16_sel_eq_u16_b(code, 255, 0x7fc0, bits);
    return bits | sign;
}

#define DSV41_N256_STORE(WEIGHTS) do { \
    const uchar256 direction = (WEIGHTS) | 0x80; \
    const uchar256 base = v_u8_shuffle_b(table, direction, 0, direction); \
    const uchar256 encoded = v_u8_sel_eq_u8_b((WEIGHTS), 7, base, base + delta, SW_MASK_EQ_ZERO); \
    DSV41_N256_WRITE(encoded); \
    destination[1] += 1; \
} while (0)

void main(tensor ids, tensor q16, tensor planes, tensor lookup, tensor output)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int experts = get_dim_size(q16, 2);
#if DSV41_N256_FP8 || DSV41_N256_NORMAL_BF16
    const uchar256 table = v_u8_ld_tnsr_b((int5){0}, lookup);
#endif
#if DSV41_N256_FUSE_SLOTS
    const int first_slot = 0;
    const int last_slot = get_dim_size(ids, 0);
    const int first_group = start[1] * 4;
    const int last_group = end[1] * 4;
#if DSV41_N256_FP8
    const int first_k = start[1] * 128;
    const int last_k = end[1] * 128;
#endif
#else
#if DSV41_N256_REUSE_SLOTS
    const int first_slot = start[1] * DSV41_N256_REUSE_SLOTS;
    const int last_slot = s_i32_min(end[1] * DSV41_N256_REUSE_SLOTS, get_dim_size(ids, 0));
#else
    const int first_slot = start[1];
    const int last_slot = end[1];
#endif
    const int first_group = start[2] * 4;
    const int last_group = end[2] * 4;
#if DSV41_N256_FP8
    const int first_k = start[2] * 128;
    const int last_k = end[2] * 128;
#endif
#endif
    for (int slot = first_slot; slot < last_slot; ++slot) {
#if !DSV41_N256_HORIZONTAL
        const int expert = s_i32_ld_g(gen_addr((int5){slot}, ids));
#if DSV41_N256_REUSE_SLOTS
        // Sharing is restricted to one affine batch access tile. Synapse
        // can slice this axis with MME while preserving SRAM ownership.
        if (slot % DSV41_N256_REUSE_SLOTS != 0 && expert >= 0 && expert < experts &&
            expert == s_i32_ld_g(gen_addr((int5){slot - 1}, ids)))
            continue;
        int repeat_end = slot + 1;
        const int tile_end = s_i32_min(last_slot, (slot / DSV41_N256_REUSE_SLOTS + 1) * DSV41_N256_REUSE_SLOTS);
        while (repeat_end < tile_end &&
               expert == s_i32_ld_g(gen_addr((int5){repeat_end}, ids)))
            ++repeat_end;
#endif
#if DSV41_N256_FP8
        if (expert < 0 || expert >= experts) {
            for (int block = start[0]; block < end[0]; ++block) {
                for (int k = first_k; k < last_k; ++k) {
                    v_f8_st_tnsr((int5){block * 256, k, slot}, output, (minifloat256){0});
                }
            }
            continue;
        }
#endif
#endif
        for (int block = start[0]; block < end[0]; ++block) {
#if DSV41_N256_HORIZONTAL
            const int blocks_per_expert = get_dim_size(q16, 1);
            const int route = block / blocks_per_expert;
            const int source_block = block % blocks_per_expert;
            const int expert = s_i32_ld_g(gen_addr((int5){slot * DSV41_N256_HORIZONTAL + route}, ids));
            if (expert < 0 || expert >= experts) {
                for (int k = first_k; k < last_k; ++k)
                    v_f8_st_tnsr((int5){block * 256, k, slot}, output, (minifloat256){0});
                continue;
            }
#else
            const int source_block = block;
#endif
            const int row = block * 256;
            for (int group = first_group; group < last_group; ++group) {
#if DSV41_N256_FP8
                const uchar256 delta = v_u8_ld_tnsr_b(
                    (int5){group * 256 + 128, source_block, expert}, planes);
#else
                const bool valid = expert >= 0 && expert < experts;
                const uchar256 original = v_u8_ld_tnsr_b(
                    (int5){group * 256, source_block, expert}, planes, 0, (uchar256){0}, valid);
                const ushort256 codes = convert_uchar256_to_ushort256(original, SW_LINEAR);
#if DSV41_N256_NORMAL_BF16
                const ushort128 scale_low_bits = codes.v1 << 7;
                const ushort128 scale_high_bits = codes.v2 << 7;
                const bfloat128 scale_low = *((bfloat128*)&scale_low_bits);
                const bfloat128 scale_high = *((bfloat128*)&scale_high_bits);
                const ushort128 negative_zero_bits = 0x8000;
                const bfloat128 negative_zero = *((bfloat128*)&negative_zero_bits);
#endif
#endif
                int5 source = {group * 2048, source_block, expert};
                int5 destination = {row, group * 32, slot};
#if DSV41_N256_FP8
                // Keep the reference eight-vector schedule available so the
                // production consumer can compare both binaries in one graph.
                uchar256 pending0 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending1 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending2 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending3 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending4 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending5 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending6 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending7 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
#if DSV41_N256_PREFETCH == 16
                uchar256 pending8 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending9 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending10 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending11 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending12 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending13 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending14 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                uchar256 pending15 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                source[0] += 64;
                for (int batch = 0; batch < 1; ++batch) {
#else
                for (int batch = 0; batch < 3; ++batch) {
#endif
                    const uchar256 next0 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next1 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next2 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next3 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next4 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next5 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next6 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next7 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
#if DSV41_N256_PREFETCH == 16
                    const uchar256 next8 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next9 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next10 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next11 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next12 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next13 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next14 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
                    const uchar256 next15 = v_u8_ld_tnsr_b(source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    source[0] += 64;
#endif
                    DSV41_N256_STORE(pending0);
                    DSV41_N256_STORE(pending1);
                    DSV41_N256_STORE(pending2);
                    DSV41_N256_STORE(pending3);
                    DSV41_N256_STORE(pending4);
                    DSV41_N256_STORE(pending5);
                    DSV41_N256_STORE(pending6);
                    DSV41_N256_STORE(pending7);
#if DSV41_N256_PREFETCH == 16
                    DSV41_N256_STORE(pending8);
                    DSV41_N256_STORE(pending9);
                    DSV41_N256_STORE(pending10);
                    DSV41_N256_STORE(pending11);
                    DSV41_N256_STORE(pending12);
                    DSV41_N256_STORE(pending13);
                    DSV41_N256_STORE(pending14);
                    DSV41_N256_STORE(pending15);
#endif
                    pending0 = next0;
                    pending1 = next1;
                    pending2 = next2;
                    pending3 = next3;
                    pending4 = next4;
                    pending5 = next5;
                    pending6 = next6;
                    pending7 = next7;
#if DSV41_N256_PREFETCH == 16
                    pending8 = next8;
                    pending9 = next9;
                    pending10 = next10;
                    pending11 = next11;
                    pending12 = next12;
                    pending13 = next13;
                    pending14 = next14;
                    pending15 = next15;
#endif
                }
                DSV41_N256_STORE(pending0);
                DSV41_N256_STORE(pending1);
                DSV41_N256_STORE(pending2);
                DSV41_N256_STORE(pending3);
                DSV41_N256_STORE(pending4);
                DSV41_N256_STORE(pending5);
                DSV41_N256_STORE(pending6);
                DSV41_N256_STORE(pending7);
#if DSV41_N256_PREFETCH == 16
                DSV41_N256_STORE(pending8);
                DSV41_N256_STORE(pending9);
                DSV41_N256_STORE(pending10);
                DSV41_N256_STORE(pending11);
                DSV41_N256_STORE(pending12);
                DSV41_N256_STORE(pending13);
                DSV41_N256_STORE(pending14);
                DSV41_N256_STORE(pending15);
#endif
#else
                #pragma unroll (8)
                for (int part = 0; part < 32; ++part) {
#if DSV41_N256_FP8
                    const uchar256 unpacked = v_u8_ld_tnsr_b(
                        source, q16, SW_UNPACK | SW_UNPCK_4_TO_8);
                    // Every shuffle lane is enabled; its income is never used.
                    // Reuse the direction register instead of clearing a vector.
                    const uchar256 direction = unpacked | 0x80;
                    const uchar256 base = v_u8_shuffle_b(table, direction, 0, direction);
                    const uchar256 encoded = v_u8_sel_eq_u8_b(
                        unpacked, 7, base, base + delta, SW_MASK_EQ_ZERO);
                    v_f8_st_tnsr(destination, output, *((minifloat256*)&encoded));
#else
                    const uchar256 unpacked = v_u8_ld_tnsr_b(
                        source, q16,
                        SW_UNPACK | SW_UNPCK_4_TO_8, (uchar256){0}, valid);
#if DSV41_N256_NORMAL_BF16
                    // This GUID is selected only for qualified UE8M0 codes
                    // 2..254. The tiny/NaN encodings retain exact_bf16 below.
                    const uchar256 directions = unpacked | 0x80;
                    const uchar256 bits = v_u8_shuffle_b(table, directions, 0, unpacked);
                    const bfloat256 base = convert_minifloat256_to_bfloat256(*((minifloat256*)&bits), SW_LINEAR);
                    const bfloat128 low = v_bf16_madd_b(base.v1, scale_low, negative_zero);
                    const bfloat128 high = v_bf16_madd_b(base.v2, scale_high, negative_zero);
                    v_bf16_st_tnsr(destination, output, low);
                    int5 upper = destination;
                    upper[0] += 128;
                    v_bf16_st_tnsr(upper, output, high);
#else
                    const ushort256 nibble = convert_uchar256_to_ushort256(unpacked, SW_LINEAR);
                    const ushort128 low = exact_bf16(nibble.v1, codes.v1);
                    const ushort128 high = exact_bf16(nibble.v2, codes.v2);
                    v_bf16_st_tnsr(destination, output, *((bfloat128*)&low));
                    int5 upper = destination;
                    upper[0] += 128;
                    v_bf16_st_tnsr(upper, output, *((bfloat128*)&high));
#endif
#endif
                    source[0] += 64;
                    destination[1] += 1;
                }
#endif
            }
        }
    }
}

#undef DSV41_N256_STORE
