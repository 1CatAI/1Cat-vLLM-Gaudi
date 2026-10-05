// SPDX-License-Identifier: Apache-2.0
#ifndef DSV41_SELECTED_KV_CODECS_H
#define DSV41_SELECTED_KV_CODECS_H
static inline bfloat128 selected_e4m3fn(ushort128 code) {
    const ushort128 exponent = (code >> 3) & 15;
    const ushort128 mantissa = code & 7;
    ushort128 tiny = (mantissa << 5) + 0x3b80;
    tiny = v_u16_sel_less_u16_b(mantissa, 4, (mantissa << 6) + 0x3b00, tiny);
    tiny = v_u16_sel_eq_u16_b(mantissa, 1, 0x3b00, tiny);
    tiny = v_u16_sel_eq_u16_b(mantissa, 0, 0, tiny);
    ushort128 bits = ((exponent + 120) << 7) | (mantissa << 4);
    bits = v_u16_sel_eq_u16_b(exponent, 0, tiny, bits);
    bits |= (code & 128) << 8;
    bits = v_u16_sel_eq_u16_b(code & 127, 0, 0, bits);
    bits = v_u16_sel_eq_u16_b(code & 127, 127, 0x7fff, bits);
    return *((bfloat128*)&bits);
}

static inline bfloat128 selected_ue8m0(ushort128 code) {
    ushort128 bits = code << 7;
    // The reference F32 multiply flushes the UE8M0 zero-code subnormal input.
    bits = v_u16_sel_eq_u16_b(code, 0, 0, bits);
    bits = v_u16_sel_eq_u16_b(code, 255, 0x7fff, bits);
    return *((bfloat128*)&bits);
}

static inline bfloat128 selected_fp4(ushort128 code) {
    const ushort128 magnitude = code & 7;
    ushort128 bits = (magnitude << 6) + 0x3f00;
    bits = v_u16_sel_eq_u16_b(magnitude, 1, 0x3f00, bits);
    bits = v_u16_sel_eq_u16_b(magnitude, 0, 0, bits);
    bits |= (code & 8) << 12;
    return *((bfloat128*)&bits);
}


#endif
