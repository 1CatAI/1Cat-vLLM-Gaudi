// SPDX-License-Identifier: Apache-2.0
#define DSV41_DECODED_KV_WRITE 1
#include "deepseek_v41_fp4_pack.h"

// The packed row is scheduler/page-table addressed.  The decoded row is the
// bounded logical prefix row, so arbitrary scheduler block IDs do not expand
// the working set or leak one request's physical allocation into another.
void main(tensor main_cache, tensor index_cache, tensor main_value,
          tensor index_value, tensor packed_position, tensor decoded_position,
          tensor decoded, tensor completion) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    const int packed_row = s_i32_ld_g(gen_addr((int5){0}, packed_position));
    const int decoded_row = s_i32_ld_g(gen_addr((int5){0}, decoded_position));
    const bool valid = packed_row >= 0 && packed_row < get_dim_size(main_cache, 1) &&
                       packed_row < get_dim_size(index_cache, 1) &&
                       decoded_row >= 0 && decoded_row < get_dim_size(decoded, 1);
    for (int group = begin[0]; group < end[0]; ++group) {
        if (valid) {
            if (group < 4)
                fp4_pack_group(index_value, index_cache, group, 0,
                               packed_row, 32, decoded, 0, decoded_row);
            else
                fp4_pack_group(main_value, main_cache, group - 4, 0,
                               packed_row, 16, decoded, 1, decoded_row);
        }
        s_i32_st_g(gen_addr((int5){group,0,0,0,0}, completion),
                   valid ? decoded_row : -1);
    }
}
