// SPDX-License-Identifier: Apache-2.0
void main(tensor scores,tensor positions,tensor candidates,tensor metadata,tensor output,
          int ratio,int reindex,int blocks_mode) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int visible=(s_i32_ld_g(gen_addr((int5){0},positions))+1)/ratio;
    const int width=blocks_mode?2048:512;
    const int64 lanes=(int64)V_LANE_ID_32;
    if(visible<=512) {
        for(int worker=begin[0];worker<end[0];++worker)
            for(int offset=worker*64;offset<width;offset+=24*64) {
                const int limit=blocks_mode?(visible+7)/8:visible;
                int64 ids=v_i32_sel_less_i32_b(lanes+offset,limit,lanes+offset,-1);
                v_i32_st_tnsr((int5){offset,0},output,ids);
            }
        return;
    }
    int count=reindex?16384:visible;
    if(blocks_mode)count=(count+7)/8;
    const unsigned threshold=s_u32_ld_g(gen_addr((int5){0},metadata));
    const int total_greater=s_i32_ld_g(gen_addr((int5){1},metadata));
    const int equal_budget=width-total_greater;
    for(int worker=begin[0];worker<end[0];++worker) {
        int greater=s_i32_ld_g(gen_addr((int5){2+worker*2},metadata));
        int equal=s_i32_ld_g(gen_addr((int5){3+worker*2},metadata));
        int written=greater+(equal<equal_budget?equal:equal_budget);
        const int first=count*worker/24,last=count*(worker+1)/24;
        for(int offset=first;offset<last;++offset) {
            const float score=s_f32_ld_g(gen_addr((int5){offset},scores));
            const unsigned bits=*((unsigned*)&score)>>16;
            const unsigned key=bits>32767 ? (~bits)&65535 : bits^32768;
            if(bits==0xff80)continue;
            const bool keep=key>threshold || (key==threshold && equal++<equal_budget);
            if(keep && written<width) {
                int logical=offset;
                if(reindex && !blocks_mode)logical=s_i32_ld_g(gen_addr((int5){offset/8,0},candidates))*8+offset%8;
                s_i32_st_g(gen_addr((int5){written++,0},output),logical);
            }
        }
        if(worker==23)for(;written<width;++written)s_i32_st_g(gen_addr((int5){written,0},output),-1);
    }
}
