// SPDX-License-Identifier: Apache-2.0
void main(tensor scores,tensor positions,tensor candidates,tensor metadata,tensor output,
          int ratio,int reindex,int blocks_mode) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int request=begin[1];request<end[1];++request) {
    const int visible=(s_i32_ld_g(gen_addr((int5){request},positions))+1)/ratio;
    const int width=blocks_mode?2048:512;
    const int64 lanes=(int64)V_LANE_ID_32;
    const int direct_limit=blocks_mode?(visible+7)/8:visible;
    if(visible<=512 || (blocks_mode && direct_limit<=2048)) {
        for(int worker=begin[0];worker<end[0];++worker)
            for(int offset=worker*64;offset<width;offset+=24*64) {
                int64 ids=v_i32_sel_less_i32_b(lanes+offset,direct_limit,lanes+offset,-1);
                v_i32_st_tnsr((int5){offset,request},output,ids);
            }
        continue;
    }
    int valid_count=visible;
    int partition_count=visible;
    if(reindex) {
        const int visible_blocks=(visible+7)/8;
        valid_count=(visible_blocks<2048?visible_blocks:2048)*8;
        partition_count=16384;
    }
    if(blocks_mode) {
        valid_count=(valid_count+7)/8;
        partition_count=(partition_count+7)/8;
    }
    const unsigned threshold=s_u32_ld_g(gen_addr((int5){0,request},metadata));
    const int total_greater=s_i32_ld_g(gen_addr((int5){1,request},metadata));
    const int equal_budget=width-total_greater;
    for(int worker=begin[0];worker<end[0];++worker) {
        int greater=s_i32_ld_g(gen_addr((int5){2+worker*2,request},metadata));
        int equal=s_i32_ld_g(gen_addr((int5){3+worker*2,request},metadata));
        int written=greater+(equal<equal_budget?equal:equal_budget);
        const int first=partition_count*worker/24,last=partition_count*(worker+1)/24;
        const int scan_last=last<valid_count?last:valid_count;
        for(int offset=first;offset<scan_last;++offset) {
            const float score=s_f32_ld_g(gen_addr((int5){offset,request},scores));
            const unsigned bits=*((unsigned*)&score)>>16;
            const unsigned key=bits>32767 ? (~bits)&65535 : bits^32768;
            if(bits==0xff80)continue;
            const bool keep=key>threshold || (key==threshold && equal++<equal_budget);
            if(keep && written<width) {
                int logical=offset;
                if(reindex && !blocks_mode)logical=s_i32_ld_g(gen_addr((int5){offset/8,request},candidates))*8+offset%8;
                s_i32_st_g(gen_addr((int5){written++,request},output),logical);
            }
        }
        if(worker==23)for(;written<width;++written)s_i32_st_g(gen_addr((int5){written,request},output),-1);
    }
    }
}
