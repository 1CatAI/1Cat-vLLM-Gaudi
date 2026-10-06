// SPDX-License-Identifier: Apache-2.0
// Pure packet layout/ID conversion; probability arithmetic stays unchanged.
void main(tensor packet,tensor values,tensor ids,tensor norms,tensor cuts,
          int columns,int ranks,int width) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int row=begin[1];row<end[1];++row) {
        for(int rank=begin[0];rank<end[0];++rank) {
            const int start=rank*(3+2*width);
            for(int col=0;col<width;col+=64) {
                v_f32_st_tnsr((int5){rank*width+col,row},values,
                    v_f32_ld_tnsr_b((int5){start+3+col,row},packet));
                const float64 token=v_f32_ld_tnsr_b((int5){start+3+width+col,row},packet);
                v_i32_st_tnsr((int5){rank*width+col,row},ids,convert_float64_to_int64(token,SW_RZ));
            }
            for(int field=0;field<3;++field) {
                const float value=s_f32_ld_g(gen_addr((int5){start+field,row},packet));
                s_f32_st_g(gen_addr((int5){rank*3+field,row},norms),value);
            }
            const float cut=s_f32_ld_g(gen_addr((int5){start+2+width,row},packet));
            s_f32_st_g(gen_addr((int5){rank,row},cuts),cut);
        }
    }
}
