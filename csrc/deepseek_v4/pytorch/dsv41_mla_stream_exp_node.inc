// SPDX-License-Identifier: Apache-2.0
// Included in shared MLA AddNode. Same paging, QK and cache ownership;
// C1 BF16 exponentials use the single BF16 KV bank for PV.
synGEMMParams qk_stream{false,true},pv_stream{false,false};
if constexpr(Reuse) {
    auto swa=BuildNode(this,graph,{DSV41_MAIN_REUSE_GATHER_GUID,
        {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(7)},
        {{{tokens,128,512},at::kBFloat16},{{tokens,128},at::kFloat}}});
    auto main=subrange(graph,syn_in(2),{tokens,640,512},1,128,640,at::kBFloat16);
    auto main_mask=subrange(graph,syn_in(3),{tokens,640},0,128,640,at::kFloat);
    auto mask=join(graph,swa[1].get(),main_mask.get(),{tokens,640},at::kFloat);
    auto ss=BuildNode(this,graph,{"batch_gemm",{syn_in(0),swa[0].get()},
        {{{tokens,heads,128},at::kFloat}},&qk_stream,sizeof(qk_stream)});
    auto sm=BuildNode(this,graph,{"batch_gemm",{syn_in(0),main.get()},
        {{{tokens,heads,512},at::kFloat}},&qk_stream,sizeof(qk_stream)});
    auto probabilities=BuildNode(this,graph,{"custom_deepseek_v41_mla_stream_exp_reuse_gaudi2",
        {ss[0].get(),sm[0].get(),mask.get(),syn_in(5),syn_in(6)},
        {{{tokens,heads,128},at::kBFloat16},{{tokens,heads,512},at::kBFloat16},
         {{tokens,heads,1},at::kFloat}}});
    auto ps=BuildNode(this,graph,{"batch_gemm",{probabilities[0].get(),swa[0].get()},
        {{{tokens,heads,512},at::kFloat}},&pv_stream,sizeof(pv_stream)});
    auto pm=BuildNode(this,graph,{"batch_gemm",{probabilities[1].get(),main.get()},
        {{{tokens,heads,512},at::kFloat}},&pv_stream,sizeof(pv_stream)});
    syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_stream_finish_reuse_gaudi2",
        {ps[0].get(),pm[0].get(),probabilities[2].get(),syn_in(4),syn_in(8)},
        {{output[0].shape,at::kBFloat16,0}}})[0]);
} else {
    int ratio=s.at(9).toInt();
    auto bank=BuildNode(this,graph,{DSV41_MAIN_PUBLISH_GATHER_GUID,
        {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),syn_in(8)},
        {{{tokens,640,512},at::kBFloat16,1},{{tokens,640},at::kFloat,2}},&ratio,sizeof(ratio)});
    auto scores=BuildNode(this,graph,{"batch_gemm",{syn_in(0),bank[0].get()},
        {{{tokens,heads,640},at::kFloat}},&qk_stream,sizeof(qk_stream)});
    auto probabilities=BuildNode(this,graph,{"custom_deepseek_v41_mla_stream_exp_publish_gaudi2",
        {scores[0].get(),scores[0].get(),bank[1].get(),syn_in(6),syn_in(7)},
        {{{tokens,heads,640},at::kBFloat16},{{tokens,heads,1},at::kFloat}}});
    auto product=BuildNode(this,graph,{"batch_gemm",{probabilities[0].get(),bank[0].get()},
        {{{tokens,heads,512},at::kFloat}},&pv_stream,sizeof(pv_stream)});
    syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_stream_finish_publish_gaudi2",
        {product[0].get(),probabilities[1].get(),syn_in(4),syn_in(9)},
        {{output[0].shape,at::kBFloat16,0}}})[0]);
    syn_out(1)=std::move(bank[0]);syn_out(2)=std::move(bank[1]);
}
return;
