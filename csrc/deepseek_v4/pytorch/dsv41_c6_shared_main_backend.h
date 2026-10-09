// SPDX-License-Identifier: Apache-2.0
// Share selected main rows within one bounded layer group; SWA stays per layer.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "perf_lib_layer_params.h"
#include "hpu_ops/op_backend.h"
#ifndef DSV41_MAIN_PUBLISH_SCHEMA
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_main_publish_mla_gaudi2"
#endif
#ifndef DSV41_MAIN_REUSE_SCHEMA
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_main_reuse_mla_gaudi2"
#endif
#ifndef DSV41_MAIN_REUSE_GATHER_GUID
#define DSV41_MAIN_REUSE_GATHER_GUID "custom_deepseek_v41_main_reuse_gather_gaudi2"
#endif
#ifndef DSV41_MAIN_PUBLISH_GATHER_GUID
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_main_publish_gather_gaudi2"
#endif
#ifndef DSV41_MAIN_SINGLE_BANK
#define DSV41_MAIN_SINGLE_BANK 0
#endif
#ifndef DSV41_MAIN_SPLIT_REUSE
#define DSV41_MAIN_SPLIT_REUSE 0
#endif
#ifndef DSV41_MAIN_SWA_CACHED
#define DSV41_MAIN_SWA_CACHED 0
#endif
#ifndef DSV41_COHERENT_SWA_BF16_PV
#define DSV41_COHERENT_SWA_BF16_PV 0
#endif
#ifndef DSV41_COHERENT_SWA
#define DSV41_COHERENT_SWA 0
#endif
#ifndef DSV41_MAIN_REUSE_BF16_PAIR
#define DSV41_MAIN_REUSE_BF16_PAIR 0
#endif
#ifndef DSV41_MAIN_FP16_PV
#define DSV41_MAIN_FP16_PV 0
#endif
#ifndef DSV41_MAIN_FP16_DIRECT
#define DSV41_MAIN_FP16_DIRECT 0
#endif
namespace {
constexpr auto pv_type = DSV41_MAIN_FP16_DIRECT ? at::kHalf : at::kFloat;
constexpr auto softmax_guid = DSV41_MAIN_FP16_DIRECT ? "custom_deepseek_v41_mla_fp16_softmax_gaudi2"
                                                   : "custom_deepseek_v41_selected_mla_softmax_gaudi2";
constexpr auto publish_name = "custom_op::" DSV41_MAIN_PUBLISH_SCHEMA;
constexpr auto reuse_name = "custom_op::" DSV41_MAIN_REUSE_SCHEMA;
#ifndef DSV41_MAIN_STREAM_EXP_PV
#define DSV41_MAIN_STREAM_EXP_PV 0
#endif
template<bool Reuse> habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto q = s.at(0).toTensor(), swa = s.at(1).toTensor(), main = s.at(2).toTensor();
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
    const auto phase = s.at(Reuse ? 9 : 10).toTensor();
    TORCH_CHECK(q.size(0)>=2 && q.size(0)<=6 && q.size(1)%8==0 &&
        phase.dim()==2 && phase.size(1)==64 && phase.size(0)>0 && phase.scalar_type()==at::kFloat &&
        phase.device()==q.device() && phase.is_contiguous() && !phase.requires_grad(),
        "Interleaved MLA requires whole head groups and the shared FP32 phase table");
#endif
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= 6 && q.size(1) >= 1 && q.size(1) <= 64 && q.size(2) == 512,
                "Shared main MLA requires bounded Q [C1-C6,H,512]");
#if DSV41_MAIN_FP16_PV
    TORCH_CHECK(q.size(0) >= 2, "Experimental FP16 PV is confined to C2-C6");
#endif
#if DSV41_COHERENT_SWA
    TORCH_CHECK(q.size(0)>=2, "Coherent SWA requires C2-C6");
#endif
    for (int i = 0; i < (Reuse ? ((DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK) ? 9 : 8) : 9); ++i) {
        const auto t = s.at(i).toTensor();
        const auto type = Reuse ? (i == 0 || i == 2 ? at::kBFloat16 :
                                  i == 1 ? (DSV41_MAIN_SWA_CACHED ? at::kBFloat16 : at::kByte) :
                                  i == 8 ? pv_type : i == 3 || i == 5 || i == 6 ? at::kFloat : at::kInt) :
                                (i == 0 || (DSV41_MAIN_SWA_CACHED && i == 1) ? at::kBFloat16 : i <= 2 ? at::kByte :
                                  i == 6 || i == 7 ? at::kFloat : at::kInt);
        TORCH_CHECK(t.scalar_type() == type && t.device() == q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Shared main MLA requires matching contiguous inference operands");
    }
    TORCH_CHECK(swa.sizes() == at::IntArrayRef({256, DSV41_MAIN_SWA_CACHED ? 512 : 528}), "Invalid SWA ring");
    TORCH_CHECK(s.at(4).toTensor().sizes() == at::IntArrayRef({q.size(0)}) &&
                s.at(Reuse ? 5 : 6).toTensor().sizes() == at::IntArrayRef({q.size(1)}) &&
                s.at(Reuse ? 6 : 7).toTensor().sizes() == at::IntArrayRef({1}) &&
                s.at(Reuse ? 7 : 8).toTensor().sizes() == at::IntArrayRef({q.size(0)}),
                "Invalid position, sink, scale or length");
#if DSV41_MAIN_SWA_CACHED
    const auto completion=s.at(Reuse ? 9 : 10).toTensor();
    TORCH_CHECK(completion.scalar_type()==at::kInt && completion.device()==q.device() && completion.is_contiguous() &&
                completion.sizes()==at::IntArrayRef({q.size(0),16}),"SWA cache requires the complete row writer signal");
#endif
#if DSV41_MAIN_STREAM_EXP_PV
    const auto phase=s.at(Reuse?8:10).toTensor();
    TORCH_CHECK(q.size(0)>=2 && phase.dim()==2 && phase.size(0)>0 && phase.size(1)==64 &&
        phase.scalar_type()==at::kFloat && phase.device()==q.device() && phase.is_contiguous() &&
        !phase.requires_grad(),"Streaming C2-C6 PV requires the shared FP32 RoPE table");
#endif
    if constexpr (Reuse) {
        TORCH_CHECK(main.sizes() == at::IntArrayRef({q.size(0), 640, 512}) &&
                    s.at(3).toTensor().sizes() == at::IntArrayRef({q.size(0), 640}), "Invalid shared main rows or mask");
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
        TORCH_CHECK(s.at(8).toTensor().sizes()==main.sizes(),"Invalid shared FP32 main values");
#endif
        return {{at::kBFloat16,
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
                 {q.size(1)/8,q.size(0),4096}
#else
                 q.sizes().vec()
#endif
                }};
    } else {
        const auto pages = s.at(5).toTensor();
        TORCH_CHECK(main.dim() == 2 && main.size(1) == 288 && main.size(0) > 0 && main.size(0) <= 0x7ffffdffLL &&
                    s.at(3).toTensor().sizes() == at::IntArrayRef({q.size(0), 512}) && pages.dim() == 1 &&
                    pages.numel() > 0 && pages.numel() <= 8192 && (s.at(9).toInt() == 1 || s.at(9).toInt() == 2),
                    "Invalid packed main, selection, page table or ratio");
        habana::OutputMetaDataVector result{{at::kBFloat16,
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
                 {q.size(1)/8,q.size(0),4096}
#else
                 q.sizes().vec()
#endif
                },
            {at::kBFloat16, {q.size(0), 640, 512}}, {at::kFloat, {q.size(0), 640}}};
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
        result.push_back({pv_type,{q.size(0),640,512}});
#endif
        return result;
    }
}
template<bool Reuse> class SharedMainMla final : public habana::OpBackend {
#include "dsv41_c6_flat_qk_member.h"
    std::vector<synapse_helpers::tensor> pv_product(synapse_helpers::graph& graph,
        synTensor probability, synTensor values, int64_t tokens, int64_t heads, int64_t width) {
        synGEMMParams parameters{false, false};
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
        auto p=ReshapeHelper(graph,probability,{tokens,heads/8,8,width},at::kFloat);
        auto v=ReshapeHelper(graph,values,{tokens,1,width,512},at::kFloat);
        synGEMMParams interleaved_pv{true,true};
        auto product=BuildNode(this,graph,{"batch_gemm",{v.get(),p.get()},
            {{{tokens,heads/8,512,8},at::kFloat}},&interleaved_pv,sizeof(interleaved_pv)});
        std::vector<synapse_helpers::tensor> result;
        result.push_back(ReshapeHelper(graph,product[0].get(),{tokens,heads,512},at::kFloat));
        return result;
#elif DSV41_MAIN_FP16_DIRECT
        return BuildNode(this, graph, {"batch_gemm", {probability, values},
            {{{tokens, heads, 512}, at::kFloat}}, &parameters, sizeof(parameters)});
#elif DSV41_MAIN_FP16_PV
        // Preserve QK, softmax, public cache planes and FP32 accumulation.
        // These conversions are part of the complete consumer, never untimed.
        auto p = BuildNode(this, graph, {"cast_f32_to_f16", {probability},
            {{{tokens, heads, width}, at::kHalf}}});
        auto v = BuildNode(this, graph, {"cast_f32_to_f16", {values},
            {{{tokens, width, 512}, at::kHalf}}});
        return BuildNode(this, graph, {"batch_gemm", {p[0].get(), v[0].get()},
            {{{tokens, heads, 512}, at::kFloat}}, &parameters, sizeof(parameters)});
#else
        return BuildNode(this, graph, {"batch_gemm", {probability, values},
            {{{tokens, heads, 512}, at::kFloat}}, &parameters, sizeof(parameters)});
#endif
    }
    synapse_helpers::tensor finish_pv(synapse_helpers::graph& graph, synTensor product,
        const at::Stack& s, int64_t tokens, int64_t heads) {
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
        auto flat=ReshapeHelper(graph,product,{tokens,heads/8,4096},at::kFloat);
        auto result=BuildNode(this,graph,{"custom_deepseek_v41_interleaved_pv_bf16_gaudi2",
            {flat.get(),syn_in(4),syn_in(9)},{{{heads/8,tokens,4096},at::kBFloat16,0}}});
        return std::move(result[0]);
#else
        (void)s;
        return std::move(BuildNode(this,graph,{"cast_f32_to_bf16",{product},
            {{{tokens,heads,512},at::kBFloat16,0}}})[0]);
#endif
    }
    synapse_helpers::tensor subrange(synapse_helpers::graph& graph,synTensor input,
        std::vector<int64_t> shape,unsigned axis,int first,int last,at::ScalarType type) {
        synSliceParams params{};
        for(unsigned i=0;i<sizeof(params.axes)/sizeof(params.axes[0]);++i){params.axes[i]=i;params.steps[i]=1;}
        for(unsigned i=0;i<shape.size();++i)params.ends[i]=shape[shape.size()-1-i];
        params.starts[axis]=first;params.ends[axis]=last;shape[shape.size()-1-axis]=last-first;
        return std::move(BuildNode(this,graph,{"slice",{input},{{shape,type}},&params,sizeof(params)})[0]);
    }
    synapse_helpers::tensor join(synapse_helpers::graph& graph,synTensor first,synTensor second,
        const std::vector<int64_t>& shape,at::ScalarType type) {
        synConcatenateParams params{};params.axis=0;
        return std::move(BuildNode(this,graph,{"concat",{first,second},{{shape,type}},&params,sizeof(params)})[0]);
    }
public:
    SharedMainMla(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string(
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
                    Reuse ? "dsv41_interleaved_reuse" : "dsv41_interleaved_publish"
#else
                    Reuse ? "dsv41_main_reuse" : "dsv41_main_publish"
#endif
                    ),
                    type, {0}, {}, {}, false) { SetOutputMetaFn(meta<Reuse>); }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta<Reuse>(s);
        const int64_t tokens = s.at(0).toTensor().size(0), heads = s.at(0).toTensor().size(1);
        synTensor query=syn_in(0);
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
        auto flat_query=ReshapeHelper(graph,query,{tokens,heads/8,4096},at::kBFloat16);
        auto roped=BuildNode(this,graph,{"custom_deepseek_v41_interleaved_query_rope_gaudi2",
            {flat_query.get(),syn_in(4),syn_in(9)},{{{tokens,heads/8,4096},at::kBFloat16}}});
        auto query_shape=ReshapeHelper(graph,roped[0].get(),{tokens,heads,512},at::kBFloat16);
        query=query_shape.get();
#endif
#if DSV41_MAIN_STREAM_EXP_PV
#include "dsv41_mla_stream_exp_node.inc"
#endif
#if DSV41_MAIN_SPLIT_REUSE
        if constexpr(Reuse) {
#if DSV41_COHERENT_SWA
            {
            TORCH_CHECK(tokens>=2 && tokens<=6, "Coherent SWA requires consecutive C2-C6 positions");
            constexpr bool bf16_pv=DSV41_COHERENT_SWA_BF16_PV;
            const int64_t swa_heads=heads*(bf16_pv?2:1);
            habana::NodeAttr decode;
            decode.guid=bf16_pv?"custom_deepseek_v41_coherent_swa_keys_gaudi2":
                                "custom_deepseek_v41_coherent_swa_decode_gaudi2";
            decode.inputs={syn_in(1),syn_in(4)};
            decode.output_attrs={{{192,512},at::kBFloat16}};
            if(!bf16_pv)decode.output_attrs.push_back({{192,512},at::kFloat});
            auto decoded=BuildNode(this,graph,std::move(decode));
            auto queries=ReshapeHelper(graph,syn_in(0),{tokens*heads,512},at::kBFloat16);
            synGEMMParams qk{false,true},pv{false,false};
            auto qs=BuildNode(this,graph,{"gemm",{queries.get(),decoded[0].get()},
                {{{tokens*heads,192},at::kFloat}},&qk,sizeof(qk)});
            auto scores_swa=ReshapeHelper(graph,qs[0].get(),{tokens,heads,192},at::kFloat);
            auto main=subrange(graph,syn_in(2),{tokens,640,512},1,128,640,at::kBFloat16);
            auto values=subrange(graph,syn_in(8),{tokens,640,512},1,128,640,pv_type);
            auto qm=BuildNode(this,graph,{"batch_gemm",{syn_in(0),main.get()},
                {{{tokens,heads,512},at::kFloat}},&qk,sizeof(qk)});
            auto probabilities=BuildNode(this,graph,{bf16_pv?
                "custom_deepseek_v41_coherent_swa_pair_softmax_gaudi2":
                "custom_deepseek_v41_coherent_swa_softmax_gaudi2",
                {scores_swa.get(),qm[0].get(),syn_in(3),syn_in(5),syn_in(6),syn_in(4),syn_in(7)},
                {{{tokens,swa_heads,192},bf16_pv?at::kBFloat16:at::kFloat},{{tokens,heads,512},at::kFloat}}});
            auto ps=ReshapeHelper(graph,probabilities[0].get(),{tokens*swa_heads,192},
                                  bf16_pv?at::kBFloat16:at::kFloat);
            auto vs=BuildNode(this,graph,{"gemm",{ps.get(),decoded[bf16_pv?0:1].get()},
                {{{tokens*swa_heads,512},at::kFloat}},&pv,sizeof(pv)});
            auto vs_rows=ReshapeHelper(graph,vs[0].get(),{tokens,swa_heads,512},at::kFloat);
            auto vm=BuildNode(this,graph,{"batch_gemm",{probabilities[1].get(),values.get()},
                {{{tokens,heads,512},at::kFloat}},&pv,sizeof(pv)});
            if(bf16_pv) {
                syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_coherent_swa_finish_gaudi2",
                    {vs_rows.get(),vm[0].get()},{{output[0].shape,at::kBFloat16,0}}})[0]);
                return;
            }
            auto sum=BuildNode(this,graph,{"add_fwd_f32",{vs_rows.get(),vm[0].get()},
                {{{tokens,heads,512},at::kFloat}}});
            syn_out(0)=std::move(BuildNode(this,graph,{"cast_f32_to_bf16",{sum[0].get()},
                {{output[0].shape,at::kBFloat16,0}}})[0]);
            return;
            }
#endif
#if DSV41_MAIN_REUSE_BF16_PAIR
            auto swa=BuildNode(this,graph,{DSV41_MAIN_REUSE_GATHER_GUID,
                {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(7)},
                {{{tokens,128,512},at::kBFloat16},{{tokens,128},at::kFloat}}});
            auto main=subrange(graph,syn_in(2),{tokens,640,512},1,128,640,at::kBFloat16);
            auto main_mask=subrange(graph,syn_in(3),{tokens,640},0,128,640,at::kFloat);
            synGEMMParams qk{false,true},pv{false,false};
            auto ss=BuildNode(this,graph,{"batch_gemm",{syn_in(0),swa[0].get()},
                {{{tokens,heads,128},at::kFloat}},&qk,sizeof(qk)});
            auto sm=BuildNode(this,graph,{"batch_gemm",{syn_in(0),main.get()},
                {{{tokens,heads,512},at::kFloat}},&qk,sizeof(qk)});
            auto scores=join(graph,ss[0].get(),sm[0].get(),{tokens,heads,640},at::kFloat);
            auto mask=join(graph,swa[1].get(),main_mask.get(),{tokens,640},at::kFloat);
            auto probability=BuildNode(this,graph,{"custom_deepseek_v41_mla_adjacent_softmax_gaudi2",
                {scores.get(),mask.get(),syn_in(5),syn_in(6)},{{{tokens,heads*2,640},at::kBFloat16}}});
            auto ps=subrange(graph,probability[0].get(),{tokens,heads*2,640},0,0,128,at::kBFloat16);
            auto pm=subrange(graph,probability[0].get(),{tokens,heads*2,640},0,128,640,at::kBFloat16);
            auto vs=BuildNode(this,graph,{"batch_gemm",{ps.get(),swa[0].get()},
                {{{tokens,heads*2,512},at::kFloat}},&pv,sizeof(pv)});
            auto vm=BuildNode(this,graph,{"batch_gemm",{pm.get(),main.get()},
                {{{tokens,heads*2,512},at::kFloat}},&pv,sizeof(pv)});
            auto sum=BuildNode(this,graph,{"add_fwd_f32",{vs[0].get(),vm[0].get()},
                {{{tokens,heads*2,512},at::kFloat}}});
            syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_adjacent_finish_gaudi2",
                {sum[0].get()},{{output[0].shape,at::kBFloat16,0}}})[0]);
            return;
#else
            auto swa=BuildNode(this,graph,{DSV41_MAIN_REUSE_GATHER_GUID,
                {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(7)
#if DSV41_MAIN_SWA_CACHED
                 ,syn_in(9)
#endif
                },
                {{{tokens,128,512},at::kBFloat16},{{tokens,128,512},pv_type},{{tokens,128},at::kFloat}}});
            auto main=subrange(graph,syn_in(2),{tokens,640,512},1,128,640,at::kBFloat16);
            auto values=subrange(graph,syn_in(8),{tokens,640,512},1,128,640,pv_type);
            auto main_mask=subrange(graph,syn_in(3),{tokens,640},0,128,640,at::kFloat);
            synGEMMParams qk{false,true},pv{false,false};
            auto score_swa=build_c6_qk(graph,query,swa[0].get(),tokens,heads,128);
            auto score_main=build_c6_qk(graph,query,main.get(),tokens,heads,512);
            auto mask=join(graph,swa[2].get(),main_mask.get(),{tokens,640},at::kFloat);
#if defined(DSV41_C6_FLAT_QK_DIRECT) && DSV41_C6_FLAT_QK_DIRECT
            auto probability=BuildNode(this,graph,{"custom_deepseek_v41_qk_flat_reuse_softmax_gaudi2",
                {score_swa[0].get(),score_main[0].get(),mask.get(),syn_in(5),syn_in(6)},
                {{{tokens,heads,640},at::kFloat}}});
#else
            auto scores=join(graph,score_swa[0].get(),score_main[0].get(),{tokens,heads,640},at::kFloat);
            auto probability=BuildNode(this,graph,{softmax_guid,
                {scores.get(),mask.get(),syn_in(5),syn_in(6)},{{{tokens,heads,640},pv_type}}});
#endif
            auto ps=subrange(graph,probability[0].get(),{tokens,heads,640},0,0,128,pv_type);
            auto pm=subrange(graph,probability[0].get(),{tokens,heads,640},0,128,640,pv_type);
            auto vs=pv_product(graph,ps.get(),swa[1].get(),tokens,heads,128);
            auto vm=pv_product(graph,pm.get(),values.get(),tokens,heads,512);
            auto sum=BuildNode(this,graph,{"add_fwd_f32",{vs[0].get(),vm[0].get()},
                {{{tokens,heads,512},at::kFloat}}});
            syn_out(0)=finish_pv(graph,sum[0].get(),s,tokens,heads);
            return;
#endif
        }
#endif
#if DSV41_MAIN_SINGLE_BANK
        {
        // Publisher produces one public rounded bank; no FP32 or duplicate BF16 cache.
        int ratio=s.at(9).toInt();
        auto bank=BuildNode(this,graph,{DSV41_MAIN_PUBLISH_GATHER_GUID,
            {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),syn_in(8)},
            {{{tokens,640,512},at::kBFloat16,1},{{tokens,640},at::kFloat,2}},&ratio,sizeof(ratio)});
        synGEMMParams qk{false,true},pv{false,false};
        auto scores=BuildNode(this,graph,{"batch_gemm",{syn_in(0),bank[0].get()},
            {{{tokens,heads,640},at::kFloat}},&qk,sizeof(qk)});
        auto probability=BuildNode(this,graph,{"custom_deepseek_v41_mla_adjacent_softmax_gaudi2",
            {scores[0].get(),bank[1].get(),syn_in(6),syn_in(7)},
            {{{tokens,heads*2,640},at::kBFloat16}}});
        auto product=BuildNode(this,graph,{"batch_gemm",{probability[0].get(),bank[0].get()},
            {{{tokens,heads*2,512},at::kFloat}},&pv,sizeof(pv)});
        syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_adjacent_finish_gaudi2",
            {product[0].get()},{{output[0].shape,at::kBFloat16,0}}})[0]);
        syn_out(1)=std::move(bank[0]);syn_out(2)=std::move(bank[1]);
        return;
        }
#endif
        auto kv = [&] {
            if constexpr (Reuse) {
                return BuildNode(this, graph, {DSV41_MAIN_REUSE_GATHER_GUID,
                    {syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(7)},
                    {{{tokens, 640, 512}, at::kBFloat16}, {{tokens, 640, 512}, at::kFloat}, {{tokens, 640}, at::kFloat}}});
            } else {
                int ratio = s.at(9).toInt();
                return BuildNode(this, graph, {DSV41_MAIN_PUBLISH_GATHER_GUID,
                    {syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(5), syn_in(8)
#if DSV41_MAIN_SWA_CACHED
                     ,syn_in(9)
#endif
                    },
                    {{{tokens, 640, 512}, at::kBFloat16},
                     {{tokens, 640, 512}, pv_type, DSV41_MAIN_SPLIT_REUSE ? 3 : -1},
                     {{tokens, 640}, at::kFloat, 2}, {{tokens, 640, 512}, at::kBFloat16, 1}}, &ratio, sizeof(ratio)});
            }
        }();
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = build_c6_qk(graph,query,kv.at(0).get(),tokens,heads,640);
#if defined(DSV41_C6_FLAT_QK_DIRECT) && DSV41_C6_FLAT_QK_DIRECT
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_qk_flat_publish_softmax_gaudi2",
            {scores.at(0).get(), scores.at(0).get(), kv.at(2).get(), syn_in(Reuse ? 5 : 6), syn_in(Reuse ? 6 : 7)},
            {{{tokens, heads, 640}, at::kFloat}}});
#else
        auto probabilities = BuildNode(this, graph, {softmax_guid,
            {scores.at(0).get(), kv.at(2).get(), syn_in(Reuse ? 5 : 6), syn_in(Reuse ? 6 : 7)},
            {{{tokens, heads, 640}, pv_type}}});
#endif
        auto product = pv_product(graph, probabilities.at(0).get(), kv.at(1).get(), tokens, heads, 640);
        syn_out(0) = finish_pv(graph,product[0].get(),s,tokens,heads);
        if constexpr (!Reuse) {
            syn_out(1) = std::move(kv.at(3));
            syn_out(2) = std::move(kv.at(2));
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
            syn_out(3) = std::move(kv.at(1));
#endif
        }
    }
};
template<bool Reuse> bool register_op() {
    const auto name = Reuse ? reuse_name : publish_name;
    habana::custom_op::registerUserCustomOp(name, "batch_gemm", [](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for (const auto& item : meta<Reuse>(s)) out.push_back({item.dtype, item.shape});
        return out;
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId d, c10::ScalarType t) {
        return std::make_shared<SharedMainMla<Reuse>>(d, t);
    });
    return true;
}
const bool registered = register_op<false>() && register_op<true>();
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
using PublishResult=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
#else
using PublishResult=std::tuple<at::Tensor,at::Tensor,at::Tensor>;
#endif
template<bool Meta> PublishResult publish(
    const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main, const at::Tensor& selected,
    const at::Tensor& positions, const at::Tensor& pages, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths, int64_t ratio
#if DSV41_MAIN_STREAM_EXP_PV || (defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT)
    ,const at::Tensor& phase
#endif
#if DSV41_MAIN_SWA_CACHED
    ,const at::Tensor& completion
#endif
    ) {
    at::Stack stack{q, swa, main, selected, positions, pages, sink, scale, lengths, ratio};
#if DSV41_MAIN_STREAM_EXP_PV || (defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT)
    stack.emplace_back(phase);
#endif
#if DSV41_MAIN_SWA_CACHED
    stack.emplace_back(completion);
#endif
    const auto out = meta<false>(stack);
    std::vector<at::Tensor> result;
    if(Meta){for(const auto& item:out)result.push_back(at::empty(item.shape,q.options().dtype(item.dtype)));}
    else {
        TORCH_CHECK(registered && q.device().type()==at::kHPU);
        auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(publish_name);
        result=descriptor.execute(stack);
    }
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
    return {result[0],result[1],result[2],result[3]};
#else
    return {result[0],result[1],result[2]};
#endif
}
template<bool Meta> at::Tensor reuse(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& mask, const at::Tensor& positions, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths
#if DSV41_MAIN_STREAM_EXP_PV
    ,const at::Tensor& phase
#endif
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
    ,const at::Tensor& values
#endif
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
    ,const at::Tensor& phase
#endif
#if DSV41_MAIN_SWA_CACHED
    ,const at::Tensor& completion
#endif
    ) {
    at::Stack stack{q, swa, main, mask, positions, sink, scale, lengths};
#if DSV41_MAIN_STREAM_EXP_PV
    stack.emplace_back(phase);
#endif
#if DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
    stack.emplace_back(values);
#endif
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
    stack.emplace_back(phase);
#endif
#if DSV41_MAIN_SWA_CACHED
    stack.emplace_back(completion);
#endif
    const auto out = meta<true>(stack);
    if (Meta) return at::empty(out[0].shape, q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(reuse_name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
#if defined(DSV41_INTERLEAVED_LAYOUT) && DSV41_INTERLEAVED_LAYOUT
    m.def(DSV41_MAIN_PUBLISH_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, Tensor phase) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(DSV41_MAIN_REUSE_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths, Tensor values, Tensor phase) -> Tensor");
#elif DSV41_MAIN_STREAM_EXP_PV
    m.def(DSV41_MAIN_PUBLISH_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, Tensor phase) -> (Tensor, Tensor, Tensor)");
    m.def(DSV41_MAIN_REUSE_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths, Tensor phase) -> Tensor");
#elif DSV41_MAIN_SWA_CACHED
    m.def(DSV41_MAIN_PUBLISH_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, Tensor completion) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(DSV41_MAIN_REUSE_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths, Tensor values, Tensor completion) -> Tensor");
#elif DSV41_MAIN_SPLIT_REUSE && !DSV41_MAIN_SINGLE_BANK
    m.def(DSV41_MAIN_PUBLISH_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(DSV41_MAIN_REUSE_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths, Tensor values) -> Tensor");
#else
    m.def(DSV41_MAIN_PUBLISH_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio) -> (Tensor, Tensor, Tensor)");
    m.def(DSV41_MAIN_REUSE_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
#endif
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl(DSV41_MAIN_PUBLISH_SCHEMA, publish<false>);
    m.impl(DSV41_MAIN_REUSE_SCHEMA, reuse<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl(DSV41_MAIN_PUBLISH_SCHEMA, publish<true>);
    m.impl(DSV41_MAIN_REUSE_SCHEMA, reuse<true>);
}
