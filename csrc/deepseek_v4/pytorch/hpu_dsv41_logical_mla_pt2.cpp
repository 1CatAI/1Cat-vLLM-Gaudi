// SPDX-License-Identifier: Apache-2.0
// Preserve selected MLA's QK/softmax/PV while absorbing its indirect layout producer.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "hpu_ops/op_backend.h"
#ifndef DSV41_LOGICAL_MLA_SCHEMA
#define DSV41_LOGICAL_MLA_SCHEMA "custom_deepseek_v41_logical_mla_gaudi2"
#endif
#ifndef DSV41_LOGICAL_MAIN_MIRROR
#define DSV41_LOGICAL_MAIN_MIRROR 0
#endif
#ifndef DSV41_LOGICAL_MLA_UNION
#define DSV41_LOGICAL_MLA_UNION 0
#endif
#ifndef DSV41_LOGICAL_MLA_DECODED
#define DSV41_LOGICAL_MLA_DECODED 0
#endif
#ifndef DSV41_LOGICAL_MLA_ORDERED
#define DSV41_LOGICAL_MLA_ORDERED 0
#endif
#ifndef DSV41_LOGICAL_MLA_STACKED_PV
#define DSV41_LOGICAL_MLA_STACKED_PV 0
#endif
#ifndef DSV41_LOGICAL_MLA_PAIR_PV
#define DSV41_LOGICAL_MLA_PAIR_PV 0
#endif
#ifndef DSV41_LOGICAL_MLA_EXP_PV
#define DSV41_LOGICAL_MLA_EXP_PV 0
#endif
#ifndef DSV41_LOGICAL_MLA_MERGED
#define DSV41_LOGICAL_MLA_MERGED 0
#endif
#ifndef DSV41_LOGICAL_MLA_VECTOR_GUID
#define DSV41_LOGICAL_MLA_VECTOR_GUID "custom_deepseek_v41_logical_mla_vector_gaudi2"
#endif
#ifndef DSV41_LOGICAL_MLA_BF16_KV_GUID
#define DSV41_LOGICAL_MLA_BF16_KV_GUID "custom_deepseek_v41_logical_mla_bf16_only_gaudi2"
#endif
namespace {
constexpr auto name = "custom_op::" DSV41_LOGICAL_MLA_SCHEMA;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto q = s.at(0).toTensor(), swa = s.at(1).toTensor(), main = s.at(2).toTensor();
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= 6 && q.size(1) >= 1 &&
                q.size(1) <= 64 && q.size(2) == 512, "Logical MLA requires Q [T,H,512], T<=6");
    for (int i = 0; i < 9; ++i) {
        const auto t = s.at(i).toTensor();
        const auto type = (i == 0 || (DSV41_LOGICAL_MLA_DECODED && i <= 2) || (DSV41_LOGICAL_MAIN_MIRROR && i == 2)) ? at::kBFloat16
            : i <= 2 ? at::kByte : (i == 6 || i == 7) ? at::kFloat : at::kInt;
        TORCH_CHECK(t.scalar_type() == type && t.device() == q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Logical MLA requires matching contiguous inference operands");
    }
    const auto selected = s.at(3).toTensor(), positions = s.at(4).toTensor(), pages = s.at(5).toTensor();
#if defined(DSV41_LOGICAL_MLA_MERGE_CACHE) && DSV41_LOGICAL_MLA_MERGE_CACHE
    TORCH_CHECK(q.size(0)>=2 && s.at(10).toBool(), "Shared sorted gather requires C2-C6 vector operands");
#endif
#if DSV41_LOGICAL_MLA_DECODED || DSV41_LOGICAL_MLA_ORDERED
    const auto completion = s.at(10).toTensor();
    TORCH_CHECK(completion.scalar_type()==at::kInt &&
        completion.numel()==q.size(0)*(DSV41_LOGICAL_MLA_ORDERED ? 36 : 1) &&
        completion.device()==q.device() && completion.is_contiguous(),
        "MLA requires the actual cache writer completion");
#endif
    TORCH_CHECK(swa.sizes() == at::IntArrayRef({256, DSV41_LOGICAL_MLA_DECODED ? 512 : 528}) &&
        main.dim() == 2 && main.size(1) == (DSV41_LOGICAL_MLA_DECODED || DSV41_LOGICAL_MAIN_MIRROR ? 512 : 288) &&
        main.size(0) > 0 && main.size(0) <= 0x7ffffdffLL &&
        selected.sizes() == at::IntArrayRef({q.size(0), 512}) && positions.sizes() == at::IntArrayRef({q.size(0)}) &&
        pages.dim() == 1 && pages.numel() > 0 && pages.numel() <= 8192 &&
        s.at(6).toTensor().sizes() == at::IntArrayRef({q.size(1)}) && s.at(7).toTensor().sizes() == at::IntArrayRef({1}) &&
        s.at(8).toTensor().sizes() == at::IntArrayRef({q.size(0)}) && (s.at(9).toInt() == 1 || s.at(9).toInt() == 2),
        "Invalid logical MLA cache, page, selection or scale contract");
#if DSV41_LOGICAL_MLA_MERGED
    TORCH_CHECK(q.size(0)>=2 && s.at(11).toInt()>=512 && s.at(11).toInt()<=65536 &&
                s.at(11).toInt()%128==0,"Merged MLA requires a bounded C2-C6 sorted selection");
#endif
#ifdef DSV41_LOGICAL_MLA_INVERSE_ROPE
    const auto phase = s.at(11).toTensor();
    TORCH_CHECK(phase.scalar_type()==at::kFloat && phase.device()==q.device() && phase.is_contiguous() &&
                phase.dim()==2 && phase.size(1)==64 && phase.size(0)>0 && phase.size(0)<=1048576,
                "Fused PV inverse RoPE requires the actual F32 phase table");
#endif
    return {{at::kBFloat16, q.sizes().vec()}};
}
class LogicalMla final : public habana::OpBackend {
public:
    LogicalMla(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string("dsv41_logical_mla"), type, {0}, {}, {}, false) {
        SetOutputMetaFn(meta);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta(s);
        const auto q = s.at(0).toTensor();
        const int64_t tokens = q.size(0), heads = q.size(1), width = 640;
        int ratio = s.at(9).toInt();
#ifdef DSV41_LOGICAL_MLA_SHARED_MME
        {
        // The indexer supplies sorted, unique logical selections. Shared
        // operands are consumed directly: no scatter into C copies of KV.
        TORCH_CHECK(tokens>=2,"Shared operands require C2-C6");
        const int64_t shared_width=256+tokens*512;
#if DSV41_LOGICAL_MLA_SHARED_MME >= 2
#if DSV41_LOGICAL_MLA_SHARED_MME == 3
        const int64_t buckets=257,bucket_width=256;
        const char* metadata_guid="custom_deepseek_v41_mla_stream_metadata_gaudi2";
        const char* decode_guid="custom_deepseek_v41_mla_stream_decode_gaudi2";
        const char* softmax_guid="custom_deepseek_v41_mla_stream_softmax_gaudi2";
#else
        const int64_t buckets=65,bucket_width=1024;
        const char* metadata_guid="custom_deepseek_v41_mla_compact_metadata_gaudi2";
        const char* decode_guid="custom_deepseek_v41_mla_compact_decode_gaudi2";
        const char* softmax_guid="custom_deepseek_v41_mla_compact_softmax_gaudi2";
#endif
        auto layout=BuildNode(this,graph,{metadata_guid,
            {syn_in(3),syn_in(4),syn_in(8)},
            {{{buckets,bucket_width},at::kInt},{{buckets,bucket_width},at::kInt},{{buckets},at::kInt}}});
        auto prefix=BuildNode(this,graph,{"custom_deepseek_v41_mla_compact_prefix_gaudi2",
            {layout[2].get()},{{{buckets+1},at::kInt}}});
        auto shared=BuildNode(this,graph,{decode_guid,
            {syn_in(1),syn_in(2),layout[0].get(),layout[1].get(),prefix[0].get(),syn_in(5)},
            {{{shared_width,512},at::kBFloat16},{{shared_width,512},at::kFloat},{{shared_width},at::kInt}},
            &ratio,sizeof(ratio)});
#else
        auto owners=BuildNode(this,graph,{"custom_deepseek_v41_mla_shared_owners_gaudi2",
            {syn_in(3),syn_in(4),syn_in(8)},
            {{{shared_width},at::kInt},{{shared_width},at::kInt}}});
        auto shared=BuildNode(this,graph,{"custom_deepseek_v41_mla_shared_decode_gaudi2",
            {syn_in(1),syn_in(2),owners[0].get(),owners[1].get(),syn_in(5)},
            {{{shared_width,512},at::kBFloat16},{{shared_width,512},at::kFloat}},&ratio,sizeof(ratio)});
#endif
        auto flat_q=BuildNode(this,graph,{"reshape",{syn_in(0)},{{{tokens*heads,512},at::kBFloat16}}});
        synGEMMParams shared_qk{false,true},shared_pv{false,false};
        auto scores=BuildNode(this,graph,{"gemm",{flat_q[0].get(),shared[0].get()},
            {{{tokens*heads,shared_width},at::kFloat}},&shared_qk,sizeof(shared_qk)});
#if DSV41_LOGICAL_MLA_SHARED_MME >= 2
        auto probability=BuildNode(this,graph,{softmax_guid,
            {scores[0].get(),shared[2].get(),syn_in(6),syn_in(7),prefix[0].get()},
            {{{tokens*heads,shared_width},at::kFloat}}});
#else
        auto probability=BuildNode(this,graph,{"custom_deepseek_v41_mla_shared_softmax_gaudi2",
            {scores[0].get(),owners[1].get(),syn_in(6),syn_in(7)},
            {{{tokens*heads,shared_width},at::kFloat}}});
#endif
        auto product=BuildNode(this,graph,{"gemm",{probability[0].get(),shared[1].get()},
            {{{tokens*heads,512},at::kFloat}},&shared_pv,sizeof(shared_pv)});
        auto rounded=BuildNode(this,graph,{"cast_f32_to_bf16",{product[0].get()},
            {{{tokens*heads,512},at::kBFloat16}}});
        syn_out(0)=std::move(BuildNode(this,graph,{"reshape",{rounded[0].get()},
            {{output[0].shape,at::kBFloat16,0}}})[0]);
        return;
        }
#endif
        auto make_kv = [&]() {
#if DSV41_LOGICAL_MLA_MERGED
        int params[2]={ratio,static_cast<int>(s.at(11).toInt())};
        return BuildNode(this,graph,{"custom_deepseek_v41_mla_merged_gaudi2",
            {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),syn_in(8)},
            {{{tokens,width,512},at::kBFloat16},{{tokens,width,512},at::kFloat},{{tokens,width},at::kFloat}},
            params,sizeof(params)});
#elif DSV41_LOGICAL_MLA_PAIR_PV || DSV41_LOGICAL_MLA_STACKED_PV || DSV41_LOGICAL_MLA_EXP_PV
        return BuildNode(this, graph, {DSV41_LOGICAL_MLA_BF16_KV_GUID,
            {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),syn_in(8)},
            {{{tokens,width,512},at::kBFloat16},{{1},at::kFloat},{{tokens,width},at::kFloat}},
            &ratio,sizeof(ratio)});
#elif DSV41_LOGICAL_MLA_UNION == 3
            auto owners = BuildNode(this, graph, {"custom_deepseek_v41_mla_hash_owners_gaudi2",
                {syn_in(3),syn_in(4),syn_in(8)}, {{{4352},at::kInt}}});
            auto shared = BuildNode(this, graph, {"custom_deepseek_v41_mla_hash_decode_gaudi2",
                {syn_in(1),syn_in(2),syn_in(3),owners[0].get(),syn_in(5)},
                {{{4352,512},at::kBFloat16}},&ratio,sizeof(ratio)});
            return BuildNode(this, graph, {"custom_deepseek_v41_mla_hash_gather_gaudi2",
                {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(8),syn_in(5),owners[0].get(),shared[0].get()},
                {{{tokens,width,512},at::kBFloat16},{{tokens,width,512},at::kFloat},{{tokens,width},at::kFloat}},
                &ratio,sizeof(ratio)});
#elif DSV41_LOGICAL_MLA_UNION == 2
            auto map = BuildNode(this, graph, {"memset", {}, {{{tokens,33024},at::kInt}}});
            auto completion = BuildNode(this, graph, {"custom_deepseek_v41_mla_slotmap_gaudi2",
                {syn_in(3),syn_in(4),syn_in(8),map.at(0).get()}, {{{tokens,640},at::kInt}}});
            return BuildNode(this, graph, {"custom_deepseek_v41_mla_slotmap_scatter_gaudi2",
                {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),map.at(0).get(),completion.at(0).get()},
                {{{tokens,width,512},at::kBFloat16},{{tokens,width,512},at::kFloat},{{tokens,width},at::kFloat}},
                &ratio,sizeof(ratio)});
#elif DSV41_LOGICAL_MLA_UNION == 1
            auto layout = BuildNode(this, graph, {"custom_deepseek_v41_mla_union_metadata_i32_gaudi2",
                {syn_in(3),syn_in(4),syn_in(8)},
                {{{tokens*512},at::kInt},{{257+tokens*512},at::kInt},{{tokens,640},at::kInt},{{4},at::kInt}}});
            return BuildNode(this, graph, {"custom_deepseek_v41_mla_union_scatter_gaudi2",
                {syn_in(1),syn_in(2),layout.at(0).get(),layout.at(1).get(),layout.at(2).get(),
                 layout.at(3).get(),syn_in(5)},
                {{{tokens,width,512},at::kBFloat16},{{tokens,width,512},at::kFloat},{{tokens,width},at::kFloat}},
                &ratio,sizeof(ratio)});
#elif DSV41_LOGICAL_MLA_DECODED || DSV41_LOGICAL_MLA_ORDERED
        return BuildNode(this, graph, {DSV41_LOGICAL_MLA_ORDERED ? "custom_deepseek_v41_logical_mla_write_ordered_gaudi2"
                                                              : "custom_deepseek_v41_logical_mla_decoded_gaudi2",
            {syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5),syn_in(8),syn_in(9)},
            {{{tokens,width,512},at::kBFloat16},{{tokens,width,512},at::kFloat},{{tokens,width},at::kFloat}},
            &ratio,sizeof(ratio)});
#else
        const char* gather = s.at(10).toBool() ? DSV41_LOGICAL_MLA_VECTOR_GUID
                                              : "custom_deepseek_v41_logical_mla_gather_gaudi2";
        return BuildNode(this, graph, {gather,
            {syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(5), syn_in(8)},
            {{{tokens, width, 512}, at::kBFloat16}, {{tokens, width, 512}, at::kFloat}, {{tokens, width}, at::kFloat}},
            &ratio, sizeof(ratio)});
#endif
        };
        auto kv = make_kv();
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = BuildNode(this, graph, {"batch_gemm", {syn_in(0), kv.at(0).get()},
            {{{tokens, heads, width}, at::kFloat}}, &qk, sizeof(qk)});
#if DSV41_LOGICAL_MLA_EXP_PV
        auto probabilities = BuildNode(this,graph,{"custom_deepseek_v41_mla_exp_bf16_gaudi2",
            {scores.at(0).get(),kv.at(2).get(),syn_in(6),syn_in(7)},
            {{{tokens,heads,width},at::kBFloat16},{{tokens,heads,1},at::kFloat}}});
        auto product = BuildNode(this,graph,{"batch_gemm",
            {probabilities.at(0).get(),kv.at(0).get()},
            {{{tokens,heads,512},at::kFloat}},&pv,sizeof(pv)});
        syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_exp_normalize_gaudi2",
            {product.at(0).get(),probabilities.at(1).get()},
            {{output.at(0).shape,at::kBFloat16,0}}}).at(0));
#elif DSV41_LOGICAL_MLA_STACKED_PV
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_stacked_softmax_gaudi2",
            {scores.at(0).get(),kv.at(2).get(),syn_in(6),syn_in(7)},
            {{{tokens,heads*2,width},at::kBFloat16}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(),kv.at(0).get()},
            {{{tokens,heads*2,512},at::kFloat}}, &pv,sizeof(pv)});
        syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_stacked_reduce_bf16_gaudi2",
            {product.at(0).get()},{{output.at(0).shape,at::kBFloat16,0}}}).at(0));
#elif DSV41_LOGICAL_MLA_PAIR_PV
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_pair_softmax_gaudi2",
            {scores.at(0).get(),kv.at(2).get(),syn_in(6),syn_in(7)},
            {{{tokens,heads,width},at::kBFloat16},{{tokens,heads,width},at::kBFloat16}}});
        auto high = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(),kv.at(0).get()},
            {{{tokens,heads,512},at::kFloat}}, &pv,sizeof(pv)});
        auto low = BuildNode(this, graph, {"batch_gemm", {probabilities.at(1).get(),kv.at(0).get()},
            {{{tokens,heads,512},at::kFloat}}, &pv,sizeof(pv)});
        syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_mla_pair_reduce_bf16_gaudi2",
            {high.at(0).get(),low.at(0).get()},{{output.at(0).shape,at::kBFloat16,0}}}).at(0));
#else
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_softmax_gaudi2",
            {scores.at(0).get(), kv.at(2).get(), syn_in(6), syn_in(7)}, {{{tokens, heads, width}, at::kFloat}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(), kv.at(1).get()},
            {{{tokens, heads, 512}, at::kFloat}}, &pv, sizeof(pv)});
#ifdef DSV41_LOGICAL_MLA_INVERSE_ROPE
        syn_out(0) = std::move(BuildNode(this, graph, {"custom_deepseek_v41_pv_rope_f32_gaudi2",
            {product.at(0).get(),syn_in(4),syn_in(9)},{{output.at(0).shape,at::kBFloat16,0}}}).at(0));
#else
        syn_out(0) = std::move(BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()},
            {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
#endif
#endif
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(name, "batch_gemm", [](const at::Stack& s) {
        const auto out = meta(s);
        return habana::PartialOutputMetaDataVector{{out.at(0).dtype, out.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId d, c10::ScalarType t) { return std::make_shared<LogicalMla>(d, t); });
    return true;
}();
template<bool Meta> at::Tensor execute(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& selected, const at::Tensor& positions, const at::Tensor& pages, const at::Tensor& sink,
    const at::Tensor& scale, const at::Tensor& lengths, int64_t ratio,
#if DSV41_LOGICAL_MLA_DECODED || DSV41_LOGICAL_MLA_ORDERED
    const at::Tensor& completion,
#endif
    bool vector
#ifdef DSV41_LOGICAL_MLA_INVERSE_ROPE
    , const at::Tensor& phase
#endif
#if DSV41_LOGICAL_MLA_MERGED
    , int64_t capacity
#endif
    ) {
#if DSV41_LOGICAL_MLA_DECODED || DSV41_LOGICAL_MLA_ORDERED
    const at::Stack stack{q,swa,main,selected,positions,pages,sink,scale,lengths,ratio,completion,vector};
#else
    const at::Stack stack{q, swa, main, selected, positions, pages, sink, scale, lengths, ratio, vector
#ifdef DSV41_LOGICAL_MLA_INVERSE_ROPE
        ,phase
#endif
#if DSV41_LOGICAL_MLA_MERGED
        ,capacity
#endif
    };
#endif
    const auto out = meta(stack);
    if (Meta) return at::empty(out.at(0).shape, q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
#ifdef DSV41_LOGICAL_MLA_INVERSE_ROPE
    m.def(DSV41_LOGICAL_MLA_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, bool vector, Tensor phase) -> Tensor");
#elif DSV41_LOGICAL_MLA_MERGED
    m.def(DSV41_LOGICAL_MLA_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, bool vector, int capacity) -> Tensor");
#elif DSV41_LOGICAL_MLA_DECODED || DSV41_LOGICAL_MLA_ORDERED
    m.def(DSV41_LOGICAL_MLA_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, Tensor completion, bool vector=False) -> Tensor");
#else
    m.def(DSV41_LOGICAL_MLA_SCHEMA "(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, bool vector=False) -> Tensor");
#endif
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl(DSV41_LOGICAL_MLA_SCHEMA, execute<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl(DSV41_LOGICAL_MLA_SCHEMA, execute<true>); }
