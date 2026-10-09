// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kQuant = "custom_deepseek_v41_woa_scale_dense_quant_gaudi2";
constexpr auto kQuantSchema = "custom_op::custom_deepseek_v41_woa_scale_dense_quant_gaudi2";
constexpr auto kRopeProjectionSchema = "custom_op::custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2";
constexpr auto kProjectionSchema = "custom_op::custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s, bool projection, bool rope = false) {
    TORCH_CHECK(s.size() == (rope ? 7 : projection ? 5 : 3), "WO handoff requires its complete operands");
    const auto x = s.at(0).toTensor();
    TORCH_CHECK(x.dim() == 3, "WO handoff requires a batched tensor");
    const auto groups = projection ? x.size(1) / (rope ? 8 : 1) : x.size(0);
    TORCH_CHECK((groups == 2 || groups == 4) && x.size(projection ? 0 : 1) >= 1 && x.size(projection ? 0 : 1) <= 6 &&
                x.sizes() == (projection ? (rope ? at::IntArrayRef({x.size(0),groups*8,512}) : at::IntArrayRef({x.size(0),groups,4096})) : at::IntArrayRef({groups,x.size(1),1024})) &&
                x.scalar_type() == (projection ? at::kBFloat16 : at::kFloat),
                "WO handoff requires the TP-parametric C1-C6 projection geometry");
    for (const auto& item : s) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.device() == x.device() && t.is_contiguous() && !t.requires_grad(),
                    "WO handoff requires matching contiguous inference operands");
    }
    if (rope) {
        const auto pos=s.at(5).toTensor(), phase=s.at(6).toTensor();
        TORCH_CHECK(pos.scalar_type()==at::kInt && pos.sizes()==at::IntArrayRef({x.size(0)}) &&
                    phase.scalar_type()==at::kFloat && phase.dim()==2 && phase.size(1)==64,
                    "WO inverse RoPE requires I32 position and packed F32 phase");
    }
    if (!projection) {
        TORCH_CHECK(s.at(1).toTensor().scalar_type() == at::kFloat &&
                    s.at(1).toTensor().sizes() == at::IntArrayRef({groups,1,1024}) &&
                    s.at(2).toTensor().scalar_type() == at::kFloat &&
                    s.at(2).toTensor().sizes() == at::IntArrayRef({groups,x.size(1),1}),
                    "WO handoff requires FP32 product and channel/activation scales");
        return {{at::ScalarType::Float8_e4m3fn,{x.size(1),groups*1024}}, {at::kFloat,{x.size(1),1}}};
    }
    TORCH_CHECK(s.at(1).toTensor().scalar_type() == at::ScalarType::Float8_e4m3fn &&
                s.at(1).toTensor().sizes() == at::IntArrayRef({groups,4096,1024}) &&
                s.at(2).toTensor().scalar_type() == at::kFloat &&
                s.at(2).toTensor().sizes() == at::IntArrayRef({groups,1,1024}) &&
                s.at(3).toTensor().scalar_type() == at::ScalarType::Float8_e4m3fn &&
                s.at(3).toTensor().sizes() == at::IntArrayRef({5120,groups*1024}) &&
                s.at(4).toTensor().scalar_type() == at::kFloat &&
                s.at(4).toTensor().sizes() == at::IntArrayRef({1,5120}),
                "WO handoff requires prepared FP8 WO weights and FP32 channel scales");
    return {{at::kBFloat16,{x.size(0),5120}}};
}
class Handoff final : public habana::OpBackend {
    bool projection_;
    bool rope_;
 public:
    Handoff(int device, c10::ScalarType dtype, bool projection, bool rope=false)
        : OpBackend(device, NO_TPC + std::string("dsv41_woa_dense_quant"), dtype,
                    projection ? std::vector<int>{0} : std::vector<int>{0,1}, {}, {}, false), projection_(projection), rope_(rope) {
        SetOutputMetaFn([projection,rope](const at::Stack& s){ return meta(s,projection,rope); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto outputs = meta(s,projection_,rope_);
        const auto tokens = s.at(0).toTensor().size(projection_ ? 0 : 1);
        const auto groups = s.at(0).toTensor().size(projection_ ? 1 : 0) / (rope_ ? 8 : 1);
        if (!projection_) {
            auto q = BuildNode(this,graph,{kQuant,{syn_in(0),syn_in(1),syn_in(2)},
                {{outputs[0].shape,outputs[0].dtype,0},{outputs[1].shape,outputs[1].dtype,1}}});
            syn_out(0)=std::move(q[0]); syn_out(1)=std::move(q[1]); return;
        }
        // Both boundaries are hand-fused TPC producers. Keep the two MMEs
        // between them; do not materialize inverse-RoPE or rounded BF16 rows.
        auto q = rope_
            ? BuildNode(this,graph,{"custom_deepseek_v41_woa_rope_quant_gaudi2",{syn_in(0),syn_in(5),syn_in(6)},
                {{{groups,tokens,4096},at::ScalarType::Float8_e4m3fn},{{groups,tokens,1},at::kFloat}}})
            : BuildNode(this,graph,{"custom_deepseek_v41_woa_quant_gaudi2",{syn_in(0)},
                {{{groups,tokens,4096},at::ScalarType::Float8_e4m3fn},{{groups,tokens,1},at::kFloat}}});
        synGEMMParams ap{false,false}, bp{false,true};
        auto a = BuildNode(this,graph,{"batch_gemm",{q[0].get(),syn_in(1)},
            {{{groups,tokens,1024},at::kFloat}},&ap,sizeof(ap)});
        auto bq = BuildNode(this,graph,{kQuant,{a[0].get(),syn_in(2),q[1].get()},
            {{{tokens,groups*1024},at::ScalarType::Float8_e4m3fn},{{tokens,1},at::kFloat}}});
        auto b = BuildNode(this,graph,{"gemm",{bq[0].get(),syn_in(3)},
            {{{tokens,5120},at::kFloat}},&bp,sizeof(bp)});
        syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_dense_scale_gaudi2",
            {b[0].get(),syn_in(4),bq[1].get()},{{{tokens,5120},at::kBFloat16,0}}})[0]);
    }
};
const bool registered = [] {
    for (bool projection : {false,true}) {
        const auto schema = projection ? kProjectionSchema : kQuantSchema;
        habana::custom_op::registerUserCustomOp(schema,kQuant,[projection](const at::Stack& s){
            habana::PartialOutputMetaDataVector r;
            for (const auto& o : meta(s,projection)) r.push_back({o.dtype,o.shape});
            return r;
        },nullptr);
        habana::KernelRegistry().add(schema,[projection](synDeviceId d,c10::ScalarType t){
            return std::make_shared<Handoff>(d,t,projection);
        });
    }
    habana::custom_op::registerUserCustomOp(kRopeProjectionSchema,kQuant,[](const at::Stack& s){
        const auto o=meta(s,true,true)[0];
        return habana::PartialOutputMetaDataVector{{o.dtype,o.shape}};
    },nullptr);
    habana::KernelRegistry().add(kRopeProjectionSchema,[](synDeviceId d,c10::ScalarType t){
        return std::make_shared<Handoff>(d,t,true,true);
    });
    return true;
}();
template<bool Meta> Pair quant(const at::Tensor& p,const at::Tensor& w,const at::Tensor& a) {
    const at::Stack s{p,w,a}; const auto out=meta(s,false);
    if constexpr (Meta) return {at::empty(out[0].shape,p.options().dtype(out[0].dtype)),
                               at::empty(out[1].shape,p.options().dtype(out[1].dtype))};
    TORCH_CHECK(registered && p.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kQuantSchema);
    const auto result=descriptor.execute(s);
    return {result[0],result[1]};
}
template<bool Meta> at::Tensor projection(const at::Tensor& x,const at::Tensor& wa,const at::Tensor& sa,
                                        const at::Tensor& wb,const at::Tensor& sb) {
    const at::Stack s{x,wa,sa,wb,sb}; const auto out=meta(s,true);
    if constexpr (Meta) return at::empty(out[0].shape,x.options());
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kProjectionSchema);
    return descriptor.execute(s)[0];
}
template<bool Meta> at::Tensor rope_projection(const at::Tensor& x,const at::Tensor& wa,const at::Tensor& sa,
                                             const at::Tensor& wb,const at::Tensor& sb,
                                             const at::Tensor& pos,const at::Tensor& phase) {
    const at::Stack s{x,wa,sa,wb,sb,pos,phase};const auto out=meta(s,true,true);
    if constexpr (Meta) return at::empty(out[0].shape,x.options());
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kRopeProjectionSchema);
    return descriptor.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2(Tensor input, Tensor woa, Tensor woa_scale, Tensor wob, Tensor wob_scale, Tensor positions, Tensor phase) -> Tensor");
    m.def("custom_deepseek_v41_woa_scale_dense_quant_gaudi2(Tensor product, Tensor weight_scale, Tensor activation_scale) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2(Tensor input, Tensor woa, Tensor woa_scale, Tensor wob, Tensor wob_scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2",rope_projection<false>);
    m.impl("custom_deepseek_v41_woa_scale_dense_quant_gaudi2",quant<false>);
    m.impl("custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2",projection<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2",rope_projection<true>);
    m.impl("custom_deepseek_v41_woa_scale_dense_quant_gaudi2",quant<true>);
    m.impl("custom_deepseek_v41_woa_wob_roundtrip_fp8_gaudi2",projection<true>);
}
