// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto reference_schema = "custom_op::custom_deepseek_v41_silu_quant_reference_gaudi2";
constexpr auto shared_schema = "custom_op::custom_deepseek_v41_shared_silu_quant_gaudi2";
constexpr auto shared_guid = "custom_deepseek_v41_shared_silu_quant_gaudi2";
constexpr auto tiled_schema = "custom_op::custom_deepseek_v41_silu_quant_feature_gaudi2";
constexpr auto reference_guid = "custom_deepseek_v41_expert_n256_silu_quant_gaudi2";
using Output = std::tuple<at::Tensor, at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s, bool shared=false) {
    const auto p=s.at(0).toTensor(), ids=s.at(1).toTensor(), sx=s.at(2).toTensor();
    const auto ch=s.at(3).toTensor(), route=s.at(4).toTensor();
    TORCH_CHECK(p.dim()==3 && p.size(0)>=1 && p.size(0)<=16384 && p.size(1)==1 && (shared ? (p.size(2)>=256 && p.size(2)<=5120 && p.size(2)%256==0) : (p.size(0)<=36 && p.size(2)==1280)),
                "Feature Silu expects [rows,1,1280] TP4 products");
    const auto rows=p.size(0);
    TORCH_CHECK(ids.sizes()==at::IntArrayRef({1,rows}) && route.sizes()==ids.sizes() &&
                sx.dim()==2 && sx.size(1)==1 && (sx.size(0)==1 || sx.size(0)==rows) &&
                ch.dim()==3 && ch.size(0)>=1 && ch.size(0)<=384 && ch.size(1)*256==p.size(2) && ch.size(2)==256,
                "Feature Silu requires matching row scales, IDs, routing and N256 channel scales");
    const at::ScalarType types[]={at::kFloat,at::kInt,at::kFloat,at::kBFloat16,at::kFloat};
    for (int i=0;i<5;++i) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.scalar_type()==types[i] && t.device()==p.device() && t.is_contiguous() && !t.requires_grad(),
                    "Feature Silu requires matching contiguous inference operands");
    }
    return {{at::ScalarType::Float8_e4m3fn,{rows,1,p.size(2)/2}}, {at::kFloat,{rows,1,1}}};
}
class FeatureSilu final : public habana::OpBackend {
public:
    FeatureSilu(int device, at::ScalarType dtype)
        : OpBackend(device, NO_TPC+std::string("dsv41_feature_silu"),dtype,{0,1},{},{},false) {
        SetOutputMetaFn([](const at::Stack& s) { return meta(s); });
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=meta(s); const auto rows=s.at(0).toTensor().size(0);
        auto stage=BuildNode(this,graph,{"custom_deepseek_v41_silu_activate_tile_gaudi2",
            {syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4)},
            {{{rows,1,640},at::kBFloat16},{{rows,1,5},at::kFloat}}});
        auto quant=BuildNode(this,graph,{"custom_deepseek_v41_silu_quant_tile_gaudi2",
            {stage.at(0).get(),stage.at(1).get()},
            {{out[0].shape,out[0].dtype,0},{out[1].shape,out[1].dtype,1}}});
        syn_out(0)=std::move(quant[0]);syn_out(1)=std::move(quant[1]);
    }
};
const bool registered=[] {
    for (const auto schema : {reference_schema,tiled_schema}) {
        habana::custom_op::registerUserCustomOp(schema,reference_guid,[](const at::Stack& s) {
            const auto out=meta(s);
            return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape},{out[1].dtype,out[1].shape}};
        },nullptr);
    }
    habana::custom_op::registerUserCustomOp(shared_schema, shared_guid, [](const at::Stack& s) {
        const auto out=meta(s, true);
        return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape},{out[1].dtype,out[1].shape}};
    },nullptr);
    habana::KernelRegistry().add(tiled_schema,[](synDeviceId d,at::ScalarType t) {
        return std::make_shared<FeatureSilu>(d,t);
    });
    return true;
}();
template<bool Meta,bool Tiled,bool Shared=false> Output execute(const at::Tensor& p,const at::Tensor& ids,const at::Tensor& sx,
                                             const at::Tensor& ch,const at::Tensor& route) {
    const at::Stack s{p,ids,sx,ch,route};const auto out=meta(s,Shared);
    if (Meta) return {at::empty(out[0].shape,p.options().dtype(out[0].dtype)),
                      at::empty(out[1].shape,p.options().dtype(out[1].dtype))};
    TORCH_CHECK(registered && p.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Shared?shared_schema:Tiled?tiled_schema:reference_schema);
    auto result=op.execute(s);return {result[0],result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_silu_quant_reference_gaudi2(Tensor product, Tensor ids, Tensor activation_scale, Tensor channel, Tensor router) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_shared_silu_quant_gaudi2(Tensor product, Tensor ids, Tensor activation_scale, Tensor channel, Tensor router) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_silu_quant_feature_gaudi2(Tensor product, Tensor ids, Tensor activation_scale, Tensor channel, Tensor router) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_silu_quant_reference_gaudi2",execute<false,false>);
    m.impl("custom_deepseek_v41_silu_quant_feature_gaudi2",execute<false,true>);
    m.impl("custom_deepseek_v41_shared_silu_quant_gaudi2",execute<false,false,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_silu_quant_reference_gaudi2",execute<true,false>);
    m.impl("custom_deepseek_v41_silu_quant_feature_gaudi2",execute<true,true>);
    m.impl("custom_deepseek_v41_shared_silu_quant_gaudi2",execute<true,false,true>);
}
