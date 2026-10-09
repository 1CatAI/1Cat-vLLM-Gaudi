// SPDX-License-Identifier: Apache-2.0
// Diagnostic outputs deliberately materialize boundaries. No performance vote.
#include "hpu_ops/op_backend.h"
#include <ATen/ATen.h>
#include "synapse_common_types.h"
#include <torch/library.h>
#include <tuple>

namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_w13_boundary_probe_gaudi2";
constexpr auto marker="custom_deepseek_v41_expert_n256_scale_gaudi2";
using Six=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    TORCH_CHECK(s.size()==8);
    const auto x=s[0].toTensor(),ids=s[1].toTensor(),routing=s[2].toTensor();
    const auto q=s[3].toTensor(),planes=s[4].toTensor(),channel=s[6].toTensor(),sx=s[7].toTensor();
    const auto rows=x.size(0),k=x.size(1),n=q.size(1)*256;
    TORCH_CHECK(rows>=2&&rows<=6&&x.scalar_type()==at::ScalarType::Float8_e4m3fn&&k==5120);
    TORCH_CHECK(ids.scalar_type()==at::kInt&&ids.sizes()==at::IntArrayRef({rows,6})&&
        routing.scalar_type()==at::kFloat&&routing.sizes()==ids.sizes());
    TORCH_CHECK(q.scalar_type()==at::kShort&&q.size(0)==384&&q.size(2)==k*64&&n==1280&&
        planes.scalar_type()==at::kShort&&planes.sizes()==at::IntArrayRef({384,n/256,k*4+128})&&
        channel.scalar_type()==at::kBFloat16&&channel.sizes()==at::IntArrayRef({384,n/256,256})&&
        sx.scalar_type()==at::kFloat&&sx.sizes()==at::IntArrayRef({rows,1}));
    for(const auto& value:s) {
        const auto t=value.toTensor();
        TORCH_CHECK(t.device()==x.device()&&t.is_contiguous()&&!t.requires_grad());
    }
    return {{at::kFloat,{rows*3,1,n*2}},{at::kBFloat16,{rows*6,1,n}},
            {at::ScalarType::Float8_e4m3fn,{rows*6,1,n/2}},
            {at::ScalarType::Float8_e4m3fn,{rows*6,1,n/2}},
            {at::kFloat,{rows*6,1,1}},{at::kFloat,{rows*6,1,1}}};
}
class Probe final:public habana::OpBackend {
public:
    Probe(int device,c10::ScalarType type)
        :OpBackend(device,NO_TPC+std::string("dsv41_w13_boundary_probe"),type,{0,1,2,3,4,5},{},{},false) {
        SetOutputMetaFn(meta);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=meta(s);const auto rows=s[0].toTensor().size(0),k=s[0].toTensor().size(1);
        const int64_t slots=rows*6,n=1280;
        auto ids=ReshapeHelper(graph,syn_in(1),{1,slots},at::kInt);
        auto expand=[&](synTensor x,int64_t width,at::ScalarType dtype,int64_t copies) {
            auto view=ReshapeHelper(graph,x,{rows,1,width},dtype);
            auto broadcast=BuildNode(this,graph,{"broadcast",{view.get()},{{{rows,copies,width},dtype}}});
            return ReshapeHelper(graph,broadcast[0].get(),{rows*copies,1,width},dtype);
        };
        auto act=expand(syn_in(0),k,at::ScalarType::Float8_e4m3fn,3);
        auto scales=expand(syn_in(7),1,at::kFloat,6);
        auto sx=ReshapeHelper(graph,scales.get(),{slots,1},at::kFloat);
        auto decoded=BuildNode(this,graph,{"custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2",
            {ids.get(),syn_in(3),syn_in(4),syn_in(5)},
            {{{slots/2,k,n*2},at::ScalarType::Float8_e4m3fn}}});
        synGEMMParams params{false,false};
        auto product=BuildNode(this,graph,{"batch_gemm",{act.get(),decoded[0].get()},
            {{out[0].shape,out[0].dtype,0}},&params,sizeof(params)});
        auto flat=ReshapeHelper(graph,product[0].get(),{slots,1,n},at::kFloat);
        auto scaled=BuildNode(this,graph,{marker,{flat.get(),ids.get(),sx.get(),syn_in(6)},
            {{out[1].shape,out[1].dtype,1}}});
        auto router=ReshapeHelper(graph,syn_in(2),{1,slots},at::kFloat);
        auto old=BuildNode(this,graph,{"custom_deepseek_v41_expert_n256_silu_quant_gaudi2",
            {flat.get(),ids.get(),sx.get(),syn_in(6),router.get()},
            {{out[2].shape,out[2].dtype,2},{out[4].shape,out[4].dtype,4}}});
        auto candidate=BuildNode(this,graph,{"custom_deepseek_v41_expert_scaled_silu_quant_gaudi2",
            {scaled[0].get(),ids.get(),sx.get(),syn_in(6),router.get()},
            {{out[3].shape,out[3].dtype,3},{out[5].shape,out[5].dtype,5}}});
        syn_out(0)=std::move(product[0]);syn_out(1)=std::move(scaled[0]);
        syn_out(2)=std::move(old[0]);syn_out(3)=std::move(candidate[0]);
        syn_out(4)=std::move(old[1]);syn_out(5)=std::move(candidate[1]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,marker,[](const at::Stack& s) {
        habana::PartialOutputMetaDataVector result;
        for(const auto& out:meta(s))result.push_back({out.dtype,out.shape});
        return result;
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId device,c10::ScalarType type) {
        return std::make_shared<Probe>(device,type);
    });return true;
}();
template<bool Meta> Six run(const at::Tensor& x,const at::Tensor& ids,const at::Tensor& routing,
    const at::Tensor& q,const at::Tensor& planes,const at::Tensor& lookup,const at::Tensor& channel,
    const at::Tensor& sx) {
    const at::Stack s{x,ids,routing,q,planes,lookup,channel,sx};const auto outputs=meta(s);
    if constexpr(Meta) {
        return {at::empty(outputs[0].shape,x.options().dtype(outputs[0].dtype)),
            at::empty(outputs[1].shape,x.options().dtype(outputs[1].dtype)),
            at::empty(outputs[2].shape,x.options().dtype(outputs[2].dtype)),
            at::empty(outputs[3].shape,x.options().dtype(outputs[3].dtype)),
            at::empty(outputs[4].shape,x.options().dtype(outputs[4].dtype)),
            at::empty(outputs[5].shape,x.options().dtype(outputs[5].dtype))};
    }
    TORCH_CHECK(ready&&x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto v=descriptor.execute(s);return {v[0],v[1],v[2],v[3],v[4],v[5]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_w13_boundary_probe_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor s13, Tensor lookup, Tensor channel13, Tensor activation_scale) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_w13_boundary_probe_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_w13_boundary_probe_gaudi2",run<true>);}
