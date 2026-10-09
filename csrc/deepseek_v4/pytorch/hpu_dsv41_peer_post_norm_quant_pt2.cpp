// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
#ifndef DSV41_PEER_POST_NORM_SCHEMA
#define DSV41_PEER_POST_NORM_SCHEMA "custom_deepseek_v41_peer_post_norm_quant_gaudi2"
#endif
namespace {
constexpr auto kSchema="custom_op::" DSV41_PEER_POST_NORM_SCHEMA;
constexpr auto kGuid="custom_deepseek_v41_peer_post_norm_quant_gaudi2";
using Outputs=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    TORCH_CHECK(s.size()==7,"Peer post/norm requires its complete operands");
    const auto x=s[0].toTensor(),r=s[1].toTensor();
    TORCH_CHECK(x.dim()==3 && (x.size(0)==1 || x.size(0)==2 || x.size(0)==4) &&
                (x.size(1)>=1 && x.size(1)<=6) && x.size(2)==5120 && r.sizes()==at::IntArrayRef({x.size(1),4,5120}),
                "Peer post/norm requires TP-parametric [P,C,5120] and C1-C6 residual");
    for(int i=0;i<6;++i) {
        const auto t=s[i].toTensor();const auto type=(i==0 || i==1 || i==5)?at::kBFloat16:at::kFloat;
        TORCH_CHECK(t.scalar_type()==type && t.device()==x.device() && t.is_contiguous() && !t.requires_grad(),
                    "Peer post/norm requires matching contiguous inference operands");
    }
    TORCH_CHECK(s[2].toTensor().sizes()==at::IntArrayRef({x.size(1),4}) &&
                s[3].toTensor().sizes()==at::IntArrayRef({x.size(1),4,4}) &&
                s[4].toTensor().sizes()==at::IntArrayRef({x.size(1),4}) &&
                s[5].toTensor().sizes()==at::IntArrayRef({5120}),"Peer post/norm gate or norm geometry changed");
    const double eps=s[6].toDouble();TORCH_CHECK(eps>0 && std::isnormal(float(eps)),"Peer post/norm epsilon invalid");
    return {{at::kBFloat16,{x.size(1),4,5120}},{at::kBFloat16,{x.size(1),5120}},{at::kBFloat16,{x.size(1),5120}},
            {at::ScalarType::Float8_e4m3fn,{x.size(1),5120}},{at::kFloat,{x.size(1),1}}};
}
class Fused final:public habana::OpBackend {
 public:
    Fused(int d,c10::ScalarType t):OpBackend(d,kGuid,t,{0,1,2,3,4},{},{},false){SetOutputMetaFn(meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=meta(s);float params[]={float(s[6].toDouble()),1.0f/5120};
#ifdef DSV41_PEER_POST_NORM_TILED
        auto post=BuildNode(this,graph,{"custom_deepseek_v41_peer_post_collapse_gaudi2",
            {syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4)},
            {{out[0].shape,out[0].dtype,0},{out[1].shape,out[1].dtype,1}}});
        auto norm=BuildNode(this,graph,{"custom_deepseek_v41_ffn_norm_quant_gaudi2",
            {post[1].get(),syn_in(5)},
            {{out[2].shape,out[2].dtype,2},{out[3].shape,out[3].dtype,3},{out[4].shape,out[4].dtype,4}},
            params,sizeof(params)});
        std::vector<synapse_helpers::tensor> values;
        values.push_back(std::move(post[0])); values.push_back(std::move(post[1]));
        for(auto& value:norm) values.push_back(std::move(value));
#else
        auto values=BuildNode(this,graph,{kGuid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5)},
            {{out[0].shape,out[0].dtype,0},{out[1].shape,out[1].dtype,1},{out[2].shape,out[2].dtype,2},
             {out[3].shape,out[3].dtype,3},{out[4].shape,out[4].dtype,4}},params,sizeof(params)});
#endif
        for(int i=0;i<5;++i)syn_out(i)=std::move(values[i]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(kSchema,kGuid,[](const at::Stack& s){
        habana::PartialOutputMetaDataVector result;for(const auto& o:meta(s))result.push_back({o.dtype,o.shape});return result;
    },nullptr);
    habana::KernelRegistry().add(kSchema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Fused>(d,t);});return true;
}();
template<bool Meta> Outputs run(const at::Tensor& x,const at::Tensor& r,const at::Tensor& post,
    const at::Tensor& comb,const at::Tensor& pre,const at::Tensor& norm,double eps) {
    const at::Stack s{x,r,post,comb,pre,norm,eps};const auto out=meta(s);std::vector<at::Tensor> result;
    if constexpr(Meta)for(const auto& o:out)result.push_back(at::empty(o.shape,x.options().dtype(o.dtype)));
    else {TORCH_CHECK(registered && x.device().type()==at::kHPU);
        auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);result=descriptor.execute(s);}
    return {result[0],result[1],result[2],result[3],result[4]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def(DSV41_PEER_POST_NORM_SCHEMA "(Tensor peers, Tensor residual, Tensor post, Tensor comb, Tensor next_pre, Tensor norm, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl(DSV41_PEER_POST_NORM_SCHEMA,run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl(DSV41_PEER_POST_NORM_SCHEMA,run<true>);}
