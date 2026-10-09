// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* names[]={"custom_deepseek_v41_weighted_mass_bins_gaudi2",
    "custom_deepseek_v41_weighted_mass_advance_gaudi2","custom_deepseek_v41_weighted_mass_finish_gaudi2"};
std::string schema(int kind) {return std::string("custom_op::")+names[kind];}
habana::OutputMetaDataVector metadata(const at::Stack& s,int kind) {
    TORCH_CHECK(s.size()==static_cast<unsigned>(kind==2?3:4),"Weighted nucleus input count");
    const auto x=s[0].toTensor(),w=s[1].toTensor(),prefix=s[2].toTensor();
    TORCH_CHECK(x.dim()==(kind==1?3:2) && x.size(0)>=1 && x.size(0)<=6 && x.scalar_type()==at::kFloat,
                "Weighted nucleus requires F32 C1-C6 matrices");
    const auto rows=x.size(0),columns=x.size(x.dim()-1),parts=kind==1?x.size(1):(columns+2047)/2048;
    for(int i=0;i<3;++i) {
        const auto t=s[i].toTensor();
        TORCH_CHECK(t.device()==x.device() && t.is_contiguous() && !t.requires_grad(),
                    "Weighted nucleus inputs must be contiguous, same device, without gradients");
    }
    if(kind==1) {
        // Advance's second input is an I32 prefix and third input is F32 mass.
        TORCH_CHECK(columns==16 && parts>=1 && parts<=64 && w.scalar_type()==at::kInt && prefix.scalar_type()==at::kFloat &&
                    w.sizes()==at::IntArrayRef({rows,1}) && prefix.sizes()==w.sizes(),
                    "Weighted advance requires [C,parts1..64,16], I32[C,1], F32[C,1]");
    } else {
        TORCH_CHECK(columns>=64 && columns<=131072 && columns%64==0 &&
                    w.scalar_type()==at::kFloat && w.sizes()==x.sizes() &&
                    prefix.scalar_type()==at::kInt && prefix.sizes()==at::IntArrayRef({rows,1}),
                    "Weighted bins/finish require F32[C,N64..131072] and I32[C,1]");
    }
    if(kind!=2) {
        const auto shift=s[3].toInt();
        TORCH_CHECK(shift>=0 && shift<=28 && shift%4==0,"Weighted radix shift must be 0,4,...,28");
    }
    if(kind==0)return {{at::kFloat,{rows,parts,16}}};
    if(kind==1)return {{at::kInt,{rows,1}},{at::kFloat,{rows,1}}};
    return {{at::kFloat,{rows,columns}},{at::kInt,{rows,parts}},{at::kInt,{rows,parts}}};
}
template<int Kind> class Weighted final:public habana::OpBackend {
public:
    Weighted(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string(names[Kind]),t,
        Kind==0?std::vector<int>{0}:Kind==1?std::vector<int>{0,1}:std::vector<int>{0,1,2},{},{},false) {
        SetOutputMetaFn([](const at::Stack& s){return metadata(s,Kind);});
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=metadata(s,Kind);
        habana::NodeAttr attr;attr.guid=names[Kind];attr.inputs={syn_in(0),syn_in(1),syn_in(2)};
        for(unsigned i=0;i<m.size();++i)attr.output_attrs.push_back({m[i].shape,m[i].dtype,int(i)});
        int32_t shift=Kind==2?0:static_cast<int32_t>(s[3].toInt());
        if constexpr(Kind!=2){attr.params=&shift;attr.param_size=sizeof(shift);}
        auto out=BuildNode(this,graph,std::move(attr));
        for(unsigned i=0;i<m.size();++i)syn_out(i)=std::move(out[i]);
    }
};
template<int Kind> bool register_op() {
    habana::custom_op::registerUserCustomOp(schema(Kind),names[Kind],[](const at::Stack& s){
        habana::PartialOutputMetaDataVector out;
        for(const auto& m:metadata(s,Kind)) {out.push_back({m.dtype,m.shape});}
        return out;
    },nullptr);
    habana::KernelRegistry().add(schema(Kind),[](synDeviceId d,c10::ScalarType t){return std::make_shared<Weighted<Kind>>(d,t);});
    return true;
}
const bool ready=register_op<0>() && register_op<1>() && register_op<2>();
template<bool Meta,int Kind> std::vector<at::Tensor> execute(const at::Stack& s) {
    const auto m=metadata(s,Kind);const auto x=s[0].toTensor();
    if constexpr(Meta) {
        std::vector<at::Tensor> out;for(const auto& t:m)out.push_back(at::empty(t.shape,x.options().dtype(t.dtype)));return out;
    }
    TORCH_CHECK(ready && x.device().type()==at::kHPU,"Weighted nucleus requires HPU");
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema(Kind));return op.execute(s);
}
template<bool M> at::Tensor bins(const at::Tensor& x,const at::Tensor& w,const at::Tensor& p,int64_t shift) {
    return execute<M,0>({x,w,p,shift})[0];
}
template<bool M> std::tuple<at::Tensor,at::Tensor> advance(const at::Tensor& x,const at::Tensor& p,const at::Tensor& t,int64_t shift) {
    auto out=execute<M,1>({x,p,t,shift});return {out[0],out[1]};
}
template<bool M> std::tuple<at::Tensor,at::Tensor,at::Tensor> finish(const at::Tensor& x,const at::Tensor& w,const at::Tensor& p) {
    auto out=execute<M,2>({x,w,p});return {out[0],out[1],out[2]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_weighted_mass_bins_gaudi2(Tensor scores, Tensor probability, Tensor prefix, int shift) -> Tensor");
    m.def("custom_deepseek_v41_weighted_mass_advance_gaudi2(Tensor bins, Tensor prefix, Tensor target, int shift) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_weighted_mass_finish_gaudi2(Tensor scores, Tensor probability, Tensor prefix) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl(names[0],bins<false>);m.impl(names[1],advance<false>);m.impl(names[2],finish<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl(names[0],bins<true>);m.impl(names[1],advance<true>);m.impl(names[2],finish<true>);
}
