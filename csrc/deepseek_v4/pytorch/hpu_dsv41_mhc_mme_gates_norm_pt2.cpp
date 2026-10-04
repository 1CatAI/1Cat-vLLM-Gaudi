// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto guid="custom_deepseek_v41_mhc_mme_gates_norm_gaudi2";
constexpr auto schema="custom_op::custom_deepseek_v41_mhc_mme_gates_norm_gaudi2";
struct Params {float epsilon;float inverse_width;};
using Outputs=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto m=s.at(0).toTensor(),r=s.at(1).toTensor(),x=s.at(2).toTensor(),w=s.at(3).toTensor();
    const auto scale=s.at(4).toTensor(),base=s.at(5).toTensor();
    const auto b=m.size(0);const auto eps=s.at(6).toDouble();
    TORCH_CHECK(m.dim()==2 && b>=1 && b<=6 && m.size(1)==48 &&
        r.sizes()==at::IntArrayRef({b,20480}) && x.sizes()==at::IntArrayRef({b,5120}) &&
        w.sizes()==at::IntArrayRef({5120}) && scale.sizes()==at::IntArrayRef({3}) &&
        base.sizes()==at::IntArrayRef({24}) && std::isnormal(float(eps)) && eps>0,
        "Invalid mHC MME/gates/norm operand shape or epsilon");
    for(unsigned i=0;i<6;++i) {
        const auto t=s.at(i).toTensor();
        const auto dtype=(i>=1 && i<=3)?at::kBFloat16:at::kFloat;
        TORCH_CHECK(t.scalar_type()==dtype && t.device()==m.device() &&
            t.is_contiguous() && !t.requires_grad(),"Invalid mHC inference tensor contract");
    }
    return {{at::kFloat,{b,24}},{at::kBFloat16,{b,5120}},
            {at::ScalarType::Float8_e4m3fn,{b,5120}},{at::kFloat,{b,1}}};
}
class Kernel final:public habana::OpBackend {
public:
    Kernel(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_mhc_mme_gates_norm"),
        t,{0,1,2,3},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        const auto meta=metadata(s);Params p{float(s.at(6).toDouble()),1.0f/5120.0f};
        auto out=BuildNode(this,g,{guid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4),syn_in(5)},
            {{meta[0].shape,meta[0].dtype,0},{meta[1].shape,meta[1].dtype,1},
             {meta[2].shape,meta[2].dtype,2},{meta[3].shape,meta[3].dtype,3}},&p,sizeof(p)});
        for(unsigned i=0;i<4;++i) syn_out(i)=std::move(out[i]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s){
        auto m=metadata(s);habana::PartialOutputMetaDataVector out;
        for(const auto& v:m) out.push_back({v.dtype,v.shape});return out;
    },[](const at::Stack& s,size_t& size)->std::shared_ptr<void>{
        size=sizeof(Params);return std::make_shared<Params>(Params{float(s.at(6).toDouble()),1.0f/5120.0f});});
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Kernel>(d,t);});
    return true;
}();
template<bool Meta> Outputs run(const at::Tensor& m,const at::Tensor& r,const at::Tensor& x,
    const at::Tensor& w,const at::Tensor& scale,const at::Tensor& base,double eps) {
    const at::Stack s{m,r,x,w,scale,base,eps};auto meta=metadata(s);std::vector<at::Tensor> out;
    if(Meta) for(const auto& v:meta) out.push_back(at::empty(v.shape,m.options().dtype(v.dtype)));
    else {TORCH_CHECK(registered && m.device().type()==at::kHPU);
        auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
        out=descriptor.execute(s);}
    return {out[0],out[1],out[2],out[3]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){m.def("custom_deepseek_v41_mhc_mme_gates_norm_gaudi2(Tensor mixes, Tensor residual, Tensor collapsed, Tensor norm, Tensor scale, Tensor base, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_mhc_mme_gates_norm_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_mhc_mme_gates_norm_gaudi2",run<true>);}
