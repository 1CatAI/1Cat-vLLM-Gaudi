// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_nucleus_mass_sample_gaudi2";
using Outputs=std::tuple<at::Tensor,at::Tensor,at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& s) {
    TORCH_CHECK(s.size()==2,"Nucleus sampler requires real logits and request controls");
    const auto x=s[0].toTensor(),c=s[1].toTensor();
    TORCH_CHECK(x.scalar_type()==at::kFloat && x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 &&
        x.size(1)>=128 && x.size(1)<=129280 && x.size(1)%128==0 &&
        c.scalar_type()==at::kFloat && c.sizes()==at::IntArrayRef({x.size(0),4}) &&
        x.device()==c.device() && x.is_contiguous() && c.is_contiguous() &&
        !x.requires_grad() && !c.requires_grad(),"Nucleus requires F32 [C,N], C1-C6 and official F32 controls");
    return {{at::kFloat,x.sizes().vec()},{at::kInt,{x.size(0)}},{at::kInt,{x.size(0)}}};
}
class Sampler final:public habana::OpBackend {
 public:
    Sampler(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_nucleus_mass"),t,{0,1,2},{},{},false){SetOutputMetaFn(meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=meta(s);const auto rows=out[0].shape[0],columns=out[0].shape[1];
        auto mass=BuildNode(this,graph,{"custom_deepseek_v41_nucleus_mass_gaudi2",{syn_in(0),syn_in(1)},
            {{out[0].shape,at::kFloat,0},{{rows,columns/32},at::kInt},{{rows,2},at::kInt}}});
        auto draw=BuildNode(this,graph,{"custom_deepseek_v41_nucleus_draw_gaudi2",
            {mass[0].get(),mass[1].get(),mass[2].get(),syn_in(1)},
            {{{rows,2},at::kInt},{{rows,columns/64},at::kInt}}});
        auto finish=BuildNode(this,graph,{"custom_deepseek_v41_nucleus_finish_gaudi2",
            {draw[0].get(),draw[1].get()},{{out[1].shape,at::kInt,1},{out[2].shape,at::kInt,2}}});
        syn_out(0)=std::move(mass[0]);syn_out(1)=std::move(finish[0]);syn_out(2)=std::move(finish[1]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,"custom_deepseek_v41_nucleus_mass_gaudi2",[](const at::Stack& s){
        habana::PartialOutputMetaDataVector result;for(const auto& m:meta(s))result.push_back({m.dtype,m.shape});return result;
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Sampler>(d,t);});return true;
}();
template<bool Meta> Outputs run(const at::Tensor& x,const at::Tensor& c) {
    const at::Stack s{x,c};const auto m=meta(s);
    if constexpr(Meta)return {at::empty(m[0].shape,x.options()),at::empty(m[1].shape,x.options().dtype(at::kInt)),
                             at::empty(m[2].shape,x.options().dtype(at::kInt))};
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);auto out=op.execute(s);
    return {out[0],out[1],out[2]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m){m.def("custom_deepseek_v41_nucleus_mass_sample_gaudi2(Tensor logits, Tensor controls) -> (Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_nucleus_mass_sample_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_nucleus_mass_sample_gaudi2",run<true>);}
