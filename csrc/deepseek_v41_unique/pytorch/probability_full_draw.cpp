// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_probability_full_draw_gaudi2";
constexpr auto guid="custom_deepseek_v41_probability_full_select_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size()==2,"Full probability draw requires q and controls");
    const auto q=s[0].toTensor(),c=s[1].toTensor();
    TORCH_CHECK(q.dim()==2 && q.size(0)>=1 && q.size(0)<=6 && q.size(1)>=64 && q.size(1)<=131072 &&
        q.size(1)%64==0 && q.scalar_type()==at::kFloat && q.is_contiguous() && !q.requires_grad() &&
        c.sizes()==at::IntArrayRef({q.size(0),4}) && c.scalar_type()==at::kFloat && c.is_contiguous() &&
        !c.requires_grad() && c.device()==q.device(),"Full draw requires F32[C1-C6,V] and F32[C,4]");
    habana::OutputMetaData m;m.shape={q.size(0)};m.dtype=at::kInt;return {m};
}
class Draw:public habana::OpBackend {
public:
    Draw(synDeviceId d,c10::ScalarType t):OpBackend(d,guid,t,{0},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        const auto q=s[0].toTensor();
        auto result=BuildNode(this,g,{guid,{syn_in(0),syn_in(1)},{{{q.size(0)},at::kInt,0}}});
        syn_out(0)=std::move(result[0]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,guid,[](const at::Stack& s) {
        const auto m=metadata(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Draw>(d,t);});
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& q,const at::Tensor& c) {
    const auto m=metadata({q,c});
    if constexpr(Meta)return at::empty(m[0].shape,q.options().dtype(m[0].dtype));
    TORCH_CHECK(ready && q.device().type()==at::kHPU,"Full probability draw requires HPU");
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute({q,c})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {m.def("custom_deepseek_v41_probability_full_draw_gaudi2(Tensor q, Tensor controls) -> Tensor");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_probability_full_draw_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_probability_full_draw_gaudi2",run<true>);}
