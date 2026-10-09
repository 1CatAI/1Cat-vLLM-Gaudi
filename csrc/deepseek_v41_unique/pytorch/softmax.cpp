// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_vocab_softmax_f32_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& stack) {
    const auto x=stack.at(0).toTensor();
    TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 && x.size(1)>=2048 &&
                x.size(1)<=131072 && x.size(1)%64==0 && x.scalar_type()==at::kFloat &&
                x.is_contiguous() && !x.requires_grad(),"Vocabulary softmax requires contiguous C1-C6 FP32 rows");
    return {{at::kFloat,x.sizes().vec()}};
}
class Softmax final: public habana::OpBackend {
public:
    Softmax(int device,c10::ScalarType type):
        OpBackend(device,NO_TPC+std::string("dsv41_vocab_softmax"),type,{0},{},{},false) {SetOutputMetaFn(meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& stack) override {
        const auto x=stack.at(0).toTensor();
        const auto parts=(x.size(1)+2047)/2048;
        auto statistics=BuildNode(this,graph,{"custom_deepseek_v41_softmax_parts_gaudi2",
            {syn_in(0)},{{{x.size(0),2,parts},at::kFloat}}});
        syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_softmax_normalize_gaudi2",
            {syn_in(0),statistics[0].get()},{{x.sizes().vec(),at::kFloat,0}}})[0]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,"custom_deepseek_v41_softmax_parts_gaudi2",
        [](const at::Stack& stack) {const auto out=meta(stack);return habana::PartialOutputMetaDataVector{
            {out[0].dtype,out[0].shape}};},nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId device,c10::ScalarType type) {
        return std::make_shared<Softmax>(device,type);
    });return true;
}();
template<bool Fake> at::Tensor run(const at::Tensor& x) {
    meta({x});
    if(Fake)return at::empty_like(x);
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute({x})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {m.def("custom_deepseek_v41_vocab_softmax_f32_gaudi2(Tensor x) -> Tensor");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_vocab_softmax_f32_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_vocab_softmax_f32_gaudi2",run<true>);}
