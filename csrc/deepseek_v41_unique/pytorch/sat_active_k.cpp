// SPDX-License-Identifier: Apache-2.0
#define DSV41_N256_BACKEND_ONLY 1
#define DSV41_N256_ACTIVE_INTERMEDIATE_ARG 12
#define DSV41_N256_W2_BOUNDED_GUID "custom_deepseek_v41_expert_n256_sat_active_k_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_expert_n256_moe_active_k_fp8_gaudi2";
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,kDecode,[](const at::Stack& stack) {
        prequant_contract(stack);
        TORCH_CHECK(stack.back().toBool() && stack.at(0).toTensor().size(0)>=2 &&
                    stack.at(0).toTensor().size(0)<=6,"SAT requires qualified C2-C6 scale planes");
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,moe_shape(stack,true)}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId device,c10::ScalarType type) {
        return std::make_shared<PreparedV41>(device,type,true,true,false,true,
            true,false,true,false,true,false,false,false,true,false,false,true);
    });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& ids,const at::Tensor& routing,
    const at::Tensor& q13,const at::Tensor& q2,const at::Tensor& s13,const at::Tensor& s2,
    const at::Tensor& lookup,const at::Tensor& c13,const at::Tensor& c2,
    const at::Tensor& quantized,const at::Tensor& sx,int64_t active_k,bool qualified) {
    const at::Stack stack{x,ids,routing,q13,q2,s13,s2,lookup,c13,c2,quantized,sx,active_k,qualified};
    prequant_contract(stack);
    TORCH_CHECK(qualified && x.size(0)>=2 && x.size(0)<=6 && active_k>0 && active_k%32==0 &&
                active_k<=q2.size(2)/64,"Invalid logical expert width");
    if(Meta)return at::empty(moe_shape(stack,true),x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_expert_n256_moe_active_k_fp8_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2, Tensor quantized, Tensor activation_scale, int active_k, bool qualified) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_expert_n256_moe_active_k_fp8_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_expert_n256_moe_active_k_fp8_gaudi2",run<true>);}
