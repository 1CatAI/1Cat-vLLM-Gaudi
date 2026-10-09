// SPDX-License-Identifier: Apache-2.0
#define DSV41_N256_BACKEND_ONLY 1
#define DSV41_N256_DRAFT_ROWS 1
#define DSV41_N256_FP8_GUID "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2";
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,kDecode,[](const at::Stack& stack) {
        fp8_contract(stack,true);
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,moe_shape(stack,true)}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t) {
        constexpr bool moe=true,fp8=true,k128=false,n256=true,fused=true;
        return std::make_shared<PreparedV41>(d,t,moe,fp8,k128,n256,fused);
    });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& ids,const at::Tensor& routing,
    const at::Tensor& q13,const at::Tensor& q2,const at::Tensor& s13,const at::Tensor& s2,
    const at::Tensor& lookup,const at::Tensor& c13,const at::Tensor& c2,bool qualified) {
    const at::Stack stack{x,ids,routing,q13,q2,s13,s2,lookup,c13,c2,qualified};
    fp8_contract(stack,true);
    if(Meta)return at::empty(moe_shape(stack,true),x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor c13, Tensor c2, bool qualified) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_mtp_sat_moe_fp8_gaudi2",run<true>);}
