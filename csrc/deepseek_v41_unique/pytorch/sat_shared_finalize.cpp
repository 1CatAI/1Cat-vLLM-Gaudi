// SPDX-License-Identifier: Apache-2.0
#define DSV41_N256_BACKEND_ONLY 1
#define DSV41_N256_TPC_SHARED_FINALIZE 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_expert_n256_pt2.cpp"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_moe_shared_finalize_gaudi2";
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,kDecode,[](const at::Stack& stack) {
        prequant_shared_contract(stack);
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,moe_shape(stack,true)}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t) {
        return std::make_shared<PreparedV41>(d,t,true,true,false,true,
            true,false,true,false,true,false,true,false,true,false,false,true);
    });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& ids,const at::Tensor& routing,
    const at::Tensor& q13,const at::Tensor& q2,const at::Tensor& s13,const at::Tensor& s2,
    const at::Tensor& lookup,const at::Tensor& c13,const at::Tensor& c2,
    const at::Tensor& quant,const at::Tensor& sx,const at::Tensor& shared,bool qualified) {
    const at::Stack stack{x,ids,routing,q13,q2,s13,s2,lookup,c13,c2,quant,sx,shared,qualified};
    prequant_shared_contract(stack);
    if(Meta)return at::empty(moe_shape(stack,true),x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return op.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_moe_shared_finalize_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor c13, Tensor c2, Tensor quant, Tensor sx, Tensor shared, bool qualified) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_moe_shared_finalize_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_moe_shared_finalize_gaudi2",run<true>);}
