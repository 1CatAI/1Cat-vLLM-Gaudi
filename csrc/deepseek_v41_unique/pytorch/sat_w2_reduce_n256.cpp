// SPDX-License-Identifier: Apache-2.0
#define DSV41_N256_W13_K_PIPELINE 1
#define DSV41_N256_SCALE_REDUCE_GUID "custom_deepseek_v41_w2_reduce_n256_gaudi2"
#define DSV41_SPLIT_SCALE_OPERATOR custom_deepseek_v41_expert_n256_moe_w2_reduce_n256_fp8_gaudi2
#include "sat_split_scale_planes.cpp"

namespace {
constexpr auto diagnostic_schema="custom_op::custom_deepseek_v41_w2_reduce_n256_gaudi2";
void diagnostic_contract(const at::Stack& stack) {
    TORCH_CHECK(stack.size()==4,"W2 consumer requires product, route IDs, activation scales and channel weights");
    const auto& p=stack[0].toTensor();const auto slots=p.size(0);
    TORCH_CHECK(p.dim()==3&&p.size(1)==1&&p.size(2)==5120&&slots>=12&&slots<=36&&slots%6==0&&
                p.scalar_type()==at::kFloat,"W2 diagnostic uses the production C2-C6 product");
    TORCH_CHECK(stack[1].toTensor().sizes()==at::IntArrayRef({1,slots})&&
                stack[1].toTensor().scalar_type()==at::kInt&&
                stack[2].toTensor().sizes()==at::IntArrayRef({slots,1})&&
                stack[2].toTensor().scalar_type()==at::kFloat&&
                stack[3].toTensor().sizes()==at::IntArrayRef({384,20,256})&&
                stack[3].toTensor().scalar_type()==at::kBFloat16,"W2 diagnostic operand contract mismatch");
    for(const auto& value:stack)TORCH_CHECK(value.toTensor().device()==p.device()&&
        value.toTensor().is_contiguous()&&!value.toTensor().requires_grad());
}
const bool diagnostic_ready=[] {
    habana::custom_op::registerUserCustomOp(diagnostic_schema,DSV41_N256_SCALE_REDUCE_GUID,[](const at::Stack& stack) {
        diagnostic_contract(stack);
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,{stack[0].toTensor().size(0)/6,1,5120}}};
    },nullptr);
    return true;
}();
template<bool Meta> at::Tensor diagnostic(const at::Tensor& p,const at::Tensor& ids,
                                        const at::Tensor& sx,const at::Tensor& channel) {
    const at::Stack stack{p,ids,sx,channel};diagnostic_contract(stack);
    if constexpr(Meta)return at::empty({p.size(0)/6,1,5120},p.options().dtype(at::kBFloat16));
    TORCH_CHECK(diagnostic_ready&&p.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(diagnostic_schema);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_w2_reduce_n256_gaudi2(Tensor product, Tensor ids, Tensor scales, Tensor channel) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_w2_reduce_n256_gaudi2",diagnostic<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_w2_reduce_n256_gaudi2",diagnostic<true>);}
