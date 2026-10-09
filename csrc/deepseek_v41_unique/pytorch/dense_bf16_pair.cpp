// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
namespace {
constexpr auto name="custom_deepseek_v41_dense_bf16_pair_gaudi2";
constexpr auto schema="custom_op::custom_deepseek_v41_dense_bf16_pair_gaudi2";
void validate(const at::Tensor& x) {
    TORCH_CHECK(x.dim()==2 && x.size(0)>=2 && x.size(0)<=6 && x.size(1)==5120 &&
                x.scalar_type()==at::kBFloat16 && x.is_contiguous() && !x.requires_grad(),
                "Joint projection requires C2-C6 contiguous BF16 model rows");
}
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,name,[](const at::Stack& stack) {
        TORCH_CHECK(stack.size()==1);const auto x=stack[0].toTensor();validate(x);
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,{x.size(0)*2,5120}}};
    },nullptr);return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x) {
    validate(x);
    if constexpr(Meta)return at::empty({x.size(0)*2,5120},x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({x})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_dense_bf16_pair_gaudi2(Tensor value) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl(name,run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl(name,run<true>);}
