// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
const char* names[]={"custom_op::custom_deepseek_v41_rope_bf16_gaudi2",
                    "custom_op::custom_deepseek_v41_rope_inverse_bf16_gaudi2"};
std::vector<int64_t> shape(const at::Tensor& x,const at::Tensor& p,const at::Tensor& t) {
    TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.dim()==3 && x.size(0)>0 && x.size(0)<=6 &&
        x.size(1)>0 && x.size(1)<=128 && x.size(2)>0 && x.size(2)<=512 && x.size(2)%128==0 &&
        p.scalar_type()==at::kInt && p.dim()==1 && p.size(0)==x.size(0) &&
        t.scalar_type()==at::kFloat && t.dim()==2 && t.size(0)>0 && t.size(1)==64 &&
        x.device()==p.device() && x.device()==t.device() && x.is_contiguous() && p.is_contiguous() &&
        t.is_contiguous() && !x.requires_grad() && !t.requires_grad(), "Invalid V4.1 RoPE input contract");
    return x.sizes().vec();
}
const bool registered=[] {
    for(unsigned mode=0;mode<2;++mode)
        habana::custom_op::registerUserCustomOp(names[mode],names[mode]+11,[](const at::Stack& s) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16,
                shape(s.at(0).toTensor(),s.at(1).toTensor(),s.at(2).toTensor())}};
        },nullptr);
    return true;
}();
template<bool Meta,unsigned Mode> at::Tensor rope(const at::Tensor& x,const at::Tensor& p,const at::Tensor& t) {
    const auto sizes=shape(x,p,t);
    if(Meta)return at::empty(sizes,x.options());
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[Mode]);
    return op.execute({x,p,t}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_rope_bf16_gaudi2(Tensor value, Tensor positions, Tensor phase) -> Tensor");
    m.def("custom_deepseek_v41_rope_inverse_bf16_gaudi2(Tensor value, Tensor positions, Tensor phase) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_rope_bf16_gaudi2",rope<false,0>);
    m.impl("custom_deepseek_v41_rope_inverse_bf16_gaudi2",rope<false,1>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_rope_bf16_gaudi2",rope<true,0>);
    m.impl("custom_deepseek_v41_rope_inverse_bf16_gaudi2",rope<true,1>);
}
