// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
const char* names[]={"custom_op::custom_deepseek_v41_prefix_layout_r1_i32_gaudi2",
                    "custom_op::custom_deepseek_v41_prefix_layout_r2_i32_gaudi2"};
habana::PartialOutputMetaDataVector meta(const at::Stack& stack) {
    const auto& s=stack.at(0).toTensor();const auto& p=stack.at(1).toTensor();const auto& b=stack.at(2).toTensor();
    TORCH_CHECK(s.scalar_type()==at::kInt && p.scalar_type()==at::kInt && b.scalar_type()==at::kInt &&
        s.dim()==2 && s.size(0)>0 && s.size(0)<=6 && s.size(1)==512 && p.dim()==1 && p.size(0)==s.size(0) &&
        b.dim()==1 && b.numel()>=8 && s.device()==p.device() && s.device()==b.device() && s.is_contiguous() &&
        p.is_contiguous() && b.is_contiguous(),"Invalid V4.1 short-search prefix layout contract");
    return {{at::kInt,{1,768}},{at::kInt,{s.size(0),640}},{at::kInt,{s.size(0)}}};
}
const bool registered=[] {
    for(unsigned mode=0;mode<2;++mode)habana::custom_op::registerUserCustomOp(names[mode],names[mode]+11,meta,nullptr);
    return true;
}();
template<bool Meta,unsigned Mode> std::tuple<at::Tensor,at::Tensor,at::Tensor> layout(
    const at::Tensor& s,const at::Tensor& p,const at::Tensor& b) {
    const auto shapes=meta({s,p,b});
    if(Meta)return {at::empty({1,768},s.options()),at::empty({s.size(0),640},s.options()),at::empty({s.size(0)},s.options())};
    TORCH_CHECK(registered && s.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[Mode]);
    auto outputs=op.execute({s,p,b});return {outputs.at(0),outputs.at(1),outputs.at(2)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_prefix_layout_r1_i32_gaudi2(Tensor selected, Tensor positions, Tensor block_table) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_prefix_layout_r2_i32_gaudi2(Tensor selected, Tensor positions, Tensor block_table) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_prefix_layout_r1_i32_gaudi2",layout<false,0>);
    m.impl("custom_deepseek_v41_prefix_layout_r2_i32_gaudi2",layout<false,1>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_prefix_layout_r1_i32_gaudi2",layout<true,0>);
    m.impl("custom_deepseek_v41_prefix_layout_r2_i32_gaudi2",layout<true,1>);
}
