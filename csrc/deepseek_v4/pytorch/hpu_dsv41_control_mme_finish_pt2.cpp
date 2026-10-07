// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_control_mme_finish_gaudi2";
constexpr auto guid="custom_deepseek_v41_control_mme_finish_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& s) {
 const auto x=s[0].toTensor(), p=s[1].toTensor();
 TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 && x.size(1)==20480 &&
 p.sizes()==at::IntArrayRef({x.size(0),48}) && x.scalar_type()==at::kBFloat16 && p.scalar_type()==at::kFloat &&
 x.is_contiguous() && p.is_contiguous() && x.device()==p.device() && !x.requires_grad() && !p.requires_grad() &&
 s[2].toDouble()>0, "Invalid control MME finish operands");
 return {{at::kFloat,{x.size(0),25}}};
}
const bool registered=[] {
 habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
  const auto m=meta(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};
 },[](const at::Stack& s,size_t& size)->std::shared_ptr<void>{size=sizeof(float);return std::make_shared<float>(float(s[2].toDouble()));});
 return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& p,double eps) {
 at::Stack s{x,p,eps};const auto m=meta(s);
 if(Meta)return at::empty(m[0].shape,x.options().dtype(at::kFloat));
 TORCH_CHECK(registered && x.device().type()==at::kHPU);
 auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
 return descriptor.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {m.def("custom_deepseek_v41_control_mme_finish_gaudi2(Tensor residual, Tensor projection, float eps) -> Tensor");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_control_mme_finish_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_control_mme_finish_gaudi2",run<true>);}
