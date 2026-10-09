// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "backend/habana_operator.h"
namespace {
constexpr auto name="custom_deepseek_v41_input_norm_bf16_gaudi2";
constexpr auto schema="custom_op::custom_deepseek_v41_input_norm_bf16_gaudi2";
void validate(const at::Tensor& x,const at::Tensor& weight,double eps) {
    TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 && x.size(1)==5120 &&
                weight.sizes()==at::IntArrayRef({5120}) && eps>0 && std::isnormal(float(eps)),
                "Decode input norm requires C1-C6 BF16 model rows and actual epsilon");
    for(const auto& tensor:{x,weight})
        TORCH_CHECK(tensor.scalar_type()==at::kBFloat16 && tensor.device()==x.device() &&
                    tensor.is_contiguous() && !tensor.requires_grad(),"Decode input norm operands mismatch");
}
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,name,[](const at::Stack& stack) {
        validate(stack[0].toTensor(),stack[1].toTensor(),stack[2].toDouble());
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,stack[0].toTensor().sizes().vec()}};
    },[](const at::Stack& stack,size_t& size)->std::shared_ptr<void> {
        size=sizeof(float);return std::make_shared<float>(float(stack[2].toDouble()));
    });return true;
}();
template<bool Meta> at::Tensor norm(const at::Tensor& x,const at::Tensor& weight,double eps) {
    validate(x,weight,eps);
    if constexpr(Meta)return at::empty(x.sizes(),x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({x,weight,eps})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_input_norm_bf16_gaudi2(Tensor value, Tensor weight, float epsilon) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl(name,norm<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl(name,norm<true>);}
