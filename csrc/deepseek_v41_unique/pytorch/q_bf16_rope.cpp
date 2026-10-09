// SPDX-License-Identifier: Apache-2.0
#define DSV41_Q_PROJECTION_BACKEND_ONLY 1
#define DSV41_Q_BF16_WEIGHT 1
#define DSV41_Q_SCALE_GUID "custom_deepseek_v41_q_bf16_rope_tiled_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_q_projection_rope_pt2.cpp"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_q_bf16_projection_rope_gaudi2";
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,kGuid,[](const at::Stack& stack) {
        const auto out=bf16_meta(stack)[0];
        return habana::PartialOutputMetaDataVector{{out.dtype,out.shape}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t) {
        return std::make_shared<Projection>(d,t,false);
    });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& weight,
    const at::Tensor& positions,const at::Tensor& phase,bool quantize) {
    const at::Stack stack{x,weight,positions,phase,quantize};
    const auto out=bf16_meta(stack)[0];
    if(Meta)return at::empty(out.shape,x.options());
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return op.execute(stack)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_q_bf16_projection_rope_gaudi2(Tensor input, Tensor weight, Tensor positions, Tensor phase, bool quantize) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_q_bf16_projection_rope_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_q_bf16_projection_rope_gaudi2",run<true>);}
