// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <array>
#include <cmath>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_mhc_control_reuse_gaudi2";
constexpr auto guid="custom_deepseek_v41_mhc_control_reuse_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto x=s[0].toTensor(),w=s[1].toTensor();
    TORCH_CHECK(x.dim()==2 && x.size(0)>=2 && x.size(0)<=6 && x.size(1)==20480 &&
                x.scalar_type()==at::kBFloat16 && w.sizes()==at::IntArrayRef({24,20480}) &&
                w.scalar_type()==at::kFloat && x.device()==w.device() && x.is_contiguous() &&
                w.is_contiguous() && !x.requires_grad() && !w.requires_grad() &&
                std::isfinite(s[2].toDouble()) && s[2].toDouble()>0,
                "Control reuse requires the actual BF16 C2-C6 residual and FP32 checkpoint weight");
    return {{at::kFloat,{x.size(0),25}}};
}
class Control final:public habana::OpBackend {
public:
    Control(int device,c10::ScalarType type)
        :OpBackend(device,NO_TPC+std::string("dsv41_control_reuse"),type,{0},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto output=metadata(s);
        std::array<float,2> parameters{static_cast<float>(s[2].toDouble()),1.0f/20480.0f};
        syn_out(0)=std::move(BuildNode(this,graph,{guid,{syn_in(0),syn_in(1)},
            {{output[0].shape,at::kFloat,0}},parameters.data(),sizeof(parameters)}).at(0));
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        const auto out=metadata(s);return habana::PartialOutputMetaDataVector{{out[0].dtype,out[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t) {
        return std::make_shared<Control>(d,t);
    });return true;
}();
template<bool Meta>at::Tensor run(const at::Tensor& x,const at::Tensor& w,double eps) {
    const at::Stack s{x,w,eps};const auto out=metadata(s);
    if(Meta)return at::empty(out[0].shape,x.options().dtype(at::kFloat));
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return op.execute(s).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mhc_control_reuse_gaudi2(Tensor x, Tensor weight, float eps) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_mhc_control_reuse_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_mhc_control_reuse_gaudi2",run<true>);}
