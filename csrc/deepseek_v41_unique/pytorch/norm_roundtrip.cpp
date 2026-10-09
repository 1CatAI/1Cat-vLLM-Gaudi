// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_norm_roundtrip_bf16_gaudi2";
struct Params {float epsilon,inverse_width;};
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size()==3,"Norm roundtrip requires value, weight and epsilon");
    const auto x=s[0].toTensor(),w=s[1].toTensor();
    TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 && (x.size(1)==1280 || x.size(1)==5120) &&
                w.sizes()==at::IntArrayRef({x.size(1)}) && x.scalar_type()==at::kBFloat16 &&
                w.scalar_type()==at::kBFloat16 && x.device()==w.device() && x.is_contiguous() &&
                w.is_contiguous() && !x.requires_grad() && !w.requires_grad() &&
                s[2].toDouble()>0 && std::isnormal(float(s[2].toDouble())),
                "Norm roundtrip requires C1-C6 checkpoint rows and positive normal epsilon");
    habana::OutputMetaData m;m.shape=x.sizes().vec();m.dtype=at::kBFloat16;
    return {m,m};
}
class Norm:public habana::OpBackend {
public:
    Norm(synDeviceId d,c10::ScalarType t):OpBackend(d,"custom_deepseek_v41_norm_roundtrip_gaudi2",t,
                                                {0,1},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        const auto shape=s[0].toTensor().sizes();Params p{float(s[2].toDouble()),1.f/shape[1]};
#if DSV41_NORM_ROUNDTRIP_TILED
        auto out=BuildNode(this,g,{"custom_deepseek_v41_norm_roundtrip_tiled_gaudi2",{syn_in(0),syn_in(1)},
            {{shape.vec(),at::kBFloat16,0},{shape.vec(),at::kBFloat16,1}},&p,sizeof(p)});
#else
        auto statistics=BuildNode(this,g,{"custom_deepseek_v41_norm_statistics_gaudi2",{syn_in(0)},
            {{{shape[0],1},at::kFloat}},&p,sizeof(p)});
        auto out=BuildNode(this,g,{"custom_deepseek_v41_norm_roundtrip_gaudi2",
            {syn_in(0),syn_in(1),statistics[0].get()},
            {{shape.vec(),at::kBFloat16,0},{shape.vec(),at::kBFloat16,1}}});
#endif
        syn_out(0)=std::move(out[0]);syn_out(1)=std::move(out[1]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,"custom_deepseek_v41_norm_roundtrip_gaudi2",
        [](const at::Stack& s){const auto m=metadata(s);
            return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape},{m[1].dtype,m[1].shape}};},nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Norm>(d,t);});
    return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor> run(const at::Tensor& x,const at::Tensor& w,double epsilon) {
    const auto m=metadata({x,w,epsilon});
    if constexpr(Meta)return {at::empty(m[0].shape,x.options()),at::empty(m[1].shape,x.options())};
    TORCH_CHECK(ready && x.device().type()==at::kHPU,"Norm roundtrip requires HPU");
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    auto out=descriptor.execute({x,w,epsilon});return {out[0],out[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {m.def("custom_deepseek_v41_norm_roundtrip_bf16_gaudi2(Tensor x, Tensor weight, float epsilon) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_norm_roundtrip_bf16_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_norm_roundtrip_bf16_gaudi2",run<true>);}
