// SPDX-License-Identifier: Apache-2.0
// Functional capability probe; this schema is never selected by serving.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_mixed_dense_probe_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto x=s[0].toTensor(),w=s[1].toTensor();
    TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.scalar_type()==at::kBFloat16 &&
        w.scalar_type()==at::ScalarType::Float8_e4m3fn && x.size(0)>=2 && x.size(0)<=6 &&
        x.size(1)==w.size(1) && x.is_contiguous() && w.is_contiguous() &&
        x.device()==w.device() && !x.requires_grad() && !w.requires_grad(),"Invalid mixed capability operands");
    return {{at::kFloat,{x.size(0),w.size(0)}}};
}
class Probe final:public habana::OpBackend {
public:
    Probe(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_mixed_dense_probe"),t,{0},{},{},false) {
        SetOutputMetaFn(meta);
    }
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        const auto m=meta(s);synGEMMParams p{false,true};
        syn_out(0)=std::move(BuildNode(this,g,{"gemm",{syn_in(0),syn_in(1)},
            {{m[0].shape,at::kFloat,0}},&p,sizeof(p)})[0]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(name,"gemm",[](const at::Stack& s) {
        const auto m=meta(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(name,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Probe>(d,t);});
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& w) {
    const auto m=meta({x,w});if(Meta)return at::empty(m[0].shape,x.options().dtype(at::kFloat));
    TORCH_CHECK(ready && x.device().type()==at::kHPU);
    auto d=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return d.execute({x,w})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {m.def("custom_deepseek_v41_mixed_dense_probe_gaudi2(Tensor x,Tensor w)->Tensor");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_mixed_dense_probe_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_mixed_dense_probe_gaudi2",run<true>);}
