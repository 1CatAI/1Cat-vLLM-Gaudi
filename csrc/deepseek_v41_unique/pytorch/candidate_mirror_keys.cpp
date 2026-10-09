// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_candidate_mirror_keys_gaudi2";
constexpr auto guid="custom_deepseek_v41_candidate_mirror_keys_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto& keys=s.at(0).toTensor();const auto& blocks=s.at(1).toTensor();
    TORCH_CHECK(keys.dim()==2 && keys.size(1)==128 && keys.size(0)>0 && keys.size(0)<=INT32_MAX
                && keys.scalar_type()==at::kBFloat16 && keys.is_contiguous()
                && blocks.dim()==2 && blocks.size(0)>=1 && blocks.size(0)<=6
                && blocks.size(1)>0 && blocks.size(1)<=2048 && blocks.size(1)%8==0
                && blocks.scalar_type()==at::kInt && blocks.is_contiguous()
                && keys.device()==blocks.device(),"Candidate keys require decoded BF16[N,128], I32[C1-C6,8..2048]");
    return {{at::kInt,{blocks.size(0),blocks.size(1)*8}},
            {at::kBFloat16,{blocks.size(0),blocks.size(1)*8,128}}};
}
class Gather final:public habana::OpBackend {
public:
    Gather(int d,c10::ScalarType t):OpBackend(d,guid,t,{0,1},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        const auto m=metadata(s);
        auto out=BuildNode(this,g,{guid,{syn_in(0),syn_in(1)},
            {{m[0].shape,m[0].dtype,0},{m[1].shape,m[1].dtype,1}}});
        syn_out(0)=std::move(out[0]);syn_out(1)=std::move(out[1]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for(const auto& m:metadata(s))out.push_back({m.dtype,m.shape});return out;
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Gather>(d,t);});
    return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor> run(const at::Tensor& keys,const at::Tensor& blocks) {
    const at::Stack s{keys,blocks};const auto m=metadata(s);
    if constexpr(Meta)return {at::empty(m[0].shape,blocks.options()),at::empty(m[1].shape,keys.options())};
    TORCH_CHECK(registered && keys.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto out=op.execute(s);return {out[0],out[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_candidate_mirror_keys_gaudi2(Tensor keys, Tensor blocks) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_candidate_mirror_keys_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_candidate_mirror_keys_gaudi2",run<true>);}
