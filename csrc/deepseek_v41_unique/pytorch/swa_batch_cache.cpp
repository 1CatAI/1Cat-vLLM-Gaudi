// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto guid="custom_deepseek_v41_swa_batch_cache_write_gaudi2";
constexpr auto schema="custom_op::custom_deepseek_v41_swa_batch_cache_ordered_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto cache=s[0].toTensor(),value=s[1].toTensor(),positions=s[2].toTensor(),decoded=s[3].toTensor();
    TORCH_CHECK(value.dim()==2 && value.size(0)>=1 && value.size(0)<=6 && value.size(1)==512 &&
        value.scalar_type()==at::kBFloat16 && cache.scalar_type()==at::kByte &&
        cache.sizes()==at::IntArrayRef({256,528}) && decoded.scalar_type()==at::kBFloat16 &&
        decoded.sizes()==at::IntArrayRef({256,512}) && positions.scalar_type()==at::kInt &&
        positions.sizes()==at::IntArrayRef({value.size(0)}),"SWA cache requires canonical C1-C6 rows and rings");
    for(const auto& entry:s) {
        const auto tensor=entry.toTensor();
        TORCH_CHECK(tensor.device()==value.device() && tensor.is_contiguous() && !tensor.requires_grad(),
                    "SWA cache requires matching contiguous inference tensors");
    }
    return {{at::kInt,{value.size(0),16}}};
}
class Write final:public habana::OpBackend {
public:
    Write(int d,c10::ScalarType t):OpBackend(d,guid,t,{0},{},{},false){SetOutputMetaFn(meta);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=meta(s);
        syn_out(0)=std::move(BuildNode(this,graph,{guid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3)},
                                                {{m[0].shape,m[0].dtype,0}}})[0]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s){
        const auto m=meta(s);return habana::PartialOutputMetaDataVector{{m[0].dtype,m[0].shape}};
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Write>(d,t);});
    return true;
}();
template<bool Meta> at::Tensor write(const at::Tensor& cache,const at::Tensor& value,
                                   const at::Tensor& positions,const at::Tensor& decoded) {
    const at::Stack s{cache,value,positions,decoded};const auto m=meta(s);
    if(Meta)return at::empty(m[0].shape,value.options().dtype(m[0].dtype));
    TORCH_CHECK(ready && value.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(s)[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    // Match C1 ordered capture: stage-owned rings are mutable, and
    // the consumer retains the complete writer signal dependency.
    m.def("custom_deepseek_v41_swa_batch_cache_ordered_gaudi2(Tensor cache, Tensor value, Tensor positions, Tensor decoded) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_swa_batch_cache_ordered_gaudi2",write<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_swa_batch_cache_ordered_gaudi2",write<true>);}
