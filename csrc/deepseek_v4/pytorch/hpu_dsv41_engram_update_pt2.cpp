// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_engram_update_bf16_gaudi2";
constexpr auto guid="custom_deepseek_v41_engram_update_bf16_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    TORCH_CHECK(stack.size()==6,"Engram update requires five tensors and epsilon");
    const auto x=stack.at(0).toTensor();const auto kv=stack.at(1).toTensor();
    const auto q=stack.at(2).toTensor();const auto k=stack.at(3).toTensor();const auto mask=stack.at(4).toTensor();
    const auto rows=x.dim()==3?x.size(0):0;
    TORCH_CHECK(rows>=1 && rows<=6 && x.sizes()==at::IntArrayRef({rows,4,5120}) &&
                kv.sizes()==at::IntArrayRef({rows,25600}) && q.sizes()==at::IntArrayRef({4,5120}) &&
                k.sizes()==q.sizes() && mask.sizes()==at::IntArrayRef({rows}),"Engram update shape contract");
    for (const auto& t:{x,kv,q,k})
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device()==x.device() &&
                    t.scalar_type()==at::kBFloat16,"Engram update requires contiguous inference BF16 operands");
    const auto eps=static_cast<float>(stack.at(5).toDouble());
    TORCH_CHECK(mask.is_contiguous() && mask.scalar_type()==at::kBool && mask.device()==x.device() &&
                std::isnormal(eps) && eps>0,"Engram update mask/epsilon contract");
    return {{at::kBFloat16,x.sizes().vec()}};
}
class EngramUpdate final:public habana::OpBackend {
public:
    EngramUpdate(int device,c10::ScalarType dtype)
        :OpBackend(device,NO_TPC+std::string("dsv41_engram_update"),dtype,{0},{},{},false) {SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& stack) override {
        const auto meta=metadata(stack);auto eps=static_cast<float>(stack.at(5).toDouble());
        auto result=BuildNode(this,graph,{guid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4)},
                    {{meta.at(0).shape,at::kBFloat16,0}},&eps,sizeof(eps)});
        syn_out(0)=std::move(result.at(0));
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& stack) {
        const auto meta=metadata(stack);return habana::PartialOutputMetaDataVector{{meta.at(0).dtype,meta.at(0).shape}};
    },[](const at::Stack& stack,size_t& size)->std::shared_ptr<void> {
        size=sizeof(float);return std::make_shared<float>(static_cast<float>(stack.at(5).toDouble()));
    });
    habana::KernelRegistry().add(schema,[](synDeviceId device,c10::ScalarType dtype) {
        return std::make_shared<EngramUpdate>(device,dtype);
    });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x,const at::Tensor& kv,const at::Tensor& q,
                                  const at::Tensor& k,const at::Tensor& mask,double epsilon) {
    const at::Stack stack{x,kv,q,k,mask,epsilon};const auto meta=metadata(stack);
    if (Meta) return at::empty(meta.at(0).shape,x.options());
    TORCH_CHECK(registered && x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_engram_update_bf16_gaudi2(Tensor residual, Tensor kv, Tensor q_weight, Tensor k_weight, Tensor active_mask, float epsilon) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_engram_update_bf16_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_engram_update_bf16_gaudi2",run<true>);}
