// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_future_epoch_gaudi2";
constexpr auto ordered="custom_op::custom_deepseek_v41_future_epoch_ordered_gaudi2";
constexpr auto guid="custom_deepseek_v41_future_epoch_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto value=stack[0].toTensor(),flags=stack[1].toTensor();
    TORCH_CHECK((value.dim()==2||value.dim()==3)&&value.size(0)>=1&&value.size(0)<=6&&
        value.scalar_type()==at::kBFloat16&&value.numel()%128==0&&value.numel()<=6*20480&&
        flags.scalar_type()==at::kInt&&flags.sizes()==at::IntArrayRef({1<<24}));
    TORCH_CHECK(value.device()==flags.device()&&value.is_contiguous()&&flags.is_contiguous()&&
        !value.requires_grad()&&!flags.requires_grad());
    return {{at::kBFloat16,value.sizes().vec()}};
}
class Advance final:public habana::OpBackend {
public:
    Advance(int device,c10::ScalarType type):OpBackend(device,NO_TPC+std::string("dsv41_future_epoch"),
        type,{0},{},{},false){SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& stack)override {
        const auto shape=metadata(stack)[0].shape;const auto elements=stack[0].toTensor().numel();
        auto flat=BuildReshape(this,graph,syn_in(0),{1,elements},at::kBFloat16);
        auto copied=BuildNode(this,graph,{guid,{flat.get(),syn_in(1)},{{{1,elements},at::kBFloat16}}});
        syn_out(0)=BuildReshape(this,graph,copied[0].get(),shape,at::kBFloat16,0);
    }
};
const bool registered=[] {
    for(const auto name:{schema,ordered}) {
        habana::custom_op::registerUserCustomOp(name,guid,[](const at::Stack& stack){
            return habana::PartialOutputMetaDataVector{{at::kBFloat16,metadata(stack)[0].shape}};
        },nullptr);
        habana::KernelRegistry().add(name,[](synDeviceId device,c10::ScalarType type){
            return std::make_shared<Advance>(device,type);
        });
    }
    return true;
}();
template<bool Meta,bool Ordered>at::Tensor run(const at::Tensor& value,const at::Tensor& flags) {
    const at::Stack stack{value,flags};const auto shape=metadata(stack)[0].shape;
    if constexpr(Meta)return at::empty(shape,value.options());
    TORCH_CHECK(registered&&value.device().type()==at::kHPU);
    auto operation=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered?ordered:schema);
    return operation.execute(stack)[0];
}
at::Tensor unwrap(const at::Tensor& value) {
    if(!at::functionalization::impl::isFunctionalTensor(value))return value;
    at::functionalization::impl::sync(value);
    return at::functionalization::impl::from_functional_tensor(value);
}
// Same owned-state protocol as the production state_rows writer: expose
// mutation on the public schema, and retain the original allocation in the
// ordered backend node. Transport flags cannot be cloned or copied back
// after the NIC has already published its completion into that allocation.
at::Tensor functionalize(const at::Tensor& value,const at::Tensor& flags) {
    auto input=unwrap(value),state=unwrap(flags);
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(ordered,"")
        .typed<at::Tensor(const at::Tensor&,const at::Tensor&)>();
    at::Tensor copied;
    {at::AutoDispatchSkipFunctionalize guard;copied=handle.call(input,state);}
    at::functionalization::impl::replace_(flags,state);
    at::functionalization::impl::commit_update(flags);
    at::functionalization::impl::sync(flags);
    return copied;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,module){
    module.def("custom_deepseek_v41_future_epoch_gaudi2(Tensor value, Tensor(a!) flags) -> Tensor");
    module.def("custom_deepseek_v41_future_epoch_ordered_gaudi2(Tensor value, Tensor flags) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,module){
    module.impl("custom_deepseek_v41_future_epoch_gaudi2",run<false,false>);
    module.impl("custom_deepseek_v41_future_epoch_ordered_gaudi2",run<false,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,module){
    module.impl("custom_deepseek_v41_future_epoch_gaudi2",run<true,false>);
    module.impl("custom_deepseek_v41_future_epoch_ordered_gaudi2",run<true,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Functionalize,module){
    module.impl("custom_deepseek_v41_future_epoch_gaudi2",functionalize);
}
