// SPDX-License-Identifier: Apache-2.0
// Keep the exact dtype reinterpretation inside a compiled recipe. The shared
// replay executor intentionally rejects host-level dtype-changing aliases.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_sampling_wire_view_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto input=stack.at(0).toTensor();const bool unpack=stack.at(1).toBool();
    TORCH_CHECK(input.dim()==2 && input.size(0)>0 && input.is_contiguous() && !input.requires_grad(),
                "Official sampling wire requires a contiguous inference matrix");
    TORCH_CHECK(input.scalar_type()==(unpack?at::kBFloat16:at::kFloat) &&
                input.size(1)>0 && (!unpack || input.size(1)%2==0),
                "Official sampling wire reinterprets complete FP32 words only");
    return {{unpack?at::kFloat:at::kBFloat16,{input.size(0),unpack?input.size(1)/2:input.size(1)*2}}};
}
class WireView final:public habana::OpBackend {
 public:
    WireView(int device,c10::ScalarType dtype):OpBackend(device,NO_TPC+std::string("sampling_wire_view"),
                                                      dtype,{0},{},{},false) {SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& graph,const at::Stack& stack) override {
        const auto output=metadata(stack).at(0);
        // The same Synapse node as Bridge ViewDtype::AddNode. No numeric cast
        // or packing arithmetic is introduced, including for NaN payloads.
        syn_out(0)=std::move(BuildNode(this,graph,{"reinterpret_cast",{syn_in(0)},
                           {{output.shape,output.dtype,0}}}).at(0));
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,"reinterpret_cast",[](const at::Stack& stack) {
        const auto output=metadata(stack).at(0);
        return habana::PartialOutputMetaDataVector{{output.dtype,output.shape}};
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId device,c10::ScalarType dtype) {
        return std::make_shared<WireView>(device,dtype);
    });return true;
}();
template<bool Meta> at::Tensor execute(const at::Tensor& input,bool unpack) {
    const at::Stack stack{input,unpack};const auto output=metadata(stack).at(0);
    if(Meta)return at::empty(output.shape,input.options().dtype(output.dtype));
    TORCH_CHECK(registered && input.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_sampling_wire_view_gaudi2(Tensor input, bool unpack) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_sampling_wire_view_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_sampling_wire_view_gaudi2",execute<true>);}
