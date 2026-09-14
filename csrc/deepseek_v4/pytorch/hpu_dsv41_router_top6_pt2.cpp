// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_router_top6_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_router_top6_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto scores = stack.at(0).toTensor(), text = stack.at(1).toTensor();
    const auto image = stack.at(2).toTensor(), mask = stack.at(3).toTensor();
    TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.size(0) > 0 &&
                scores.size(0) <= 512 && scores.size(1) == 384 && text.scalar_type() == at::kFloat &&
                image.scalar_type() == at::kFloat && text.sizes() == at::IntArrayRef({384}) &&
                image.sizes() == text.sizes() && mask.scalar_type() == at::kBool &&
                mask.sizes() == at::IntArrayRef({scores.size(0)}), "V4.1 router requires F32 [T,384], two F32 biases and bool [T]");
    for (const auto& v : stack) {
        const auto t = v.toTensor();
        TORCH_CHECK(t.is_contiguous() && t.device() == scores.device() && !t.requires_grad());
    }
    return {{at::kInt, {scores.size(0), 6}}, {at::kFloat, {scores.size(0), 6}}};
}
class Router final : public habana::OpBackend {
public:
    Router(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_router_top6"), dtype, {0,1}, {}, {}, false) {SetOutputMetaFn(metadata);}
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto out = metadata(stack);
        auto result = BuildNode(this, graph, {kGuid, {syn_in(0),syn_in(1),syn_in(2),syn_in(3)},
            {{out.at(0).shape,at::kInt,0}, {out.at(1).shape,at::kFloat,1}}});
        syn_out(0) = std::move(result.at(0)); syn_out(1) = std::move(result.at(1));
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kSchema, kGuid, [](const at::Stack& stack) {
        const auto out = metadata(stack);
        return habana::PartialOutputMetaDataVector{{at::kInt,out.at(0).shape},{at::kFloat,out.at(1).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kSchema, [](synDeviceId device, c10::ScalarType dtype) {return std::make_shared<Router>(device,dtype);});
    return true;
}();
template<bool Meta> Pair run(const at::Tensor& x, const at::Tensor& text, const at::Tensor& image, const at::Tensor& mask) {
    const auto out = metadata({x,text,image,mask});
    if (Meta) return {at::empty(out.at(0).shape,x.options().dtype(at::kInt)),at::empty(out.at(1).shape,x.options())};
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    auto result = descriptor.execute({x,text,image,mask}); return {result.at(0),result.at(1)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_router_top6_gaudi2(Tensor scores, Tensor text_bias, Tensor image_bias, Tensor image_mask) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {m.impl("custom_deepseek_v41_router_top6_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {m.impl("custom_deepseek_v41_router_top6_gaudi2",run<true>);}
