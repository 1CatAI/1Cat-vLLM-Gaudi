// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_dense_fp8_gaudi2";
constexpr auto kQuantSchema = "custom_op::custom_deepseek_v41_dense_quant_gaudi2";
constexpr auto kQuant = "custom_deepseek_v41_dense_quant_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& stack, bool quant) {
    const auto x = stack.at(0).toTensor();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8192 &&
                (x.size(1) == 1280 || x.size(1) == 4096), "Dense FP8 requires BF16 [T,1280|4096]");
    for (const auto& item : stack) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "Dense FP8 requires matching contiguous inference tensors");
    }
    if (quant) return {{at::ScalarType::Float8_e4m3fn, x.sizes().vec()}, {at::kFloat, {x.size(0), 1}}};
    const auto w = stack.at(1).toTensor(), scale = stack.at(2).toTensor();
    const auto n = x.size(1) == 1280 ? 16384 : 5120;
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.sizes() == at::IntArrayRef({n,x.size(1)}) &&
                scale.scalar_type() == at::kFloat && scale.sizes() == at::IntArrayRef({1,n}),
                "Dense FP8 requires prepared Gaudi2 [N,K] weights and F32 [1,N] channel scales");
    return {{at::kBFloat16, {x.size(0),n}}};
}
class Dense final : public habana::OpBackend {
    bool quant_;
public:
    Dense(int device, c10::ScalarType dtype, bool quant)
        : OpBackend(device, NO_TPC + std::string("dsv41_dense_fp8"), dtype,
                    quant ? std::vector<int>{0,1} : std::vector<int>{0}, {}, {}, false), quant_(quant) {
        SetOutputMetaFn([quant](const at::Stack& stack) { return meta(stack, quant); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = meta(stack, quant_);
        const auto shape = stack.at(0).toTensor().sizes().vec();
        if (quant_) {
            auto q = BuildNode(this, graph, {kQuant, {syn_in(0)},
                {{shape, at::ScalarType::Float8_e4m3fn, 0}, {{shape[0],1}, at::kFloat, 1}}});
            syn_out(0) = std::move(q.at(0)); syn_out(1) = std::move(q.at(1)); return;
        }
        auto q = BuildNode(this, graph, {kQuant, {syn_in(0)},
            {{shape, at::ScalarType::Float8_e4m3fn}, {{shape[0],1}, at::kFloat}}});
        synGEMMParams params{false,true};
        auto product = BuildNode(this, graph, {"gemm", {q.at(0).get(), syn_in(1)},
            {{output.at(0).shape, at::kFloat}}, &params, sizeof(params)});
        auto scaled = BuildNode(this, graph, {"custom_deepseek_v41_dense_scale_gaudi2",
            {product.at(0).get(), syn_in(2), q.at(1).get()}, {{output.at(0).shape, at::kBFloat16, 0}}});
        syn_out(0) = std::move(scaled.at(0));
    }
};
const bool registered = [] {
    for (bool quant : {false,true}) {
        const auto schema = quant ? kQuantSchema : kSchema;
        habana::custom_op::registerUserCustomOp(schema, kQuant, [quant](const at::Stack& stack) {
            habana::PartialOutputMetaDataVector result;
            for (const auto& x : meta(stack, quant)) result.push_back({x.dtype,x.shape});
            return result;
        }, nullptr);
        habana::KernelRegistry().add(schema, [quant](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<Dense>(device,dtype,quant);
        });
    }
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s) {
    const auto output = meta({x,w,s},false);
    if (Meta) return at::empty(output.at(0).shape,x.options());
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    return descriptor.execute({x,w,s}).at(0);
}
template<bool Meta> Pair quant(const at::Tensor& x) {
    const auto output = meta({x},true);
    if (Meta) return {at::empty(output.at(0).shape,x.options().dtype(at::ScalarType::Float8_e4m3fn)),
                      at::empty(output.at(1).shape,x.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kQuantSchema);
    auto q = descriptor.execute({x});
    return {q.at(0),q.at(1)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_dense_fp8_gaudi2(Tensor input, Tensor weight, Tensor channel_scale) -> Tensor");
    m.def("custom_deepseek_v41_dense_quant_gaudi2(Tensor input) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_dense_fp8_gaudi2",run<false>);
    m.impl("custom_deepseek_v41_dense_quant_gaudi2",quant<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_dense_fp8_gaudi2",run<true>);
    m.impl("custom_deepseek_v41_dense_quant_gaudi2",quant<true>);
}
