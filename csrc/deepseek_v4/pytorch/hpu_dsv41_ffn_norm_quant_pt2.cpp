// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kSchema =
    "custom_op::custom_deepseek_v41_ffn_norm_quant_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_ffn_norm_quant_gaudi2";
constexpr auto kDualSchema = "custom_op::custom_deepseek_v41_ffn_norm_dual_quant_gaudi2";
constexpr auto kDualGuid = "custom_deepseek_v41_ffn_norm_dual_quant_gaudi2";
constexpr auto kBf16Schema = "custom_op::custom_deepseek_v41_ffn_norm_dual_bf16_quant_gaudi2";
constexpr auto kBf16Guid = "custom_deepseek_v41_ffn_norm_dual_bf16_quant_gaudi2";
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
using DualOutputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
struct Params { float epsilon; float inverse_width; };

void contract(const at::Tensor& x, const at::Tensor& weight, double epsilon) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 16384 &&
                x.size(1) == 5120 &&
                weight.sizes() == at::IntArrayRef({5120}) &&
                std::isnormal(static_cast<float>(epsilon)) && epsilon > 0,
                "V4.1 FFN norm/quant requires [1..16384,5120], weight [5120], positive normal epsilon");
    for (const auto& value : {x, weight})
        TORCH_CHECK(value.is_contiguous() && !value.requires_grad() &&
                    value.device() == x.device() &&
                    value.scalar_type() == at::kBFloat16,
                    "V4.1 FFN norm/quant requires matching contiguous inference BF16 tensors");
}

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto& x = stack.at(0).toTensor();
    contract(x, stack.at(1).toTensor(), stack.at(2).toDouble());
    return {{at::kBFloat16, x.sizes().vec()},
            {at::ScalarType::Float8_e4m3fn, x.sizes().vec()},
            {at::kFloat, {x.size(0), 1}}};
}

class FfnNormQuant final : public habana::OpBackend {
public:
    FfnNormQuant(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_ffn_norm_quant"),
                    dtype, {0, 1, 2}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,
                 const at::Stack& stack) override {
        const auto meta = metadata(stack);
        Params scalar{float(stack.at(2).toDouble()), 1.0f / 5120.0f};
        auto result = BuildNode(this, graph,
            {kGuid, {syn_in(0), syn_in(1)},
             {{meta.at(0).shape, at::kBFloat16, 0},
              {meta.at(1).shape, at::ScalarType::Float8_e4m3fn, 1},
              {meta.at(2).shape, at::kFloat, 2}},
             &scalar, sizeof(scalar)});
        syn_out(0) = std::move(result.at(0));
        syn_out(1) = std::move(result.at(1));
        syn_out(2) = std::move(result.at(2));
    }
};

template<bool Bf16 = false>
class FfnNormDualQuant final : public habana::OpBackend {
public:
    FfnNormDualQuant(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_ffn_norm_dual_quant"),
                    dtype, {0, 1, 2, 3, 4}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& stack) {
            auto meta = metadata(stack);
            auto q = meta.at(1), scale = meta.at(2);
            meta.push_back(q);
            meta.push_back(scale);
            return meta;
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto meta = metadata(stack);
        Params scalar{float(stack.at(2).toDouble()), 1.0f / 5120.0f};
        auto result = BuildNode(this, graph,
            {Bf16 ? kBf16Guid : kDualGuid, {syn_in(0), syn_in(1)},
             {{meta.at(0).shape, at::kBFloat16, 0},
              {meta.at(1).shape, at::ScalarType::Float8_e4m3fn, 1},
              {meta.at(2).shape, at::kFloat, 2},
              {meta.at(1).shape, at::ScalarType::Float8_e4m3fn, 3},
              {meta.at(2).shape, at::kFloat, 4}}, &scalar, sizeof(scalar)});
        for (size_t i = 0; i < 5; ++i) syn_out(i) = std::move(result.at(i));
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSchema, kGuid,
        [](const at::Stack& stack) {
            const auto meta = metadata(stack);
            return habana::PartialOutputMetaDataVector{
                {meta.at(0).dtype, meta.at(0).shape},
                {meta.at(1).dtype, meta.at(1).shape},
                {meta.at(2).dtype, meta.at(2).shape}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(Params);
            return std::make_shared<Params>(Params{
                float(stack.at(2).toDouble()), 1.0f / 5120.0f});
        });
    habana::KernelRegistry().add(
        kSchema, [](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<FfnNormQuant>(device, dtype);
        });
    for (const auto* name : {kDualSchema, kBf16Schema}) {
    habana::custom_op::registerUserCustomOp(name, name + 11,
        [](const at::Stack& stack) {
            const auto meta = metadata(stack);
            return habana::PartialOutputMetaDataVector{
                {meta.at(0).dtype, meta.at(0).shape},
                {meta.at(1).dtype, meta.at(1).shape},
                {meta.at(2).dtype, meta.at(2).shape},
                {meta.at(1).dtype, meta.at(1).shape},
                {meta.at(2).dtype, meta.at(2).shape}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(Params);
            return std::make_shared<Params>(Params{float(stack.at(2).toDouble()), 1.0f / 5120.0f});
        });
    }
    habana::KernelRegistry().add(kBf16Schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<FfnNormDualQuant<true>>(device, dtype);
    });
    habana::KernelRegistry().add(kDualSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<FfnNormDualQuant<false>>(device, dtype);
    });
    return true;
}();

template<bool Meta>
Outputs run(const at::Tensor& x, const at::Tensor& weight, double epsilon) {
    const at::Stack stack{x, weight, epsilon};
    const auto meta = metadata(stack);
    if (Meta)
        return {at::empty(meta.at(0).shape, x.options().dtype(at::kBFloat16)),
                at::empty(meta.at(1).shape,
                          x.options().dtype(at::ScalarType::Float8_e4m3fn)),
                at::empty(meta.at(2).shape, x.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::
        getUserCustomOpDescriptor(kSchema);
    auto output = descriptor.execute(stack);
    return {output.at(0), output.at(1), output.at(2)};
}
template<bool Meta, bool Bf16 = false>
DualOutputs run_dual(const at::Tensor& x, const at::Tensor& weight, double epsilon) {
    const at::Stack stack{x, weight, epsilon};
    const auto meta = metadata(stack);
    if constexpr (Meta) {
        return {at::empty(meta[0].shape, x.options().dtype(meta[0].dtype)),
                at::empty(meta[1].shape, x.options().dtype(meta[1].dtype)),
                at::empty(meta[2].shape, x.options().dtype(meta[2].dtype)),
                at::empty(meta[1].shape, x.options().dtype(meta[1].dtype)),
                at::empty(meta[2].shape, x.options().dtype(meta[2].dtype))};
    }
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Bf16 ? kBf16Schema : kDualSchema);
    auto output = descriptor.execute(stack);
    return {output.at(0), output.at(1), output.at(2), output.at(3), output.at(4)};
}

}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_ffn_norm_dual_bf16_quant_gaudi2(Tensor value, Tensor weight, float epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_ffn_norm_dual_quant_gaudi2(Tensor value, Tensor weight, float epsilon) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_ffn_norm_quant_gaudi2(Tensor value, Tensor weight, float epsilon) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_ffn_norm_dual_bf16_quant_gaudi2", run_dual<false, true>);
    m.impl("custom_deepseek_v41_ffn_norm_dual_quant_gaudi2", run_dual<false>);
    m.impl("custom_deepseek_v41_ffn_norm_quant_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_ffn_norm_dual_bf16_quant_gaudi2", run_dual<true, true>);
    m.impl("custom_deepseek_v41_ffn_norm_dual_quant_gaudi2", run_dual<true>);
    m.impl("custom_deepseek_v41_ffn_norm_quant_gaudi2", run<true>);
}
