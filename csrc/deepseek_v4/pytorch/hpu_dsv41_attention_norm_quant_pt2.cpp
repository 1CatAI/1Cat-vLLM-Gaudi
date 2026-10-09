// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"

#ifndef DSV41_ATTN_NORM_QUANT_OP
#define DSV41_ATTN_NORM_QUANT_OP "custom_deepseek_v41_attention_norm_quant_gaudi2"
#endif
namespace {
constexpr auto kSchema = "custom_op::" DSV41_ATTN_NORM_QUANT_OP;
constexpr auto kGuid = DSV41_ATTN_NORM_QUANT_OP;
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
struct Params { float epsilon; float inverse_width; };

void contract(const at::Tensor& x, const at::Tensor& weight, double epsilon) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 512 &&
                x.size(1) == 5120 &&
                weight.sizes() == at::IntArrayRef({5120}) &&
                std::isnormal(static_cast<float>(epsilon)) && epsilon > 0,
                "V4.1 attention norm/quant requires [1..512,5120], weight [5120], positive normal epsilon");
    for (const auto& value : {x, weight})
        TORCH_CHECK(value.is_contiguous() && !value.requires_grad() &&
                    value.device() == x.device() &&
                    value.scalar_type() == at::kBFloat16,
                    "V4.1 attention norm/quant requires matching contiguous inference BF16 tensors");
}

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto& x = stack.at(0).toTensor();
    contract(x, stack.at(1).toTensor(), stack.at(2).toDouble());
    return {{at::kBFloat16, x.sizes().vec()},
            {at::ScalarType::Float8_e4m3fn, x.sizes().vec()},
            {at::kFloat, {x.size(0), 1}}};
}

class AttentionNormQuant final : public habana::OpBackend {
public:
    AttentionNormQuant(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_attention_norm_quant"),
                    dtype, {0, 1, 2}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,
                 const at::Stack& stack) override {
        const auto meta = metadata(stack);
        Params scalar{float(stack.at(2).toDouble()), 1.0f / 5120.0f};
        auto result = BuildNode(this, graph,
            {kGuid, {syn_in(0), syn_in(1)},
             {{meta.at(1).shape, at::ScalarType::Float8_e4m3fn, 1},
              {meta.at(2).shape, at::kFloat, 2},
              {meta.at(0).shape, at::kBFloat16, 0}},
             &scalar, sizeof(scalar)});
        syn_out(0) = std::move(result.at(2));
        syn_out(1) = std::move(result.at(0));
        syn_out(2) = std::move(result.at(1));
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSchema, kGuid,
        [](const at::Stack& stack) {
            const auto meta = metadata(stack);
            return habana::PartialOutputMetaDataVector{
                {meta.at(1).dtype, meta.at(1).shape},
                {meta.at(2).dtype, meta.at(2).shape},
                {meta.at(0).dtype, meta.at(0).shape}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(Params);
            return std::make_shared<Params>(Params{
                float(stack.at(2).toDouble()), 1.0f / 5120.0f});
        });
    habana::KernelRegistry().add(
        kSchema, [](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<AttentionNormQuant>(device, dtype);
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
    return {output.at(2), output.at(0), output.at(1)};
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def(DSV41_ATTN_NORM_QUANT_OP "(Tensor value, Tensor weight, float epsilon) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl(DSV41_ATTN_NORM_QUANT_OP, run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl(DSV41_ATTN_NORM_QUANT_OP, run<true>);
}
