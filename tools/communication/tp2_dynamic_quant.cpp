// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <limits>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::tp2_dynamic_quant";
constexpr auto guid = "tp2_dynamic_quant_bf16_gaudi2";

void validate(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous() && !x.requires_grad(),
                "native TP2 quant requires contiguous rank-2 inference BF16 input");
    TORCH_CHECK(x.size(0) == 1 && x.size(0) <= std::numeric_limits<int32_t>::max() &&
                x.size(1) >= 256 && x.size(1) <= 17408 && x.size(1) % 128 == 0,
                "native TP2 quant requires B=1 and D in [256,17408], divisible by 128");
}

class Tp2DynamicQuant final : public habana::OpBackend {
public:
    Tp2DynamicQuant(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("tp2_dynamic_quant"), dtype, {0, 1}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& stack) {
            const auto& x = stack.at(0).toTensor();
            validate(x);
            return habana::OutputMetaDataVector{
                {at::ScalarType::Float8_e4m3fn, {x.size(0), x.size(1)}}, {at::kFloat, {x.size(0), 1}}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto& x = stack.at(0).toTensor();
        validate(x);
        const std::vector<int64_t> shape{x.size(0), x.size(1)};
        auto result = BuildNode(this, graph, {guid, {syn_in(0)},
            {{shape, at::ScalarType::Float8_e4m3fn, 0}, {{x.size(0), 1}, at::kFloat, 1}}});
        syn_out(0) = std::move(result.at(0));
        syn_out(1) = std::move(result.at(1));
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, guid, [](const at::Stack& stack) {
        const auto& x = stack.at(0).toTensor();
        validate(x);
        return habana::PartialOutputMetaDataVector{
            {at::ScalarType::Float8_e4m3fn, {x.size(0), x.size(1)}},
            {at::kFloat, {x.size(0), 1}}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Tp2DynamicQuant>(device, dtype);
    });
    return true;
}();

std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& x) {
    validate(x);
    TORCH_CHECK(x.device().type() == at::kHPU && registered, "native TP2 quant requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result = descriptor.execute({x});
    return {result.at(0), result.at(1)};
}

std::tuple<at::Tensor, at::Tensor> meta(const at::Tensor& x) {
    validate(x);
    return {at::empty({x.size(0), x.size(1)}, x.options().dtype(at::ScalarType::Float8_e4m3fn)),
            at::empty({x.size(0), 1}, x.options().dtype(at::kFloat))};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("tp2_dynamic_quant(Tensor input) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("tp2_dynamic_quant", run); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("tp2_dynamic_quant", meta); }
