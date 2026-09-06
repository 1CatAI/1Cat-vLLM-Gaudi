// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <limits>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::flashinfer_gaudi_silu_mul_quant";
constexpr auto guid = "flashinfer_gaudi_silu_mul_quant_bf16_gaudi2";

void validate(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous() && !x.requires_grad(),
                "native SiLU-quant requires contiguous rank-2 inference BF16 input");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= std::numeric_limits<int32_t>::max() &&
                x.size(1) >= 512 && x.size(1) <= 34816 && x.size(1) % 256 == 0,
                "native SiLU-quant requires B>0 and D in [256,17408], divisible by 128");
}

class SiluQuant final : public habana::OpBackend {
public:
    SiluQuant(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("flashinfer_gaudi_silu_mul_quant"), dtype, {0, 1}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& stack) {
            const auto& x = stack.at(0).toTensor();
            validate(x);
            return habana::OutputMetaDataVector{
                {at::ScalarType::Float8_e4m3fn, {x.size(0), x.size(1) / 2}}, {at::kFloat, {x.size(0), 1}}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto& x = stack.at(0).toTensor();
        validate(x);
        const std::vector<int64_t> shape{x.size(0), x.size(1) / 2};
        synSplitParams split_params{};
        split_params.axis = 0;
        auto halves = BuildNode(this, graph, {"split", {syn_in(0)},
            {{shape, at::kBFloat16}, {shape, at::kBFloat16}}, &split_params, sizeof(split_params)});
        auto silu = BuildNode(this, graph, {"silu_fwd_bf16", {halves.at(0).get()}, {{shape, at::kBFloat16}}});
        auto result = BuildNode(this, graph, {guid, {syn_in(0), silu.at(0).get()},
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
            {at::ScalarType::Float8_e4m3fn, {x.size(0), x.size(1) / 2}},
            {at::kFloat, {x.size(0), 1}}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<SiluQuant>(device, dtype);
    });
    return true;
}();

std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& x) {
    validate(x);
    TORCH_CHECK(x.device().type() == at::kHPU && registered, "native SiLU-quant requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result = descriptor.execute({x});
    return {result.at(0), result.at(1)};
}

std::tuple<at::Tensor, at::Tensor> meta(const at::Tensor& x) {
    validate(x);
    return {at::empty({x.size(0), x.size(1) / 2}, x.options().dtype(at::ScalarType::Float8_e4m3fn)),
            at::empty({x.size(0), 1}, x.options().dtype(at::kFloat))};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("flashinfer_gaudi_silu_mul_quant(Tensor input) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("flashinfer_gaudi_silu_mul_quant", run); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("flashinfer_gaudi_silu_mul_quant", meta); }
