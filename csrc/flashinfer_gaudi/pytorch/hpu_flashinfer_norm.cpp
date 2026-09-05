// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include <limits>
#include "hpu_custom_op_pt2.h"
#include "../norm_quant_params.h"

namespace {
constexpr auto kSchema = "custom_op::flashinfer_gaudi_add_rmsnorm_quant";
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

void validate(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double epsilon) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(0) <= std::numeric_limits<int32_t>::max() &&
                x.size(1) >= 256 && x.size(1) <= 17408 && x.size(1) % 128 == 0,
                "Norm-quant requires [B,D], B>0 and D divisible by 128 in [256,17408]");
    TORCH_CHECK(residual.sizes() == x.sizes() && weight.dim() == 1 && weight.size(0) == x.size(1),
                "Norm-quant residual and weight shapes do not match input");
    TORCH_CHECK(std::isnormal(static_cast<float>(epsilon)) && epsilon > 0,
                "Norm-quant epsilon must be a positive normal FP32 value");
    for (const auto* tensor : {&x, &residual, &weight}) {
        TORCH_CHECK(tensor->scalar_type() == at::kBFloat16 && tensor->is_contiguous() && !tensor->requires_grad(),
                    "Norm-quant requires contiguous inference BF16 tensors");
        TORCH_CHECK(tensor->device() == x.device(), "Norm-quant inputs must use the same device");
    }
}

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSchema, kAddRmsNormQuantGuid,
        [](const at::Stack& stack) {
            const auto& x = stack.at(0).toTensor();
            validate(x, stack.at(1).toTensor(), stack.at(2).toTensor(), stack.at(3).toDouble());
            return habana::PartialOutputMetaDataVector{
                {at::ScalarType::Float8_e4m3fn, x.sizes().vec()}, {at::kFloat, {x.size(0), 1}},
                {at::kBFloat16, x.sizes().vec()}, {at::kBFloat16, x.sizes().vec()}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(AddRmsNormQuantParams);
            return std::make_shared<AddRmsNormQuantParams>(AddRmsNormQuantParams{
                static_cast<float>(stack.at(3).toDouble()), 1.0f / static_cast<float>(stack.at(0).toTensor().size(1)),
                stack.at(4).toBool() ? 1.0f / 240.0f : 0.004180908203125f});
        });
    return true;
}();

Outputs run(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double epsilon, bool precise_scale) {
    validate(x, residual, weight, epsilon);
    TORCH_CHECK(x.device().type() == c10::DeviceType::HPU, "Norm-quant requires HPU inputs");
    TORCH_CHECK(registered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    const auto outputs = descriptor.execute({x, residual, weight, epsilon, precise_scale});
    return {outputs.at(0), outputs.at(1), outputs.at(2), outputs.at(3)};
}

Outputs meta(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double epsilon, bool) {
    validate(x, residual, weight, epsilon);
    return {at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn)),
            at::empty({x.size(0), 1}, x.options().dtype(at::kFloat)),
            at::empty_like(x), at::empty_like(x)};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("flashinfer_gaudi_add_rmsnorm_quant(Tensor input, Tensor residual, Tensor weight, float epsilon, bool precise_scale=False) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("flashinfer_gaudi_add_rmsnorm_quant", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("flashinfer_gaudi_add_rmsnorm_quant", meta);
}
