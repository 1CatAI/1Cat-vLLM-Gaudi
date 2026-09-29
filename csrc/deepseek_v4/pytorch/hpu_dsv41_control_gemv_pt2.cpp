// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* schema = "custom_op::custom_deepseek_v41_control_gemv_f32_gaudi2";
constexpr const char* batch4_schema = "custom_op::custom_deepseek_v41_control_batch4_f32_gaudi2";
constexpr const char* prefetch_schema = "custom_op::custom_deepseek_v41_control_prefetch_f32_gaudi2";
constexpr const char* rrms_schema =
    "custom_op::custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2";
constexpr const char* rrms_guid =
    "custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2";
struct RrmsParams { float epsilon; float inverse_width; };
void validate(const at::Tensor& input, const at::Tensor& weight) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                input.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 64 && input.size(1) == 20480 &&
                weight.sizes() == at::IntArrayRef({24, 20480}) &&
                input.is_contiguous() && weight.is_contiguous() && input.device() == weight.device(),
                "V4.1 control GEMV requires contiguous FP32 [B1..64,20480] and [24,20480]");
}
void validate_rrms(const at::Tensor& input, const at::Tensor& weight,
                   double epsilon) {
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 &&
                weight.scalar_type() == at::kFloat &&
                input.dim() == 2 && input.size(0) >= 1 &&
                input.size(0) <= 2048 && input.size(1) == 20480 &&
                weight.sizes() == at::IntArrayRef({24, 20480}) &&
                input.is_contiguous() && weight.is_contiguous() &&
                input.device() == weight.device() &&
                std::isfinite(static_cast<float>(epsilon)) && epsilon > 0,
                "V4.1 fused control/RRMS requires contiguous BF16 [1..2048,20480], "
                "FP32 [24,20480], and positive finite epsilon");
}
const bool registered = [] {
    for (const auto* name : {schema, batch4_schema, prefetch_schema}) {
    habana::custom_op::registerUserCustomOp(name, name + 11,
        [](const at::Stack& stack) {
            validate(stack.at(0).toTensor(), stack.at(1).toTensor());
            habana::PartialOutputMetaData output;
            output.dtype = at::kFloat;
            output.shape = {stack.at(0).toTensor().size(0), 24};
            return habana::PartialOutputMetaDataVector{output};
        }, nullptr);
    }
    return true;
}();
const bool rrms_registered = [] {
    habana::custom_op::registerUserCustomOp(
        rrms_schema, rrms_guid,
        [](const at::Stack& stack) {
            validate_rrms(stack.at(0).toTensor(), stack.at(1).toTensor(),
                          stack.at(2).toDouble());
            habana::PartialOutputMetaData output;
            output.dtype = at::kFloat;
            output.shape = {stack.at(0).toTensor().size(0), 25};
            return habana::PartialOutputMetaDataVector{output};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(RrmsParams);
            return std::make_shared<RrmsParams>(RrmsParams{
                float(stack.at(2).toDouble()), 1.0f / 20480.0f});
        });
    return true;
}();
template<bool Batch4 = false, bool Prefetch = false>
at::Tensor run(const at::Tensor& input, const at::Tensor& weight) {
    validate(input, weight);
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        Prefetch ? prefetch_schema : Batch4 ? batch4_schema : schema);
    auto outputs = descriptor.execute({input, weight});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}
at::Tensor meta(const at::Tensor& input, const at::Tensor& weight) {
    validate(input, weight);
    return at::empty({input.size(0), 24}, input.options());
}
template<bool Meta>
at::Tensor run_rrms(const at::Tensor& input, const at::Tensor& weight,
                    double epsilon) {
    validate_rrms(input, weight, epsilon);
    if constexpr (Meta)
        return at::empty({input.size(0), 25},
                         input.options().dtype(at::kFloat));
    TORCH_CHECK(rrms_registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::
        getUserCustomOpDescriptor(rrms_schema);
    auto outputs = descriptor.execute({input, weight, epsilon});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_control_gemv_f32_gaudi2(Tensor input, Tensor weight) -> Tensor");
    m.def("custom_deepseek_v41_control_batch4_f32_gaudi2(Tensor input, Tensor weight) -> Tensor");
    m.def("custom_deepseek_v41_control_prefetch_f32_gaudi2(Tensor input, Tensor weight) -> Tensor");
    m.def("custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2(Tensor input, Tensor weight, float epsilon) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_control_gemv_f32_gaudi2", run<false>);
    m.impl("custom_deepseek_v41_control_batch4_f32_gaudi2", run<true>);
    m.impl("custom_deepseek_v41_control_prefetch_f32_gaudi2", run<false, true>);
    m.impl("custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2",
           run_rrms<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_control_gemv_f32_gaudi2", meta);
    m.impl("custom_deepseek_v41_control_batch4_f32_gaudi2", meta);
    m.impl("custom_deepseek_v41_control_prefetch_f32_gaudi2", meta);
    m.impl("custom_deepseek_v41_control_gemv_rrms_bf16_gaudi2",
           run_rrms<true>);
}
