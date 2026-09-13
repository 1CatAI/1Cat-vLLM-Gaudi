// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* schema = "custom_op::custom_deepseek_v41_control_gemv_f32_gaudi2";
void validate(const at::Tensor& input, const at::Tensor& weight) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat &&
                input.sizes() == at::IntArrayRef({1, 20480}) && weight.sizes() == at::IntArrayRef({24, 20480}) &&
                input.is_contiguous() && weight.is_contiguous() && input.device() == weight.device(),
                "V4.1 control GEMV requires contiguous FP32 [1,20480] and [24,20480]");
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "custom_deepseek_v41_control_gemv_f32_gaudi2",
        [](const at::Stack&) {
            habana::PartialOutputMetaData output;
            output.dtype = at::kFloat;
            output.shape = {1, 24};
            return habana::PartialOutputMetaDataVector{output};
        }, nullptr);
    return true;
}();
at::Tensor run(const at::Tensor& input, const at::Tensor& weight) {
    validate(input, weight);
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto outputs = descriptor.execute({input, weight});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}
at::Tensor meta(const at::Tensor& input, const at::Tensor& weight) {
    validate(input, weight);
    return at::empty({1, 24}, input.options());
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_control_gemv_f32_gaudi2(Tensor input, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_control_gemv_f32_gaudi2", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_control_gemv_f32_gaudi2", meta);
}
