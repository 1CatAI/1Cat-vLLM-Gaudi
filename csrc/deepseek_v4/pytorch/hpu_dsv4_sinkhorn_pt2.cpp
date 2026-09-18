// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* schema = "custom_op::custom_deepseek_v4_sinkhorn4_gaudi2";
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "custom_deepseek_v4_sinkhorn4_gaudi2",
        [](const at::Stack& inputs) {
            habana::PartialOutputMetaData output;
            output.dtype = at::kFloat;
            output.shape = inputs.at(0).toTensor().sizes().vec();
            return habana::PartialOutputMetaDataVector{output};
        }, nullptr);
    return true;
}();

void validate(const at::Tensor& input) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && input.dim() == 3 &&
                input.size(0) >= 1 && input.size(0) <= 8192 && input.size(1) == 4 && input.size(2) == 4 &&
                input.is_contiguous(), "V4 Sinkhorn requires contiguous FP32 [T,4,4], T in [1,8192]");
}
at::Tensor run(const at::Tensor& input) {
    validate(input);
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto outputs = descriptor.execute({input});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}
at::Tensor meta(const at::Tensor& input) {
    validate(input);
    return at::empty_like(input);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v4_sinkhorn4_gaudi2(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v4_sinkhorn4_gaudi2", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v4_sinkhorn4_gaudi2", meta);
}
