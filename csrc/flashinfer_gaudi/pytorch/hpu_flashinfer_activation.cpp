// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <limits>
#include "hpu_custom_op_pt2.h"

namespace {
constexpr const char* schema = "custom_op::flashinfer_gaudi_silu_and_mul";
constexpr const char* guid = "flashinfer_gaudi_silu_and_mul_bf16_gaudi2";

void validate(const at::Tensor& input) {
    TORCH_CHECK(!input.requires_grad(), "SiLU native kernel is inference-only");
    TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16, "SiLU native kernel requires BF16");
    TORCH_CHECK(input.dim() == 2 && input.is_contiguous(), "SiLU native kernel requires contiguous rank-2 input");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && input.size(1) % 256 == 0,
                "SiLU native kernel requires B>0 and 2D divisible by 256");
    TORCH_CHECK(input.size(0) <= std::numeric_limits<int32_t>::max() &&
                input.size(1) <= std::numeric_limits<int32_t>::max(), "SiLU dimensions exceed index range");
}

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        schema, guid,
        [](const at::Stack& stack) {
            const auto& input = stack.at(0).toTensor();
            validate(input);
            habana::PartialOutputMetaData meta;
            meta.dtype = input.scalar_type();
            meta.shape = {input.size(0), input.size(1) / 2};
            return habana::PartialOutputMetaDataVector{meta};
        }, nullptr);
    return true;
}();

at::Tensor run(const at::Tensor& input) {
    validate(input);
    TORCH_CHECK(input.device().type() == c10::DeviceType::HPU, "SiLU requires HPU input");
    TORCH_CHECK(registered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    std::vector<c10::IValue> inputs{input};
    return descriptor.execute(inputs).at(0);
}

at::Tensor meta(const at::Tensor& input) {
    validate(input);
    return at::empty({input.size(0), input.size(1) / 2}, input.options());
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("flashinfer_gaudi_silu_and_mul(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("flashinfer_gaudi_silu_and_mul", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("flashinfer_gaudi_silu_and_mul", meta);
}
