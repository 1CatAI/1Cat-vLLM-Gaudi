// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* schema = "custom_op::custom_deepseek_v41_quant_roundtrip_bf16_gaudi2";
constexpr const char* wide_schema = "custom_op::custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2";
const bool registered = [] {
    for (bool wide : {false, true}) {
    habana::custom_op::registerUserCustomOp(wide ? wide_schema : schema, wide ? "custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2" : "custom_deepseek_v41_quant_roundtrip_bf16_gaudi2",
        [](const at::Stack& inputs) {
            habana::PartialOutputMetaData output;
            output.dtype = at::kBFloat16;
            output.shape = inputs.at(0).toTensor().sizes().vec();
            return habana::PartialOutputMetaDataVector{output};
        }, nullptr);
    }
    return true;
}();

void validate(const at::Tensor& input) {
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous() &&
                input.size(0) >= 1 && input.size(0) <= 8192 && input.size(1) >= 32 &&
                input.size(1) <= 131072 && input.size(1) % 32 == 0,
                "V4.1 activation quantization requires contiguous BF16 [T,K], T<=8192, K%32=0");
}
template <bool Wide>
at::Tensor run(const at::Tensor& input) {
    validate(input);
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Wide ? wide_schema : schema);
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
    m.def("custom_deepseek_v41_quant_roundtrip_bf16_gaudi2(Tensor input) -> Tensor");
    m.def("custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_quant_roundtrip_bf16_gaudi2", run<false>);
    m.impl("custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2", run<true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_quant_roundtrip_bf16_gaudi2", meta);
    m.impl("custom_deepseek_v41_quant_roundtrip_wide_bf16_gaudi2", meta);
}
