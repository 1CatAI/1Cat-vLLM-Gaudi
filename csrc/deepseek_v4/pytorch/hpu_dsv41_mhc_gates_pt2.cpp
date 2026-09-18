// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kName = "custom_deepseek_v41_mhc_gates_f32_gaudi2";
constexpr auto kSchema = "custom_op::custom_deepseek_v41_mhc_gates_f32_gaudi2";

void validate(const at::Tensor& mixes, const at::Tensor& rrms, const at::Tensor& scale,
              const at::Tensor& base) {
    TORCH_CHECK(mixes.scalar_type() == at::kFloat && mixes.dim() == 2 && mixes.size(0) >= 1 &&
                mixes.size(0) <= 8192 && mixes.size(1) == 24 && mixes.is_contiguous(),
                "V4.1 mHC gates require contiguous FP32 mixes [T,24]");
    TORCH_CHECK(rrms.scalar_type() == at::kFloat && rrms.sizes() == at::IntArrayRef({mixes.size(0), 1}) &&
                rrms.is_contiguous(), "V4.1 mHC gates require contiguous FP32 rrms [T,1]");
    TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.numel() == 3 && scale.is_contiguous(),
                "V4.1 mHC gates require contiguous FP32 scale [3]");
    TORCH_CHECK(base.scalar_type() == at::kFloat && base.numel() == 24 && base.is_contiguous(),
                "V4.1 mHC gates require contiguous FP32 base [24]");
}

habana::PartialOutputMetaDataVector metadata(const at::Stack& inputs) {
    const auto tokens = inputs.at(0).toTensor().size(0);
    habana::PartialOutputMetaData output;
    output.dtype = at::kFloat;
    output.shape = {tokens, 24};
    return {output};
}

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kSchema, kName, metadata, nullptr);
    return true;
}();

at::Tensor run(
    const at::Tensor& mixes, const at::Tensor& rrms, const at::Tensor& scale, const at::Tensor& base) {
    validate(mixes, rrms, scale, base);
    TORCH_CHECK(registered && mixes.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    auto outputs = descriptor.execute({mixes, rrms, scale, base});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}

at::Tensor meta(
    const at::Tensor& mixes, const at::Tensor& rrms, const at::Tensor& scale, const at::Tensor& base) {
    validate(mixes, rrms, scale, base);
    return at::empty({mixes.size(0), 24}, mixes.options());
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_mhc_gates_f32_gaudi2(Tensor mixes, Tensor rrms, Tensor scale, Tensor base) "
          "-> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_mhc_gates_f32_gaudi2", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_mhc_gates_f32_gaudi2", meta);
}
