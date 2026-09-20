// SPDX-License-Identifier: Apache-2.0
// Pure V4.1 KV codecs. Mutable cache updates keep their existing graph nodes.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
const char* names[] = {"custom_op::custom_deepseek_v41_swa_pack_bf16_gaudi2",
    "custom_op::custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2",
    "custom_op::custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2",
    "custom_op::custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2"};
std::vector<int64_t> shape(const at::Tensor& value, unsigned mode) {
    const unsigned group = mode == 1 ? 16 : 32;
    TORCH_CHECK(value.scalar_type() == at::kBFloat16 && value.dim() == 2 && value.is_contiguous() &&
        !value.requires_grad() && value.size(0) > 0 && value.size(0) <= 8192 && value.size(1) > 0 &&
        value.size(1) <= 16384 && value.size(1) % group == 0, "Invalid V4.1 pure KV codec input");
    return {value.size(0), mode == 3 ? value.size(1) :
        (mode == 0 ? value.size(1) : value.size(1) / 2) + value.size(1) / group};
}
const bool registered = [] {
    for (unsigned mode = 0; mode < 4; ++mode)
        habana::custom_op::registerUserCustomOp(names[mode], names[mode] + 11, [mode](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{mode == 3 ? at::kBFloat16 : at::kByte,
                                                        shape(stack.at(0).toTensor(), mode)}};
        }, nullptr);
    return true;
}();
template<bool Meta, unsigned Mode> at::Tensor pack(const at::Tensor& value) {
    const auto sizes = shape(value, Mode);
    if (Meta) return at::empty(sizes, value.options().dtype(Mode == 3 ? at::kBFloat16 : at::kByte));
    TORCH_CHECK(registered && value.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[Mode]);
    return op.execute({value}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_swa_pack_bf16_gaudi2(Tensor value) -> Tensor");
    m.def("custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2(Tensor value) -> Tensor");
    m.def("custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2(Tensor value) -> Tensor");
    m.def("custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2(Tensor value) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_swa_pack_bf16_gaudi2", pack<false, 0>);
    m.impl("custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2", pack<false, 1>);
    m.impl("custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2", pack<false, 2>);
    m.impl("custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2", pack<false, 3>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_swa_pack_bf16_gaudi2", pack<true, 0>);
    m.impl("custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2", pack<true, 1>);
    m.impl("custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2", pack<true, 2>);
    m.impl("custom_deepseek_v41_fp4_roundtrip_g32_bf16_gaudi2", pack<true, 3>);
}
