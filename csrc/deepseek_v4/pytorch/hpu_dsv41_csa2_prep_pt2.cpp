// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto kIndices = "custom_op::custom_deepseek_v41_c1_indices_i32_gaudi2";
void indices_contract(const at::Tensor& position, const at::Tensor& compressed, int64_t ratio) {
    TORCH_CHECK(position.scalar_type() == at::kInt && position.sizes() == at::IntArrayRef({1}) &&
                compressed.scalar_type() == at::kInt && compressed.sizes() == at::IntArrayRef({1,512}) &&
                ratio >= 0 && ratio <= 2 && position.is_contiguous() && compressed.is_contiguous() &&
                position.device() == compressed.device(), "V4.1 C1 indices require I32 position/publication, ratio0/1/2");
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kIndices, "custom_deepseek_v41_c1_indices_i32_gaudi2",
        [](const at::Stack& stack) {
            const auto ratio = stack.at(2).toInt();
            indices_contract(stack.at(0).toTensor(),stack.at(1).toTensor(),ratio);
            return habana::PartialOutputMetaDataVector{{at::kInt,{1,ratio ? 640 : 128}},{at::kInt,{1}}};
        }, [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(int32_t);
            return std::make_shared<int32_t>(stack.at(2).toInt());
        });
    return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor> indices(
    const at::Tensor& position, const at::Tensor& compressed, int64_t ratio) {
    indices_contract(position,compressed,ratio);
    if (Meta) return {at::empty({1,ratio ? 640 : 128},position.options()),at::empty({1},position.options())};
    TORCH_CHECK(registered && position.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kIndices);
    auto output = descriptor.execute({position,compressed,ratio});
    return {output.at(0),output.at(1)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_c1_indices_i32_gaudi2(Tensor position, Tensor compressed, int ratio) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_c1_indices_i32_gaudi2",indices<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_c1_indices_i32_gaudi2",indices<true>);
}
