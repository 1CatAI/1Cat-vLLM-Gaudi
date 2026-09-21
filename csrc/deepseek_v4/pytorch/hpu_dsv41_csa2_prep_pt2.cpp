// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include <cmath>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto kIndices = "custom_op::custom_deepseek_v41_c1_indices_i32_gaudi2";
constexpr auto kNorm = "custom_op::custom_deepseek_v41_attention_norm_bf16_gaudi2";
constexpr auto kFinalCollapseNorm =
    "custom_op::custom_deepseek_v41_final_collapse_norm_bf16_gaudi2";
struct NormParams { float epsilon; };
void norm_contract(const at::Tensor& x, const at::Tensor& weight, double epsilon) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8192 &&
                (x.size(1) == 512 || x.size(1) == 1280 || x.size(1) == 5120) &&
                weight.sizes() == at::IntArrayRef({x.size(1)}) &&
                std::isnormal(static_cast<float>(epsilon)) && epsilon > 0,
                "Attention/final norm requires [T,512|1280|5120], weight [D], positive normal epsilon");
    for (const auto& value : {x, weight})
        TORCH_CHECK(value.is_contiguous() && !value.requires_grad() && value.device() == x.device() &&
                    value.scalar_type() == at::kBFloat16,
                    "Attention norm requires matching inference BF16 tensors");
}
void final_collapse_norm_contract(const at::Tensor& residual,
                                  const at::Tensor& pre_mix,
                                  const at::Tensor& weight,
                                  double epsilon) {
    TORCH_CHECK(residual.dim() == 3 && residual.size(0) >= 1 &&
                    residual.size(0) <= 2 && residual.size(1) == 4 &&
                    residual.size(2) == 5120 &&
                    pre_mix.sizes() ==
                        at::IntArrayRef({residual.size(0), int64_t(4)}) &&
                    weight.sizes() == at::IntArrayRef({int64_t(5120)}) &&
                    std::isnormal(static_cast<float>(epsilon)) && epsilon > 0,
                "Final collapse norm requires residual [T,4,5120], "
                "pre_mix [T,4], weight [5120], T<=2");
    TORCH_CHECK(residual.is_contiguous() && pre_mix.is_contiguous() &&
                    weight.is_contiguous() && !residual.requires_grad() &&
                    !pre_mix.requires_grad() && !weight.requires_grad() &&
                    residual.device() == pre_mix.device() &&
                    residual.device() == weight.device() &&
                    residual.scalar_type() == at::kBFloat16 &&
                    pre_mix.scalar_type() == at::kFloat &&
                    weight.scalar_type() == at::kBFloat16,
                "Final collapse norm requires matching inference tensors");
}
void indices_contract(const at::Tensor& position, const at::Tensor& compressed, int64_t ratio) {
    TORCH_CHECK(position.scalar_type() == at::kInt && position.sizes() == at::IntArrayRef({1}) &&
                compressed.scalar_type() == at::kInt && compressed.sizes() == at::IntArrayRef({1,512}) &&
                ratio >= 0 && ratio <= 2 && position.is_contiguous() && compressed.is_contiguous() &&
                position.device() == compressed.device(), "V4.1 C1 indices require I32 position/publication, ratio0/1/2");
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kNorm, "custom_deepseek_v41_attention_norm_bf16_gaudi2",
        [](const at::Stack& stack) {
            norm_contract(stack.at(0).toTensor(), stack.at(1).toTensor(), stack.at(2).toDouble());
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, stack.at(0).toTensor().sizes().vec()}};
        }, [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(NormParams);
            return std::make_shared<NormParams>(NormParams{float(stack.at(2).toDouble())});
        });
    habana::custom_op::registerUserCustomOp(
        kFinalCollapseNorm,
        "custom_deepseek_v41_final_collapse_norm_bf16_gaudi2",
        [](const at::Stack& stack) {
            const auto& residual = stack.at(0).toTensor();
            final_collapse_norm_contract(residual, stack.at(1).toTensor(),
                                         stack.at(2).toTensor(),
                                         stack.at(3).toDouble());
            return habana::PartialOutputMetaDataVector{{
                at::kBFloat16, {residual.size(0), residual.size(2)}}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(NormParams);
            return std::make_shared<NormParams>(
                NormParams{float(stack.at(3).toDouble())});
        });
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
template<bool Meta> at::Tensor norm(const at::Tensor& x, const at::Tensor& weight, double epsilon) {
    norm_contract(x, weight, epsilon);
    if (Meta) return at::empty_like(x);
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kNorm);
    return descriptor.execute({x, weight, epsilon}).at(0);
}
template<bool Meta> at::Tensor final_collapse_norm(
    const at::Tensor& residual, const at::Tensor& pre_mix,
    const at::Tensor& weight, double epsilon) {
    final_collapse_norm_contract(residual, pre_mix, weight, epsilon);
    if (Meta)
        return at::empty({residual.size(0), residual.size(2)},
                         residual.options());
    TORCH_CHECK(registered && residual.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
            kFinalCollapseNorm);
    return descriptor.execute({residual, pre_mix, weight, epsilon}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_attention_norm_bf16_gaudi2(Tensor value, Tensor weight, float epsilon) -> Tensor");
    m.def("custom_deepseek_v41_final_collapse_norm_bf16_gaudi2(Tensor residual, Tensor pre_mix, Tensor weight, float epsilon) -> Tensor");
    m.def("custom_deepseek_v41_c1_indices_i32_gaudi2(Tensor position, Tensor compressed, int ratio) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_attention_norm_bf16_gaudi2",norm<false>);
    m.impl("custom_deepseek_v41_final_collapse_norm_bf16_gaudi2",final_collapse_norm<false>);
    m.impl("custom_deepseek_v41_c1_indices_i32_gaudi2",indices<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_attention_norm_bf16_gaudi2",norm<true>);
    m.impl("custom_deepseek_v41_final_collapse_norm_bf16_gaudi2",final_collapse_norm<true>);
    m.impl("custom_deepseek_v41_c1_indices_i32_gaudi2",indices<true>);
}
