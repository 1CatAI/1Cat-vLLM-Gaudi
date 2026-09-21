// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kPair =
    "custom_op::custom_deepseek_v41_compressor_pair_bf16_gaudi2";
constexpr auto kPairOrdered =
    "custom_op::custom_deepseek_v41_compressor_pair_ordered_bf16_gaudi2";

void validate(const at::Tensor& kv_history,
              const at::Tensor& score_history,
              const at::Tensor& kv,
              const at::Tensor& score,
              const at::Tensor& position) {
    const auto device = kv.device();
    for (const auto& value : {kv_history, score_history, kv, score})
        TORCH_CHECK(value.scalar_type() == at::kFloat &&
                        value.device() == device && value.is_contiguous() &&
                        !value.requires_grad(),
                    "V4.1 compressor pair requires contiguous inference FP32 tensors");
    TORCH_CHECK(kv_history.sizes() == at::IntArrayRef({8, 512}) &&
                    score_history.sizes() == at::IntArrayRef({8, 512}) &&
                    kv.sizes() == at::IntArrayRef({1, 512}) &&
                    score.sizes() == at::IntArrayRef({1, 512}) &&
                    position.scalar_type() == at::kInt &&
                    position.sizes() == at::IntArrayRef({1}) &&
                    position.device() == device && position.is_contiguous() &&
                    !position.requires_grad(),
                "V4.1 compressor pair requires history [8,512], rows [1,512], position I32[1]");
}

const bool registered = [] {
    for (auto name : {kPair, kPairOrdered}) {
        habana::custom_op::registerUserCustomOp(
            name, "custom_deepseek_v41_compressor_pair_bf16_gaudi2",
            [](const at::Stack& stack) {
                validate(stack.at(0).toTensor(), stack.at(1).toTensor(),
                         stack.at(2).toTensor(), stack.at(3).toTensor(),
                         stack.at(4).toTensor());
                return habana::PartialOutputMetaDataVector{
                    {at::kBFloat16, {1, 512}}};
            }, nullptr);
    }
    return true;
}();

template<bool Meta, bool Ordered>
at::Tensor pair(const at::Tensor& kv_history,
                const at::Tensor& score_history,
                const at::Tensor& kv,
                const at::Tensor& score,
                const at::Tensor& position) {
    validate(kv_history, score_history, kv, score, position);
    if constexpr (Meta)
        return at::empty({1, 512}, kv.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && kv.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::
        getUserCustomOpDescriptor(Ordered ? kPairOrdered : kPair);
    return descriptor.execute(
        {kv_history, score_history, kv, score, position}).at(0);
}

at::Tensor unwrap(const at::Tensor& value) {
    if (!at::functionalization::impl::isFunctionalTensor(value)) return value;
    at::functionalization::impl::sync(value);
    return at::functionalization::impl::from_functional_tensor(value);
}

void update(const at::Tensor& destination, const at::Tensor& value) {
    at::functionalization::impl::replace_(destination, value);
    at::functionalization::impl::commit_update(destination);
    at::functionalization::impl::sync(destination);
}

at::Tensor functionalize(const at::Tensor& kv_history,
                         const at::Tensor& score_history,
                         const at::Tensor& kv,
                         const at::Tensor& score,
                         const at::Tensor& position) {
    auto kh = unwrap(kv_history), sh = unwrap(score_history);
    auto k = unwrap(kv), s = unwrap(score), p = unwrap(position);
    static auto handle = c10::Dispatcher::singleton()
        .findSchemaOrThrow(kPairOrdered, "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&,
                          const at::Tensor&)>();
    at::Tensor output;
    {
        at::AutoDispatchSkipFunctionalize guard;
        output = handle.call(kh, sh, k, s, p);
    }
    update(kv_history, kh);
    update(score_history, sh);
    return output;
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_compressor_pair_bf16_gaudi2(Tensor(a!) kv_history, Tensor(b!) score_history, Tensor kv, Tensor score, Tensor position) -> Tensor");
    m.def("custom_deepseek_v41_compressor_pair_ordered_bf16_gaudi2(Tensor kv_history, Tensor score_history, Tensor kv, Tensor score, Tensor position) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_compressor_pair_bf16_gaudi2",
           pair<false, false>);
    m.impl("custom_deepseek_v41_compressor_pair_ordered_bf16_gaudi2",
           pair<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_compressor_pair_bf16_gaudi2",
           pair<true, false>);
    m.impl("custom_deepseek_v41_compressor_pair_ordered_bf16_gaudi2",
           pair<true, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
    m.impl("custom_deepseek_v41_compressor_pair_bf16_gaudi2", functionalize);
}
