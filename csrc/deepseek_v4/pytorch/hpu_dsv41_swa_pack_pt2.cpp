// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kPack = "custom_op::custom_deepseek_v41_swa_pack_bf16_gaudi2";
constexpr auto kWrite = "custom_op::custom_deepseek_v41_swa_pack_write_bf16_gaudi2";
constexpr auto kOrdered = "custom_op::custom_deepseek_v41_swa_pack_write_ordered_bf16_gaudi2";

void validate_value(const at::Tensor& value) {
    TORCH_CHECK(value.scalar_type() == at::kBFloat16 && value.dim() == 2 && value.is_contiguous() &&
                !value.requires_grad() && value.size(0) > 0 && value.size(0) <= 8192 &&
                value.size(1) > 0 && value.size(1) <= 16384 && value.size(1) % 32 == 0,
                "V4.1 SWA packing requires contiguous inference BF16 [T,K], K divisible by 32");
}
void validate_write(const at::Tensor& cache, const at::Tensor& value, const at::Tensor& position) {
    validate_value(value);
    TORCH_CHECK(value.sizes() == at::IntArrayRef({1,512}) && cache.scalar_type() == at::kByte &&
                cache.dim() == 2 && cache.size(0) > 0 && cache.size(0) <= 512 && cache.size(1) == 528 &&
                cache.is_contiguous() && !cache.requires_grad() && position.scalar_type() == at::kInt &&
                position.sizes() == at::IntArrayRef({1}) && position.is_contiguous() &&
                cache.device() == value.device() && position.device() == value.device(),
                "V4.1 SWA row update requires C1 BF16 [1,512], U8 cache [S,528] and device I32 [1]");
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kPack, "custom_deepseek_v41_swa_pack_bf16_gaudi2",
        [](const at::Stack& stack) {
            const auto value = stack.at(0).toTensor(); validate_value(value);
            return habana::PartialOutputMetaDataVector{{at::kByte, {value.size(0), value.size(1) * 33 / 32}}};
        }, nullptr);
    for (auto name : {kWrite, kOrdered}) {
        habana::custom_op::registerUserCustomOp(name, "custom_deepseek_v41_swa_pack_write_bf16_gaudi2",
            [](const at::Stack& stack) {
                validate_write(stack.at(0).toTensor(), stack.at(1).toTensor(), stack.at(2).toTensor());
                return habana::PartialOutputMetaDataVector{{at::kInt, {16}}};
            }, nullptr);
    }
    return true;
}();

template<bool Meta> at::Tensor pack(const at::Tensor& value) {
    validate_value(value);
    if (Meta) return at::empty({value.size(0), value.size(1) * 33 / 32}, value.options().dtype(at::kByte));
    TORCH_CHECK(registered && value.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPack);
    return descriptor.execute({value}).at(0);
}
template<bool Meta, bool Ordered> at::Tensor write(
    const at::Tensor& cache, const at::Tensor& value, const at::Tensor& position) {
    validate_write(cache, value, position);
    if (Meta) return at::empty({16}, position.options());
    TORCH_CHECK(registered && value.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered ? kOrdered : kWrite);
    return descriptor.execute({cache, value, position}).at(0);
}
at::Tensor unwrap(const at::Tensor& tensor) {
    if (!at::functionalization::impl::isFunctionalTensor(tensor)) return tensor;
    at::functionalization::impl::sync(tensor);
    return at::functionalization::impl::from_functional_tensor(tensor);
}
at::Tensor functionalize(const at::Tensor& cache, const at::Tensor& value, const at::Tensor& position) {
    auto storage = unwrap(cache), input = unwrap(value), slot = unwrap(position);
    static auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(kOrdered, "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
    at::Tensor completion;
    {
        at::AutoDispatchSkipFunctionalize guard;
        completion = handle.call(storage, input, slot);
    }
    at::functionalization::impl::replace_(cache, storage);
    at::functionalization::impl::commit_update(cache);
    at::functionalization::impl::sync(cache);
    return completion;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_swa_pack_bf16_gaudi2(Tensor value) -> Tensor");
    m.def("custom_deepseek_v41_swa_pack_write_bf16_gaudi2(Tensor(a!) cache, Tensor value, Tensor position) -> Tensor");
    // Private functionalization form. Its completion tensor must be consumed
    // by ordered attention before the mutable cache is read.
    m.def("custom_deepseek_v41_swa_pack_write_ordered_bf16_gaudi2(Tensor cache, Tensor value, Tensor position) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_swa_pack_bf16_gaudi2", pack<false>);
    m.impl("custom_deepseek_v41_swa_pack_write_bf16_gaudi2", write<false, false>);
    m.impl("custom_deepseek_v41_swa_pack_write_ordered_bf16_gaudi2", write<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_swa_pack_bf16_gaudi2", pack<true>);
    m.impl("custom_deepseek_v41_swa_pack_write_bf16_gaudi2", write<true, false>);
    m.impl("custom_deepseek_v41_swa_pack_write_ordered_bf16_gaudi2", write<true, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
    m.impl("custom_deepseek_v41_swa_pack_write_bf16_gaudi2", functionalize);
}
