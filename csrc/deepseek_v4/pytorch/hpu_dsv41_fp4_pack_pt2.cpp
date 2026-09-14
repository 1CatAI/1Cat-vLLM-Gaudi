// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto kWrite = "custom_op::custom_deepseek_v41_fp4_cache_write_bf16_gaudi2";
constexpr auto kOrdered = "custom_op::custom_deepseek_v41_fp4_cache_write_ordered_bf16_gaudi2";
void value_contract(const at::Tensor& value, unsigned group) {
    TORCH_CHECK(value.scalar_type() == at::kBFloat16 && value.dim() == 2 && value.is_contiguous() &&
                !value.requires_grad() && value.size(0) > 0 && value.size(0) <= 8192 && value.size(1) > 0 &&
                value.size(1) <= 16384 && value.size(1) % group == 0, "Invalid V4.1 FP4 packing input");
}
void write_contract(const at::Stack& stack) {
    const auto main = stack.at(0).toTensor(), index = stack.at(1).toTensor();
    const auto main_value = stack.at(2).toTensor(), index_value = stack.at(3).toTensor();
    const auto position = stack.at(4).toTensor();
    value_contract(main_value, 16); value_contract(index_value, 32);
    TORCH_CHECK(main_value.sizes() == at::IntArrayRef({1,512}) && index_value.sizes() == at::IntArrayRef({1,128}) &&
                main.dim() == 2 && index.dim() == 2 && main.size(0) > 0 && main.size(0) <= 1024 &&
                main.size(0) == index.size(0) && main.size(1) == 288 && index.size(1) == 68 &&
                main.scalar_type() == at::kByte && index.scalar_type() == at::kByte &&
                position.scalar_type() == at::kInt && position.sizes() == at::IntArrayRef({1}),
                "V4.1 FP4 cache update is C1 with main288 and index68 rows");
    for (const auto& tensor : {main, index, main_value, index_value, position})
        TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad() && tensor.device() == main.device(),
                    "FP4 cache update inputs must be contiguous inference tensors on the same device");
}
const bool registered = [] {
    for (auto schema : {kWrite, kOrdered})
        habana::custom_op::registerUserCustomOp(schema, "custom_deepseek_v41_fp4_cache_write_bf16_gaudi2",
            [](const at::Stack& stack) {
                write_contract(stack); return habana::PartialOutputMetaDataVector{{at::kInt, {36}}};
            }, nullptr);
    return true;
}();
template<bool Meta, bool Ordered> at::Tensor write(const at::Tensor& main, const at::Tensor& index,
    const at::Tensor& main_value, const at::Tensor& index_value, const at::Tensor& position) {
    const at::Stack stack{main,index,main_value,index_value,position}; write_contract(stack);
    if (Meta) return at::empty({36}, position.options());
    TORCH_CHECK(registered && main.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered ? kOrdered : kWrite);
    return descriptor.execute(stack).at(0);
}
at::Tensor unwrap(const at::Tensor& tensor) {
    if (!at::functionalization::impl::isFunctionalTensor(tensor)) return tensor;
    at::functionalization::impl::sync(tensor);
    return at::functionalization::impl::from_functional_tensor(tensor);
}
at::Tensor functionalize(const at::Tensor& main, const at::Tensor& index, const at::Tensor& main_value,
    const at::Tensor& index_value, const at::Tensor& position) {
    auto main_ = unwrap(main), index_ = unwrap(index), mv = unwrap(main_value), iv = unwrap(index_value);
    auto pos = unwrap(position);
    static auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(kOrdered, "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
    at::Tensor completion;
    {
        at::AutoDispatchSkipFunctionalize guard;
        completion = handle.call(main_, index_, mv, iv, pos);
    }
    for (const auto& update : {std::pair<const at::Tensor*, const at::Tensor*>{&main, &main_}, {&index, &index_}}) {
        at::functionalization::impl::replace_(*update.first, *update.second);
        at::functionalization::impl::commit_update(*update.first);
        at::functionalization::impl::sync(*update.first);
    }
    return completion;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_fp4_cache_write_bf16_gaudi2(Tensor(a!) main, Tensor(b!) index, Tensor main_value, Tensor index_value, Tensor position) -> Tensor");
    m.def("custom_deepseek_v41_fp4_cache_write_ordered_bf16_gaudi2(Tensor main, Tensor index, Tensor main_value, Tensor index_value, Tensor position) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_fp4_cache_write_bf16_gaudi2", write<false, false>);
    m.impl("custom_deepseek_v41_fp4_cache_write_ordered_bf16_gaudi2", write<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_fp4_cache_write_bf16_gaudi2", write<true, false>);
    m.impl("custom_deepseek_v41_fp4_cache_write_ordered_bf16_gaudi2", write<true, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
    m.impl("custom_deepseek_v41_fp4_cache_write_bf16_gaudi2", functionalize);
}
