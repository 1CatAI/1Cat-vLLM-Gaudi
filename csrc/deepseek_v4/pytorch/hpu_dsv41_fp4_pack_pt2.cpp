// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
#ifndef DSV41_FP4_WRITE_ROWS
#define DSV41_FP4_WRITE_ROWS 0
#endif
#if DSV41_FP4_WRITE_ROWS
#define DSV41_FP4_WRITE_SCHEMA "custom_deepseek_v41_fp4_cache_write_rows_gaudi2"
#define DSV41_FP4_ORDERED_SCHEMA "custom_deepseek_v41_fp4_cache_write_rows_ordered_gaudi2"
#else
#define DSV41_FP4_WRITE_SCHEMA "custom_deepseek_v41_fp4_cache_write_bf16_gaudi2"
#define DSV41_FP4_ORDERED_SCHEMA "custom_deepseek_v41_fp4_cache_write_ordered_bf16_gaudi2"
#endif
namespace {
constexpr auto kWrite = "custom_op::" DSV41_FP4_WRITE_SCHEMA;
constexpr auto kOrdered = "custom_op::" DSV41_FP4_ORDERED_SCHEMA;
std::vector<int64_t> completion_shape(const at::Stack& stack) {
    return DSV41_FP4_WRITE_ROWS ? std::vector<int64_t>{stack.at(2).toTensor().size(0),36}
                              : std::vector<int64_t>{36};
}
void value_contract(const at::Tensor& value, unsigned group) {
    TORCH_CHECK(value.scalar_type() == at::kBFloat16 && value.dim() == 2 && value.is_contiguous() &&
                !value.requires_grad() && value.size(0) > 0 && value.size(0) <= 16384 && value.size(1) > 0 &&
                value.size(1) <= 16384 && value.size(1) % group == 0, "Invalid V4.1 FP4 packing input");
}
void write_contract(const at::Stack& stack) {
    const auto main = stack.at(0).toTensor(), index = stack.at(1).toTensor();
    const auto main_value = stack.at(2).toTensor(), index_value = stack.at(3).toTensor();
    const auto position = stack.at(4).toTensor();
    value_contract(main_value, 16); value_contract(index_value, 32);
    const auto tokens=main_value.size(0);
    TORCH_CHECK(tokens <= (DSV41_FP4_WRITE_ROWS ? 6 : 1) && main_value.size(1)==512 &&
                index_value.sizes() == at::IntArrayRef({tokens,128}) &&
                main.dim() == 2 && index.dim() == 2 && main.size(0) > 0 &&
                main.size(0) <= (DSV41_FP4_WRITE_ROWS ? 1048576 : 1024) &&
                main.size(0) == index.size(0) && main.size(1) == 288 && index.size(1) == 68 &&
                main.scalar_type() == at::kByte && index.scalar_type() == at::kByte &&
                position.scalar_type() == at::kInt && position.sizes() == at::IntArrayRef({tokens}),
                "V4.1 FP4 cache update requires matching token rows, main288 and index68 caches");
    for (const auto& tensor : {main, index, main_value, index_value, position})
        TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad() && tensor.device() == main.device(),
                    "FP4 cache update inputs must be contiguous inference tensors on the same device");
}
const bool registered = [] {
    for (auto schema : {kWrite, kOrdered})
        habana::custom_op::registerUserCustomOp(schema, DSV41_FP4_WRITE_SCHEMA,
            [](const at::Stack& stack) {
                write_contract(stack); return habana::PartialOutputMetaDataVector{{at::kInt, completion_shape(stack)}};
            }, nullptr);
    return true;
}();
template<bool Meta, bool Ordered> at::Tensor write(const at::Tensor& main, const at::Tensor& index,
    const at::Tensor& main_value, const at::Tensor& index_value, const at::Tensor& position) {
    const at::Stack stack{main,index,main_value,index_value,position}; write_contract(stack);
    if (Meta) return at::empty(completion_shape(stack), position.options());
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
    m.def(DSV41_FP4_WRITE_SCHEMA "(Tensor(a!) main, Tensor(b!) index, Tensor main_value, Tensor index_value, Tensor position) -> Tensor");
    m.def(DSV41_FP4_ORDERED_SCHEMA "(Tensor main, Tensor index, Tensor main_value, Tensor index_value, Tensor position) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl(DSV41_FP4_WRITE_SCHEMA, write<false, false>);
    m.impl(DSV41_FP4_ORDERED_SCHEMA, write<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl(DSV41_FP4_WRITE_SCHEMA, write<true, false>);
    m.impl(DSV41_FP4_ORDERED_SCHEMA, write<true, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
    m.impl(DSV41_FP4_WRITE_SCHEMA, functionalize);
}
