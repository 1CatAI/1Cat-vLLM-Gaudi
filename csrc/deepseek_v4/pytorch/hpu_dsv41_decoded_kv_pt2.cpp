// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kSwa = "custom_op::custom_deepseek_v41_swa_decoded_write_bf16_gaudi2";
constexpr auto kSwaOrdered = "custom_op::custom_deepseek_v41_swa_decoded_ordered_bf16_gaudi2";
constexpr auto kFp4 = "custom_op::custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2";
constexpr auto kFp4Ordered = "custom_op::custom_deepseek_v41_fp4_decoded_ordered_bf16_gaudi2";
constexpr auto kAttention = "custom_op::custom_deepseek_v41_decoded_attn_bf16_gaudi2";
using Outputs = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
struct AttentionParams { int32_t offset; int32_t main_rows; };

void tensor(const at::Tensor& value, at::ScalarType type, const at::Device& device) {
    TORCH_CHECK(value.scalar_type() == type && value.device() == device && value.is_contiguous() &&
                !value.requires_grad(), "Decoded KV requires contiguous inference tensors with matching dtype/device");
}
void swa_contract(const at::Tensor& cache, const at::Tensor& value, const at::Tensor& position,
                  const at::Tensor& decoded, int64_t offset) {
    tensor(cache, at::kByte, value.device()); tensor(value, at::kBFloat16, value.device());
    tensor(position, at::kInt, value.device()); tensor(decoded, at::kBFloat16, value.device());
    TORCH_CHECK(cache.sizes() == at::IntArrayRef({512,528}) && value.sizes() == at::IntArrayRef({1,512}) &&
                position.sizes() == at::IntArrayRef({1}) && decoded.dim() == 2 && decoded.size(1) == 512 &&
                decoded.size(0) <= 40 * 512 && offset >= 0 && offset % 512 == 0 &&
                offset <= decoded.size(0) - 512, "Invalid decoded SWA C1 row contract");
}
void fp4_contract(const at::Tensor& main, const at::Tensor& index, const at::Tensor& mv,
                  const at::Tensor& iv, const at::Tensor& position, const at::Tensor& decoded) {
    tensor(main, at::kByte, mv.device()); tensor(index, at::kByte, mv.device());
    tensor(mv, at::kBFloat16, mv.device()); tensor(iv, at::kBFloat16, mv.device());
    tensor(position, at::kInt, mv.device()); tensor(decoded, at::kBFloat16, mv.device());
    TORCH_CHECK(main.dim() == 2 && main.size(1) == 288 && main.size(0) > 0 && main.size(0) <= 1024 &&
                index.sizes() == at::IntArrayRef({main.size(0),68}) &&
                decoded.sizes() == at::IntArrayRef({main.size(0),512}) &&
                mv.sizes() == at::IntArrayRef({1,512}) && iv.sizes() == at::IntArrayRef({1,128}) &&
                position.sizes() == at::IntArrayRef({1}), "Invalid decoded FP4 C1 row contract");
}
void attention_contract(const at::Stack& stack) {
    const auto q = stack.at(0).toTensor(), swa = stack.at(1).toTensor(), main = stack.at(2).toTensor();
    const auto ids = stack.at(3).toTensor(), sink = stack.at(4).toTensor(), scale = stack.at(5).toTensor();
    const auto lengths = stack.at(6).toTensor();
    for (int i : {0,1,2}) tensor(stack.at(i).toTensor(), at::kBFloat16, q.device());
    for (int i : {3,6,7,8}) tensor(stack.at(i).toTensor(), at::kInt, q.device());
    for (int i : {4,5}) tensor(stack.at(i).toTensor(), at::kFloat, q.device());
    const auto offset = stack.at(9).toInt(), rows = stack.at(10).toInt();
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(1) > 0 && q.size(1) <= 64 && q.size(2) == 512 &&
                swa.dim() == 2 && swa.size(1) == 512 && main.dim() == 2 && main.size(1) == 512 &&
                offset >= 0 && offset % 512 == 0 && offset <= swa.size(0) - 512 && offset <= 40 * 512 &&
                rows >= 0 && rows <= main.size(0) && rows <= 512 &&
                ids.dim() == 2 && ids.size(0) == q.size(0) && ids.size(1) > 0 &&
                sink.sizes() == at::IntArrayRef({q.size(1)}) && scale.sizes() == at::IntArrayRef({1}) &&
                lengths.sizes() == at::IntArrayRef({q.size(0)}), "Invalid decoded attention tensor contract");
    for (int i : {7,8}) TORCH_CHECK(stack.at(i).toTensor().dim() == 1 && stack.at(i).toTensor().numel() > 0,
                                   "Decoded attention requires actual write completions");
}
const bool registered = [] {
    for (auto name : {kSwa, kSwaOrdered}) {
        habana::custom_op::registerUserCustomOp(name, "custom_deepseek_v41_swa_decoded_write_bf16_gaudi2",
            [](const at::Stack&) { return habana::PartialOutputMetaDataVector{{at::kInt,{16}}}; },
            [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
                size = sizeof(int32_t); return std::make_shared<int32_t>(stack.at(4).toInt());
            });
    }
    for (auto name : {kFp4, kFp4Ordered}) {
        habana::custom_op::registerUserCustomOp(name, "custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2",
            [](const at::Stack&) { return habana::PartialOutputMetaDataVector{{at::kInt,{36}}}; }, nullptr);
    }
    habana::custom_op::registerUserCustomOp(kAttention, "custom_deepseek_v41_decoded_attn_bf16_gaudi2",
        [](const at::Stack& stack) {
            const auto q = stack.at(0).toTensor();
            return habana::PartialOutputMetaDataVector{{at::kBFloat16,q.sizes().vec()},
                {at::kFloat,{q.size(0),q.size(1)}},{at::kFloat,{q.size(0),q.size(1)}}};
        }, [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(AttentionParams);
            return std::make_shared<AttentionParams>(AttentionParams{int32_t(stack.at(9).toInt()),
                                                                    int32_t(stack.at(10).toInt())});
        });
    return true;
}();
template<bool Meta, bool Ordered> at::Tensor swa_write(const at::Tensor& cache, const at::Tensor& value,
    const at::Tensor& position, const at::Tensor& decoded, int64_t offset) {
    swa_contract(cache,value,position,decoded,offset);
    if (Meta) return at::empty({16},position.options());
    TORCH_CHECK(registered && value.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered ? kSwaOrdered : kSwa);
    return op.execute({cache,value,position,decoded,offset}).at(0);
}
template<bool Meta, bool Ordered> at::Tensor fp4_write(const at::Tensor& main, const at::Tensor& index,
    const at::Tensor& mv, const at::Tensor& iv, const at::Tensor& position, const at::Tensor& decoded) {
    fp4_contract(main,index,mv,iv,position,decoded);
    if (Meta) return at::empty({36},position.options());
    TORCH_CHECK(registered && main.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Ordered ? kFp4Ordered : kFp4);
    return op.execute({main,index,mv,iv,position,decoded}).at(0);
}
template<bool Meta> Outputs attention(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths,
    const at::Tensor& swa_done, const at::Tensor& main_done, int64_t offset, int64_t rows) {
    const at::Stack stack{q,swa,main,ids,sink,scale,lengths,swa_done,main_done,offset,rows};
    attention_contract(stack);
    if (Meta) return {at::empty_like(q),at::empty({q.size(0),q.size(1)},q.options().dtype(at::kFloat)),
                                      at::empty({q.size(0),q.size(1)},q.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAttention);
    auto out = op.execute(stack); return {out.at(0),out.at(1),out.at(2)};
}
at::Tensor unwrap(const at::Tensor& tensor) {
    if (!at::functionalization::impl::isFunctionalTensor(tensor)) return tensor;
    at::functionalization::impl::sync(tensor);
    return at::functionalization::impl::from_functional_tensor(tensor);
}
void update(const at::Tensor& destination, const at::Tensor& value) {
    at::functionalization::impl::replace_(destination,value);
    at::functionalization::impl::commit_update(destination);
    at::functionalization::impl::sync(destination);
}
at::Tensor swa_functionalize(const at::Tensor& cache, const at::Tensor& value, const at::Tensor& position,
                            const at::Tensor& decoded, int64_t offset) {
    auto c = unwrap(cache), v = unwrap(value), p = unwrap(position), d = unwrap(decoded);
    static auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(kSwaOrdered, "")
        .typed<at::Tensor(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t)>();
    at::Tensor done;
    { at::AutoDispatchSkipFunctionalize guard; done = handle.call(c,v,p,d,offset); }
    update(cache,c); update(decoded,d); return done;
}
at::Tensor fp4_functionalize(const at::Tensor& main, const at::Tensor& index, const at::Tensor& mv,
    const at::Tensor& iv, const at::Tensor& position, const at::Tensor& decoded) {
    auto m = unwrap(main), i = unwrap(index), v = unwrap(mv), k = unwrap(iv), p = unwrap(position), d = unwrap(decoded);
    static auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(kFp4Ordered, "")
        .typed<at::Tensor(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,
                         const at::Tensor&,const at::Tensor&)>();
    at::Tensor done;
    { at::AutoDispatchSkipFunctionalize guard; done = handle.call(m,i,v,k,p,d); }
    update(main,m); update(index,i); update(decoded,d); return done;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_swa_decoded_write_bf16_gaudi2(Tensor(a!) cache, Tensor value, Tensor position, Tensor(b!) decoded, int offset) -> Tensor");
    m.def("custom_deepseek_v41_swa_decoded_ordered_bf16_gaudi2(Tensor cache, Tensor value, Tensor position, Tensor decoded, int offset) -> Tensor");
    m.def("custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2(Tensor(a!) main, Tensor(b!) index, Tensor main_value, Tensor index_value, Tensor position, Tensor(c!) decoded) -> Tensor");
    m.def("custom_deepseek_v41_fp4_decoded_ordered_bf16_gaudi2(Tensor main, Tensor index, Tensor main_value, Tensor index_value, Tensor position, Tensor decoded) -> Tensor");
    m.def("custom_deepseek_v41_decoded_attn_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, Tensor scale, Tensor lengths, Tensor swa_completion, Tensor main_completion, int offset, int main_rows) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_swa_decoded_write_bf16_gaudi2",swa_write<false,false>);
    m.impl("custom_deepseek_v41_swa_decoded_ordered_bf16_gaudi2",swa_write<false,true>);
    m.impl("custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2",fp4_write<false,false>);
    m.impl("custom_deepseek_v41_fp4_decoded_ordered_bf16_gaudi2",fp4_write<false,true>);
    m.impl("custom_deepseek_v41_decoded_attn_bf16_gaudi2",attention<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_swa_decoded_write_bf16_gaudi2",swa_write<true,false>);
    m.impl("custom_deepseek_v41_swa_decoded_ordered_bf16_gaudi2",swa_write<true,true>);
    m.impl("custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2",fp4_write<true,false>);
    m.impl("custom_deepseek_v41_fp4_decoded_ordered_bf16_gaudi2",fp4_write<true,true>);
    m.impl("custom_deepseek_v41_decoded_attn_bf16_gaudi2",attention<true>);
}
TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
    m.impl("custom_deepseek_v41_swa_decoded_write_bf16_gaudi2",swa_functionalize);
    m.impl("custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2",fp4_functionalize);
}
