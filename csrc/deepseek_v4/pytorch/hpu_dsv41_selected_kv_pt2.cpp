// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kGather = "custom_deepseek_v41_selected_kv_bf16_gaudi2";
constexpr auto kCacheOrderedGather = "custom_deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2";
constexpr auto kCacheOrderedSchema = "custom_op::custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2";
constexpr auto kOrderedGather = "custom_deepseek_v41_selected_kv_ordered_bf16_gaudi2";
constexpr auto kOrderedSchema = "custom_op::custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2";
constexpr auto kGatherSchema = "custom_op::custom_deepseek_v41_selected_kv_bf16_gaudi2";
constexpr auto kAttentionSchema = "custom_op::custom_deepseek_v41_packed_attention_bf16_gaudi2";
constexpr auto kLengthsSchema = "custom_op::custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2";
constexpr auto kPairExpGuid = "custom_deepseek_v41_sparse_attn_pair_exp_bf16_gaudi2";
constexpr auto kPairExpSchema = "custom_op::custom_deepseek_v41_sparse_attn_pair_exp_bf16_gaudi2";
constexpr auto kHeadPairGuid = "custom_deepseek_v41_sparse_attn_head_pair_bf16_gaudi2";
constexpr auto kHeadPairSchema = "custom_op::custom_deepseek_v41_sparse_attn_head_pair_bf16_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;

void tensor_contract(const at::Tensor& value, at::ScalarType dtype, const at::Device& device) {
    TORCH_CHECK(value.device() == device && value.scalar_type() == dtype && value.is_contiguous() &&
                !value.requires_grad(), "V4.1 selected KV requires contiguous matching inference tensors");
}
int slots(const at::Stack& stack, bool attention, bool lengths = false, bool ordered = false, bool compressed = false) {
    const unsigned shift = attention ? 1 : 0;
    const auto swa = stack.at(shift).toTensor(), main = stack.at(shift+1).toTensor(), ids = stack.at(shift+2).toTensor();
    tensor_contract(swa, at::kByte, swa.device());
    tensor_contract(main, at::kByte, swa.device());
    tensor_contract(ids, at::kInt, swa.device());
    TORCH_CHECK(swa.dim() == 2 && swa.size(1) == 528 && swa.size(0) > 0 && swa.size(0) <= 512 &&
                main.dim() == 2 && (main.size(1) == 288 || main.size(1) == 528) && main.size(0) > 0 && main.size(0) <= 512 &&
                ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 && ids.size(1) <= 1024,
                "V4.1 selected KV is C1 with SWA528 and main288 (or inactive SWA alias)");
    if (attention) {
        const auto q = stack.at(0).toTensor(), sink = stack.at(4).toTensor(), scale = stack.at(5).toTensor();
        tensor_contract(q, at::kBFloat16, swa.device());
        tensor_contract(sink, at::kFloat, swa.device());
        tensor_contract(scale, at::kFloat, swa.device());
        TORCH_CHECK(q.dim() == 3 && q.size(0) == 1 && q.size(1) == 32 && q.size(2) == 512 &&
                    sink.sizes() == at::IntArrayRef({32}) && scale.numel() == 1,
                    "V4.1 packed attention requires C1 TP2 query [1,32,512]");
        if (lengths) {
            const auto bound = stack.at(6).toTensor();
            tensor_contract(bound, at::kInt, swa.device());
            TORCH_CHECK(bound.sizes() == at::IntArrayRef({1}), "V4.1 attention length must be one device integer");
        }
    }
    if (ordered) {
        const auto completion = stack.at(7).toTensor();
        tensor_contract(completion, at::kInt, swa.device());
        TORCH_CHECK(completion.sizes() == at::IntArrayRef({16}), "SWA write completion must cover all 16 groups");
    }
    if (compressed) {
        const auto completion = stack.at(8).toTensor();
        tensor_contract(completion, at::kInt, swa.device());
        TORCH_CHECK(completion.sizes() == at::IntArrayRef({36}), "FP4 cache completion must cover all 36 groups");
    }
    return ids.size(1);
}

class SelectedKV final : public habana::OpBackend {
    bool attention_;
    bool lengths_;
    bool ordered_;
    bool compressed_;
public:
    SelectedKV(int device, c10::ScalarType dtype, bool attention, bool lengths, bool ordered, bool compressed)
        : OpBackend(device, NO_TPC + std::string("dsv41_selected_kv"), dtype, attention ? std::vector<int>{0} : std::vector<int>{0, 1}, {}, {}, false), attention_(attention), lengths_(lengths), ordered_(ordered), compressed_(compressed) {
        SetOutputMetaFn([attention, lengths, ordered, compressed](const at::Stack& stack) {
            const int k = slots(stack, attention, lengths, ordered, compressed);
            if (attention) return habana::OutputMetaDataVector{{at::kBFloat16, stack.at(0).toTensor().sizes().vec()}};
            return habana::OutputMetaDataVector{{at::kBFloat16, {k, 512}}, {at::kInt, {1, k}}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const int k = slots(stack, attention_, lengths_, ordered_, compressed_);
        const unsigned shift = attention_ ? 1 : 0;
        if (!attention_) {
            const bool vector = stack.at(3).toBool();
            auto result = BuildNode(this, graph, {vector ? "custom_deepseek_v41_selected_kv_vec_bf16_gaudi2" : kGather, {syn_in(0), syn_in(1), syn_in(2)},
                {{{k,512}, at::kBFloat16, 0}, {{1,k}, at::kInt, 1}}});
            syn_out(0) = std::move(result.at(0)); syn_out(1) = std::move(result.at(1));
            return;
        }
        std::vector<synTensor> gather_inputs{syn_in(shift), syn_in(shift+1), syn_in(shift+2)};
        if (ordered_) gather_inputs.push_back(syn_in(7));
        if (compressed_) gather_inputs.push_back(syn_in(8));
        const bool valid_only = ordered_ && stack.at(compressed_ ? 9 : 8).toBool();
        const bool vector = ordered_ && stack.at(compressed_ ? 10 : 9).toBool();
        const bool paired_exp = ordered_ && stack.at(compressed_ ? 11 : 10).toBool();
        const bool head_pair = ordered_ && stack.at(compressed_ ? 12 : 11).toBool();
        TORCH_CHECK(!(paired_exp && head_pair), "Choose one attention variant");
        TORCH_CHECK(!vector || valid_only, "Vector selected KV requires the ordered valid-only consumer");
        const char* guid = vector ? (compressed_ ? "custom_deepseek_v41_selected_kv_vec_cache_bf16_gaudi2"
                                                 : "custom_deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2")
                                     : valid_only ? (compressed_ ? "custom_deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2"
                                                    : "custom_deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2")
                                     : (compressed_ ? kCacheOrderedGather : ordered_ ? kOrderedGather : kGather);
        auto selected = BuildNode(this, graph, {guid,
            gather_inputs,
            {{{k,512}, at::kBFloat16}, {{1,k}, at::kInt}}});
        std::vector<synTensor> inputs{syn_in(0), selected.at(0).get(), selected.at(1).get(), syn_in(4), syn_in(5)};
        if (lengths_) inputs.push_back(syn_in(6));
        auto result = BuildNode(this, graph, {head_pair ? kHeadPairGuid : paired_exp ? kPairExpGuid : lengths_ ? "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2"
                                                     : "custom_deepseek_v4_sparse_attn_bf16_gaudi2", inputs,
            {{{1,32,512}, at::kBFloat16, 0}, {{1,32}, at::kFloat}, {{1,32}, at::kFloat}}});
        syn_out(0) = std::move(result.at(0));
    }
};
const bool registered = [] {
    for (const auto& descriptor : {std::make_pair(kPairExpSchema, kPairExpGuid),
                                   std::make_pair(kHeadPairSchema, kHeadPairGuid)}) {
    habana::custom_op::registerUserCustomOp(descriptor.first, descriptor.second, [](const at::Stack& stack) {
        const auto& q = stack.at(0).toTensor();
        return habana::PartialOutputMetaDataVector{
            {at::kBFloat16, q.sizes().vec()}, {at::kFloat, {q.size(0), q.size(1)}},
            {at::kFloat, {q.size(0), q.size(1)}}};
    }, nullptr);
    }
    for (int mode : {0, 1, 2, 3, 4}) {
        const bool attention = mode != 0, lengths = mode >= 2, ordered = mode >= 3, compressed = mode == 4;
        const auto schema = compressed ? kCacheOrderedSchema : ordered ? kOrderedSchema : lengths ? kLengthsSchema : attention ? kAttentionSchema : kGatherSchema;
        habana::custom_op::registerUserCustomOp(schema, kGather, [attention, lengths, ordered, compressed](const at::Stack& stack) {
            const int k = slots(stack, attention, lengths, ordered, compressed);
            if (attention) return habana::PartialOutputMetaDataVector{{at::kBFloat16, {1,32,512}}};
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, {k,512}}, {at::kInt, {1,k}}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [attention, lengths, ordered, compressed](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<SelectedKV>(device, dtype, attention, lengths, ordered, compressed);
        });
    }
    return true;
}();
template<bool Meta> Pair gather(const at::Tensor& swa, const at::Tensor& main, const at::Tensor& ids, bool vector) {
    const at::Stack stack{swa, main, ids, vector}; const int k = slots(stack, false);
    if (Meta) return {at::empty({k,512}, swa.options().dtype(at::kBFloat16)), at::empty({1,k}, ids.options())};
    TORCH_CHECK(registered && swa.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGatherSchema);
    auto outputs = descriptor.execute(stack);
    return {outputs.at(0), outputs.at(1)};
}
template<bool Meta> at::Tensor attention(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale) {
    const at::Stack stack{q,swa,main,ids,sink,scale}; slots(stack, true);
    if (Meta) return at::empty_like(q);
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAttentionSchema);
    return descriptor.execute(stack).at(0);
}
template<bool Meta> at::Tensor attention_lengths(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    const at::Stack stack{q,swa,main,ids,sink,scale,lengths}; slots(stack, true, true);
    if (Meta) return at::empty_like(q);
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kLengthsSchema);
    return descriptor.execute(stack).at(0);
}
template<bool Meta> at::Tensor attention_ordered_lengths(const at::Tensor& q, const at::Tensor& swa,
    const at::Tensor& main, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths, const at::Tensor& completion, bool valid_only, bool vector, bool paired_exp, bool head_pair) {
    TORCH_CHECK(!(paired_exp && head_pair), "Choose one attention variant");
    const at::Stack stack{q,swa,main,ids,sink,scale,lengths,completion,valid_only,vector,paired_exp,head_pair};
    TORCH_CHECK(!vector || valid_only, "Vector selected KV requires valid_only"); slots(stack, true, true, true);
    if (Meta) return at::empty_like(q);
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kOrderedSchema);
    return descriptor.execute(stack).at(0);
}
template<bool Meta> at::Tensor attention_ordered_cache_lengths(const at::Tensor& q, const at::Tensor& swa,
    const at::Tensor& main, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths, const at::Tensor& completion, const at::Tensor& compressed_completion, bool valid_only, bool vector, bool paired_exp, bool head_pair) {
    TORCH_CHECK(!(paired_exp && head_pair), "Choose one attention variant");
    const at::Stack stack{q,swa,main,ids,sink,scale,lengths,completion,compressed_completion,valid_only,vector,paired_exp,head_pair};
    TORCH_CHECK(!vector || valid_only, "Vector selected KV requires valid_only");
    slots(stack, true, true, true, true);
    if (Meta) return at::empty_like(q);
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kCacheOrderedSchema);
    return descriptor.execute(stack).at(0);
}

template<bool Meta, bool HeadPair = false> std::tuple<at::Tensor, at::Tensor, at::Tensor> attention_pair_exp(
    const at::Tensor& q, const at::Tensor& kv, const at::Tensor& indices,
    const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    for (const auto& tensor : {q, kv}) tensor_contract(tensor, at::kBFloat16, q.device());
    for (const auto& tensor : {indices, lengths}) tensor_contract(tensor, at::kInt, q.device());
    for (const auto& tensor : {sink, scale}) tensor_contract(tensor, at::kFloat, q.device());
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && kv.dim() == 2 && kv.size(1) == 512);
    TORCH_CHECK(!HeadPair || (q.size(1) > 0 && q.size(1) % 2 == 0));
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0));
    TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == q.size(0));
    TORCH_CHECK(sink.dim() == 1 && sink.size(0) == q.size(1) && scale.numel() == 1);
    if (Meta) return {at::empty_like(q),
        at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat)),
        at::empty({q.size(0), q.size(1)}, q.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(HeadPair ? kHeadPairSchema : kPairExpSchema);
    auto outputs = descriptor.execute({q, kv, indices, sink, scale, lengths});
    return {outputs.at(0), outputs.at(1), outputs.at(2)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_sparse_attn_pair_exp_bf16_gaudi2(Tensor q, Tensor kv, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_sparse_attn_head_pair_bf16_gaudi2(Tensor q, Tensor kv, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, Tensor scale, Tensor lengths, Tensor completion, Tensor compressed_completion, bool valid_only=False, bool vector=False, bool paired_exp=False, bool head_pair=False) -> Tensor");
    m.def("custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, Tensor scale, Tensor lengths, Tensor completion, bool valid_only=False, bool vector=False, bool paired_exp=False, bool head_pair=False) -> Tensor");
    m.def("custom_deepseek_v41_selected_kv_bf16_gaudi2(Tensor swa, Tensor main, Tensor indices, bool vector=False) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_packed_attention_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, Tensor scale) -> Tensor");
    m.def("custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_sparse_attn_pair_exp_bf16_gaudi2", attention_pair_exp<false>);
    m.impl("custom_deepseek_v41_sparse_attn_head_pair_bf16_gaudi2", attention_pair_exp<false, true>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2", attention_ordered_cache_lengths<false>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2", attention_ordered_lengths<false>);
    m.impl("custom_deepseek_v41_selected_kv_bf16_gaudi2", gather<false>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_gaudi2", attention<false>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2", attention_lengths<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_sparse_attn_pair_exp_bf16_gaudi2", attention_pair_exp<true>);
    m.impl("custom_deepseek_v41_sparse_attn_head_pair_bf16_gaudi2", attention_pair_exp<true, true>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_ordered_cache_lengths_gaudi2", attention_ordered_cache_lengths<true>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_ordered_lengths_gaudi2", attention_ordered_lengths<true>);
    m.impl("custom_deepseek_v41_selected_kv_bf16_gaudi2", gather<true>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_gaudi2", attention<true>);
    m.impl("custom_deepseek_v41_packed_attention_bf16_lengths_gaudi2", attention_lengths<true>);
}
