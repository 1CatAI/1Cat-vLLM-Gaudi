// SPDX-License-Identifier: Apache-2.0
// Selected-row decoder and its paged attention consumer. The paged variant
// keeps the selected BF16 rows as an internal graph value instead of exposing
// a Python-side unpack_swa/cat chain.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "backend/helpers/create_tensor.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kGather = "custom_deepseek_v41_selected_kv_bf16_gaudi2";
constexpr auto kGatherSchema = "custom_op::custom_deepseek_v41_paged_selected_kv_bf16_gaudi2";
constexpr auto kAttention = "custom_deepseek_v41_paged_attention_bf16_gaudi2";
constexpr auto kAttentionSchema = "custom_op::custom_deepseek_v41_paged_attention_bf16_gaudi2";
constexpr auto kLengthsSchema = "custom_op::custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2";
constexpr auto kPackedExpSchema = "custom_op::custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2";
constexpr auto kVectorScalesSchema = "custom_op::custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2";
constexpr auto kSramSchema = "custom_op::custom_deepseek_v41_paged_attention_sram_bf16_gaudi2";
constexpr auto kHeadSchema = "custom_op::custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;

void tensor_contract(const at::Tensor& value, at::ScalarType dtype, const at::Device& device) {
    TORCH_CHECK(value.device() == device && value.scalar_type() == dtype && value.is_contiguous() &&
                !value.requires_grad(), "V4.1 selected KV requires contiguous matching inference tensors");
}
int slots(const at::Stack& stack) {
    const auto swa = stack.at(0).toTensor(), main = stack.at(1).toTensor(), ids = stack.at(2).toTensor();
    tensor_contract(swa, at::kByte, swa.device());
    tensor_contract(main, at::kByte, swa.device());
    tensor_contract(ids, at::kInt, swa.device());
    TORCH_CHECK(swa.dim() == 2 && swa.size(1) == 528 && swa.size(0) > 0 && swa.size(0) <= 512 &&
                main.dim() == 2 && main.size(1) == 288 && main.size(0) > 0 &&
                main.size(0) <= 0x7fffffffLL - 512 &&
                ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 && ids.size(1) <= 4096,
                "V4.1 selected KV requires SWA528, paged main288 and up to 4096 ordered row IDs");
    return ids.size(1);
}

class SelectedKV final : public habana::OpBackend {
public:
    SelectedKV(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_selected_kv"), dtype, {0, 1}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& stack) {
            const int k = slots(stack);
            return habana::OutputMetaDataVector{{at::kBFloat16, {k, 512}}, {at::kInt, {1, k}}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const int k = slots(stack);
        auto result = BuildNode(this, graph, {kGather, {syn_in(0), syn_in(1), syn_in(2)},
            {{{k, 512}, at::kBFloat16, 0}, {{1, k}, at::kInt, 1}}});
        syn_out(0) = std::move(result.at(0));
        syn_out(1) = std::move(result.at(1));
    }
};

class PagedAttention final : public habana::OpBackend {
    bool lengths_;
    bool packed_exp_;
    bool vector_scales_;
    bool sram_;
    bool head_vector_;
public:
    PagedAttention(int device, c10::ScalarType dtype, bool lengths = false, bool packed_exp = false, bool vector_scales = false, bool sram = false, bool head_vector = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_paged_selected_attention"), dtype, {0}, {}, {}, false),
          lengths_(lengths), packed_exp_(packed_exp), vector_scales_(vector_scales), sram_(sram), head_vector_(head_vector) {
        SetOutputMetaFn([](const at::Stack& stack) {
            const auto& q = stack.at(0).toTensor();
            slots({stack.at(1), stack.at(2), stack.at(3)});
            return habana::OutputMetaDataVector{{at::kBFloat16, q.sizes().vec()}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto& q = stack.at(0).toTensor();
        const int k = slots({stack.at(1), stack.at(2), stack.at(3)});
        // These are internal recipe values. A final-result index would bind
        // them to this operator's sole public output (or to absent output 1),
        // both breaking lowering and preventing SRAM-only consumption.
        const std::string gather_guid = vector_scales_
            ? "custom_deepseek_v41_selected_kv_vector_scales_bf16_gaudi2" : kGather;
        std::vector<synapse_helpers::tensor> selected;
        if (sram_ && !graph.is_dry_run() && !isOutputInfMode()) {
            // Synapse locks tensor properties when a producer is attached.
            // Create the private values and bind SRAM before adding that node.
            const int device = SynInput(1).ref().device_id();
            selected.emplace_back(habana_helpers::create_tensor(
                {k, 512}, {512, 1}, graph, false, false, device, at::kBFloat16));
            selected.emplace_back(habana_helpers::create_tensor(
                {1, k}, {k, 1}, graph, false, false, device, at::kInt));
            // A nonpersistent RMW section selects SRAM without changing the
            // kernel's store instructions. The graph retires its section only
            // after tensor references are released, including failed compile.
            synSectionHandle section = nullptr;
            TORCH_CHECK(synSectionCreate(&section, 0, graph.get_graph_handle()) == synSuccess,
                        "Could not create selected-KV SRAM section");
            TORCH_CHECK(synSectionSetPersistent(section, false) == synSuccess &&
                        synSectionSetRMW(section, true) == synSuccess &&
                        synTensorAssignToSection(selected.at(0).get(), section, 0) == synSuccess,
                        "Could not bind selected KV to graph-owned SRAM");
            graph.add_node({syn_in(1), syn_in(2), syn_in(3)},
                           {selected.at(0).get(), selected.at(1).get()},
                           nullptr, 0, gather_guid, nullptr, nullptr, nullptr,
                           deterministic, getContextHints());
        } else {
            // Keep the standard intermediate shapes and node metadata in SIF.
            selected = BuildNode(this, graph, {gather_guid, {syn_in(1), syn_in(2), syn_in(3)},
                {{{k, 512}, at::kBFloat16}, {{1, k}, at::kInt}}});
        }
        const auto tokens = q.size(0);
        if (head_vector_) {
            const int64_t width = stack.at(4).toTensor().size(1);
            auto scores = BuildNode(this, graph, {"custom_deepseek_v41_attn_scores_f32_gaudi2",
                {syn_in(0), selected.at(0).get(), syn_in(4), syn_in(6), syn_in(7)},
                {{{tokens, width, 32}, at::kFloat}}});
            int32_t rows = k;
            auto factors = BuildNode(this, graph, {"custom_deepseek_v41_attn_recurrence_f32_gaudi2",
                {scores.at(0).get(), syn_in(4), syn_in(5), syn_in(7)},
                {{{tokens, width, 2, 32}, at::kFloat}, {{tokens, 32}, at::kFloat}}, &rows, sizeof(rows)});
            auto values = BuildNode(this, graph, {"custom_deepseek_v41_attn_values_f32_gaudi2",
                {selected.at(0).get(), syn_in(4), syn_in(7), factors.at(0).get(), factors.at(1).get()},
                {{{tokens, 512, 32}, at::kFloat}}});
            auto rounded = BuildNode(this, graph, {"cast_f32_to_bf16", {values.at(0).get()},
                {{{tokens, 512, 32}, at::kBFloat16}}});
            syn_out(0) = BuildPermute(this, graph, rounded.at(0).get(), {tokens, 512, 32},
                                     {0, 2, 1}, at::kBFloat16, 0);
            return;
        }
        std::vector<synTensor> inputs{syn_in(0), selected.at(0).get(), syn_in(4), syn_in(5), syn_in(6)};
        if (lengths_) inputs.push_back(syn_in(7));
        auto result = BuildNode(this, graph, {
            packed_exp_ ? "custom_deepseek_v41_sparse_attn_packed_exp_bf16_gaudi2" :
            lengths_ ? "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2"
                     : "custom_deepseek_v4_sparse_attn_bf16_gaudi2", inputs,
            {{{tokens, 32, 512}, at::kBFloat16, 0},
             {{tokens, 32}, at::kFloat}, {{tokens, 32}, at::kFloat}}});
        syn_out(0) = std::move(result.at(0));
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kGatherSchema, kGather, [](const at::Stack& stack) {
        const int k = slots(stack);
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, {k, 512}}, {at::kInt, {1, k}}};
    }, nullptr);
    habana::KernelRegistry().add(kGatherSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<SelectedKV>(device, dtype);
    });
    habana::custom_op::registerUserCustomOp(kAttentionSchema, kAttention, [](const at::Stack& stack) {
        const auto& q = stack.at(0).toTensor();
        TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == 512 && q.size(0) > 0 && q.size(0) <= 6,
                    "V4.1 paged attention requires query [tokens,32,512]");
        const auto& swa = stack.at(1).toTensor();
        const auto& main = stack.at(2).toTensor();
        const auto& ids = stack.at(3).toTensor();
        const auto& indices = stack.at(4).toTensor();
        tensor_contract(q, at::kBFloat16, q.device());
        tensor_contract(indices, at::kInt, q.device());
        TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0) && indices.size(1) > 0,
                    "V4.1 paged attention indices must match query tokens");
        slots({swa, main, ids});
        const auto sink = stack.at(5).toTensor(), scale = stack.at(6).toTensor();
        tensor_contract(sink, at::kFloat, q.device());
        tensor_contract(scale, at::kFloat, q.device());
        TORCH_CHECK(sink.sizes() == at::IntArrayRef({32}) && scale.numel() == 1,
                    "V4.1 paged attention sink/scale contract changed");
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, q.sizes().vec()}};
    }, nullptr);
    habana::KernelRegistry().add(kAttentionSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PagedAttention>(device, dtype);
    });
    for (const int mode : {0, 1, 2, 3, 4}) {
        const bool packed = mode == 1, vector_scales = mode >= 2, sram = mode >= 3, head_vector = mode == 4;
        const auto schema = head_vector ? kHeadSchema : sram ? kSramSchema : vector_scales ? kVectorScalesSchema : (packed ? kPackedExpSchema : kLengthsSchema);
        habana::custom_op::registerUserCustomOp(schema, kAttention, [](const at::Stack& stack) {
            const auto& q = stack.at(0).toTensor();
            const auto& indices = stack.at(4).toTensor();
            const auto& lengths = stack.at(7).toTensor();
            slots({stack.at(1), stack.at(2), stack.at(3)});
            tensor_contract(q, at::kBFloat16, q.device());
            tensor_contract(indices, at::kInt, q.device());
            tensor_contract(lengths, at::kInt, q.device());
            TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == 512 && q.size(0) >= 1 && q.size(0) <= 6 &&
                        indices.dim() == 2 && indices.size(0) == q.size(0) && indices.size(1) > 0 &&
                        lengths.dim() == 1 && lengths.size(0) == q.size(0), "V4.1 prefix attention contract changed");
            const auto sink = stack.at(5).toTensor(), scale = stack.at(6).toTensor();
            tensor_contract(sink, at::kFloat, q.device());
            tensor_contract(scale, at::kFloat, q.device());
            TORCH_CHECK(sink.sizes() == at::IntArrayRef({32}) && scale.numel() == 1);
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, q.sizes().vec()}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [packed, vector_scales, sram, head_vector](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<PagedAttention>(device, dtype, true, packed, vector_scales, sram, head_vector);
        });
    }
    return true;
}();

template<bool Meta> Pair gather(const at::Tensor& swa, const at::Tensor& main, const at::Tensor& ids) {
    const at::Stack stack{swa, main, ids};
    const int k = slots(stack);
    if (Meta) return {at::empty({k, 512}, swa.options().dtype(at::kBFloat16)), at::empty({1, k}, ids.options())};
    TORCH_CHECK(registered && swa.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kGatherSchema);
    auto outputs = descriptor.execute(stack);
    return {outputs.at(0), outputs.at(1)};
}

template<bool Meta> at::Tensor paged_attention(const at::Tensor& q, const at::Tensor& swa,
    const at::Tensor& main, const at::Tensor& ids, const at::Tensor& indices,
    const at::Tensor& sink, const at::Tensor& scale) {
    const at::Stack stack{q, swa, main, ids, indices, sink, scale};
    if (Meta) {
        TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == 512);
        return at::empty_like(q);
    }
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kAttentionSchema);
    return descriptor.execute(stack).at(0);
}

template<bool Meta, bool Packed = false, bool VectorScales = false, bool Sram = false, bool HeadVector = false> at::Tensor paged_attention_lengths(const at::Tensor& q, const at::Tensor& swa,
    const at::Tensor& main, const at::Tensor& ids, const at::Tensor& indices,
    const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    const at::Stack stack{q, swa, main, ids, indices, sink, scale, lengths};
    if (Meta) {
        TORCH_CHECK(q.dim() == 3 && q.size(1) == 32 && q.size(2) == 512 &&
                    lengths.dim() == 1 && lengths.size(0) == q.size(0));
        return at::empty_like(q);
    }
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        HeadVector ? kHeadSchema : Sram ? kSramSchema : VectorScales ? kVectorScalesSchema : (Packed ? kPackedExpSchema : kLengthsSchema));
    return descriptor.execute(stack).at(0);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_paged_selected_kv_bf16_gaudi2(Tensor swa, Tensor main, Tensor indices) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_paged_attention_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale) -> Tensor");
    m.def("custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_attention_sram_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_paged_selected_kv_bf16_gaudi2", gather<false>);
    m.impl("custom_deepseek_v41_paged_attention_bf16_gaudi2", paged_attention<false>);
    m.impl("custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2", paged_attention_lengths<false>);
    m.impl("custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2", paged_attention_lengths<false, false, true>);
    m.impl("custom_deepseek_v41_paged_attention_sram_bf16_gaudi2", paged_attention_lengths<false, false, true, true>);
    m.impl("custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2", paged_attention_lengths<false, false, true, true, true>);
    m.impl("custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2", paged_attention_lengths<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_paged_selected_kv_bf16_gaudi2", gather<true>);
    m.impl("custom_deepseek_v41_paged_attention_bf16_gaudi2", paged_attention<true>);
    m.impl("custom_deepseek_v41_paged_attention_lengths_bf16_gaudi2", paged_attention_lengths<true>);
    m.impl("custom_deepseek_v41_paged_attention_vector_scales_bf16_gaudi2", paged_attention_lengths<true, false, true>);
    m.impl("custom_deepseek_v41_paged_attention_sram_bf16_gaudi2", paged_attention_lengths<true, false, true, true>);
    m.impl("custom_deepseek_v41_paged_attention_head_vector_bf16_gaudi2", paged_attention_lengths<true, false, true, true, true>);
    m.impl("custom_deepseek_v41_paged_attention_packed_exp_bf16_gaudi2", paged_attention_lengths<true, true>);
}
