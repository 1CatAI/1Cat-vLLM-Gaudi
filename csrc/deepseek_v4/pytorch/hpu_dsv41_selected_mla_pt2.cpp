// SPDX-License-Identifier: Apache-2.0
// Reuse main 92c82b97's QK/softmax/PV algorithm with the retained paged cache.
// Token-specific selection is a BMM batch axis; all heads share each KV row.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "backend/helpers/create_tensor.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kPaged = "custom_op::custom_deepseek_v41_paged_mla_mme_gaudi2";
constexpr auto kPagedDirect = "custom_op::custom_deepseek_v41_paged_mla_direct_mme_gaudi2";
constexpr auto kSelected = "custom_op::custom_deepseek_v41_selected_mla_mme_gaudi2";
constexpr auto kBatchPaged = "custom_op::custom_deepseek_v41_batch_paged_mla_mme_gaudi2";
constexpr auto kBatchPacked = "custom_op::custom_deepseek_v41_batch_packed_mla_mme_gaudi2";
constexpr auto kBatchPackedSram = "custom_op::custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2";
constexpr auto kBatchPackedVector = "custom_op::custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2";
constexpr auto kPrefill = "custom_op::custom_deepseek_v41_prefill_mla_mme_gaudi2";
constexpr auto kPagedProjection =
    "custom_op::custom_deepseek_v41_paged_mla_woa_wob_fp8_roundtrip_gaudi2";

habana::OutputMetaDataVector meta(const at::Stack& s, bool paged, bool prefill = false, bool sram = false) {
    const auto q = s.at(0).toTensor();
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= (sram ? 8 : (prefill ? 64 : 6)) &&
                q.size(1) >= 1 && q.size(1) <= 64 && q.size(2) == 512,
                "Selected MLA query exceeds its bounded [T,H,512] tile");
    const int index = paged ? 4 : 2;
    for (int i = 0; i < (paged ? 8 : 6); ++i) {
        const auto t = s.at(i).toTensor();
        auto dtype = i == index || i == index + 3 ? at::kInt : at::kFloat;
        if (i == 0 || (!paged && i == 1)) dtype = at::kBFloat16;
        if (paged && (i == 1 || i == 2)) dtype = at::kByte;
        if (paged && i == 3) dtype = at::kInt;
        TORCH_CHECK(t.scalar_type() == dtype && t.device() == q.device() && t.is_contiguous() &&
                    !t.requires_grad(), "Selected MLA requires matching contiguous inference tensors");
    }
    const auto ids = s.at(index).toTensor();
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) == q.size(0) && ids.size(1) > 0 &&
                ids.size(1) <= 640 && ids.size(1) % 64 == 0 &&
                s.at(index + 1).toTensor().sizes() == at::IntArrayRef({q.size(1)}) &&
                s.at(index + 2).toTensor().sizes() == at::IntArrayRef({1}) &&
                s.at(index + 3).toTensor().sizes() == at::IntArrayRef({q.size(0)}),
                "Selected MLA index, sink, scale or length shape changed");
    const auto cache = s.at(1).toTensor();
    if (paged) {
        const auto main = s.at(2).toTensor(), rows = s.at(3).toTensor();
        TORCH_CHECK(cache.dim() == 2 && cache.size(0) > 0 && cache.size(0) <= (prefill ? 64 * 256 : 512) && cache.size(1) == 528 &&
                    main.dim() == 2 && main.size(0) > 0 && main.size(0) <= 0x7fffbfffLL && main.size(1) == 288 &&
                    rows.dim() == 2 && rows.size(0) == 1 && rows.size(1) > 0 && rows.size(1) <= (prefill ? 64 * 640 : 4096),
                    "Paged MLA requires SWA528, FP4 main288 and bounded selected row IDs");
    } else {
        TORCH_CHECK(cache.dim() == 2 && cache.size(0) > 0 &&
                    cache.size(0) <= (prefill ? 131072 : 4096) && cache.size(1) == 512,
                    "Selected MLA requires a bounded BF16 KV working set");
    }
    return {{at::kBFloat16, q.sizes().vec()}};
}

habana::OutputMetaDataVector projection_meta(const at::Stack& s) {
    TORCH_CHECK(s.size() == 14, "Paged MLA projection requires its complete producer/consumer operands");
    (void)meta(s, true);
    const auto q = s.at(0).toTensor();
    const auto woa = s.at(8).toTensor(), woa_scale = s.at(9).toTensor();
    const auto positions = s.at(10).toTensor(), phase = s.at(11).toTensor();
    const auto wob = s.at(12).toTensor(), wob_scale = s.at(13).toTensor();
    TORCH_CHECK(q.size(0) == 1 && (q.size(1) == 16 || q.size(1) == 32),
                "Paged MLA projection requires the C1 TP4/TP2 head geometry");
    TORCH_CHECK(woa.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                woa.sizes() == at::IntArrayRef({q.size(1) / 8, 4096, 1024}) &&
                woa_scale.scalar_type() == at::kFloat &&
                woa_scale.sizes() == at::IntArrayRef({q.size(1) / 8, 1, 1024}) &&
                wob.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                wob.sizes() == at::IntArrayRef({5120, q.size(1) * 128}) &&
                wob_scale.scalar_type() == at::kFloat &&
                wob_scale.sizes() == at::IntArrayRef({1, 5120}),
                "Paged MLA projection requires prepared FP8 weights and FP32 channel scales");
    TORCH_CHECK(positions.scalar_type() == at::kInt && positions.sizes() == at::IntArrayRef({1}) &&
                phase.scalar_type() == at::kFloat && phase.dim() == 2 &&
                phase.size(0) > 0 && phase.size(1) == 64,
                "Paged MLA projection requires a logical position and native rotary table");
    for (int i = 8; i < 14; ++i) {
        const auto value = s.at(i).toTensor();
        TORCH_CHECK(value.device() == q.device() && value.is_contiguous() && !value.requires_grad(),
                    "Paged MLA projection operands must share the contiguous inference contract");
    }
    return {{at::kBFloat16, {1, 5120}}};
}

class SelectedMla final : public habana::OpBackend {
    bool paged_;
    bool prefill_;
    bool direct_packed_;
    bool sram_;
    bool vector_;
    bool projection_;
    bool packed_gather_;
public:
    SelectedMla(int device, c10::ScalarType dtype, bool paged, bool prefill = false,
                bool direct_packed = false, bool sram = false, bool vector = false,
                bool projection = false, bool packed_gather = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_selected_mla_mme"), dtype, {0}, {}, {}, false),
          paged_(paged), prefill_(prefill), direct_packed_(direct_packed), sram_(sram), vector_(vector),
          projection_(projection), packed_gather_(packed_gather) {
        TORCH_CHECK(!projection || (paged && !prefill), "Projection requires packed C1 MLA");
        TORCH_CHECK(!packed_gather || (paged && !prefill && !projection), "Direct gather requires packed C1 MLA");
        SetOutputMetaFn([paged, prefill, sram, projection, packed_gather](const at::Stack& s) {
            TORCH_CHECK(!packed_gather || s.at(0).toTensor().size(0) == 1, "Direct packed gather requires C1");
            return projection ? projection_meta(s) : meta(s, paged, prefill, sram);
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = projection_ ? projection_meta(s) : meta(s, paged_, prefill_, sram_);
        const auto q = s.at(0).toTensor();
        const int index = paged_ ? 4 : 2;
        const int64_t tokens = q.size(0), heads = q.size(1), width = s.at(index).toTensor().size(1);
        const char* packed_guid = packed_gather_ ? "custom_deepseek_v41_paged_mla_gather_gaudi2" : vector_ ? "custom_deepseek_v41_selected_packed_mla_vector_gaudi2"
                                          : "custom_deepseek_v41_selected_packed_mla_gather_gaudi2";
        std::vector<synapse_helpers::tensor> selected;
        synTensor cache = syn_in(1);
        if (paged_ && !direct_packed_ && !packed_gather_) {
            const int64_t rows = s.at(3).toTensor().size(1);
            if (!graph.is_dry_run() && !isOutputInfMode()) {
                const int device = SynInput(1).ref().device_id();
                selected.emplace_back(habana_helpers::create_tensor(
                    {rows, 512}, {512, 1}, graph, false, false, device, at::kBFloat16));
                selected.emplace_back(habana_helpers::create_tensor(
                    {1, rows}, {rows, 1}, graph, false, false, device, at::kInt));
                synSectionHandle section = nullptr;
                TORCH_CHECK(synSectionCreate(&section, 0, graph.get_graph_handle()) == synSuccess &&
                            synSectionSetPersistent(section, false) == synSuccess &&
                            synSectionSetRMW(section, true) == synSuccess &&
                            synTensorAssignToSection(selected.at(0).get(), section, 0) == synSuccess,
                            "Could not bind selected MLA KV to recipe-owned SRAM");
                graph.add_node({syn_in(1), syn_in(2), syn_in(3)},
                    {selected.at(0).get(), selected.at(1).get()}, nullptr, 0,
                    "custom_deepseek_v41_selected_kv_vector_scales_bf16_gaudi2",
                    nullptr, nullptr, nullptr, deterministic, getContextHints());
            } else {
                selected = BuildNode(this, graph, {"custom_deepseek_v41_selected_kv_vector_scales_bf16_gaudi2",
                    {syn_in(1), syn_in(2), syn_in(3)}, {{{rows, 512}, at::kBFloat16}, {{1, rows}, at::kInt}}});
            }
            cache = selected.at(0).get();
        }
        std::vector<synapse_helpers::tensor> kv;
        if (sram_ && !graph.is_dry_run() && !isOutputInfMode()) {
            const int device = SynInput(1).ref().device_id();
            // Synapse requires all RMW operands of one node to share a
            // section. An eight-query tile uses less than the 16 MiB cap.
            synSectionHandle section = nullptr;
            TORCH_CHECK(synSectionCreate(&section, 0, graph.get_graph_handle()) == synSuccess &&
                        synSectionSetPersistent(section, false) == synSuccess &&
                        synSectionSetRMW(section, true) == synSuccess,
                        "Could not create packed MLA SRAM section");
            uint64_t offset = 0;
            for (int i = 0; i < 3; ++i) {
                const auto shape = i < 2 ? std::vector<int64_t>{tokens, width, 512}
                                         : std::vector<int64_t>{tokens, width};
                const auto stride = i < 2 ? std::vector<int64_t>{width * 512, 512, 1}
                                          : std::vector<int64_t>{width, 1};
                kv.emplace_back(habana_helpers::create_tensor(
                    shape, stride, graph, false, false, device, i == 0 ? at::kBFloat16 : at::kFloat));
                TORCH_CHECK(synTensorAssignToSection(kv.back().get(), section, offset) == synSuccess,
                            "Could not bind packed MLA operands to recipe-owned SRAM");
                const uint64_t bytes = tokens * width * (i < 2 ? 512 : 1) * (i == 0 ? 2 : 4);
                offset += (bytes + 127) & ~uint64_t(127);
            }
            graph.add_node({syn_in(1), syn_in(2), syn_in(3), syn_in(index), syn_in(index + 3)},
                {kv.at(0).get(), kv.at(1).get(), kv.at(2).get()}, nullptr, 0,
                packed_guid,
                nullptr, nullptr, nullptr, deterministic, getContextHints());
        } else {
            kv = (direct_packed_ || packed_gather_)
            ? BuildNode(this, graph, {packed_guid,
                {syn_in(1), syn_in(2), syn_in(3), syn_in(index), syn_in(index + 3)},
                {{{tokens, width, 512}, at::kBFloat16}, {{tokens, width, 512}, at::kFloat},
                 {{tokens, width}, at::kFloat}}})
            : BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_gather_gaudi2",
                {cache, syn_in(index), syn_in(index + 3)},
                {{{tokens, width, 512}, at::kBFloat16}, {{tokens, width, 512}, at::kFloat},
                 {{tokens, width}, at::kFloat}}});
        }
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = BuildNode(this, graph, {"batch_gemm", {syn_in(0), kv.at(0).get()},
            {{{tokens, heads, width}, at::kFloat}}, &qk, sizeof(qk)});
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_softmax_gaudi2",
            {scores.at(0).get(), kv.at(2).get(), syn_in(index + 1), syn_in(index + 2)},
            {{{tokens, heads, width}, at::kFloat}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(), kv.at(1).get()},
            {{{tokens, heads, 512}, at::kFloat}}, &pv, sizeof(pv)});
        if (projection_) {
            // Keep the established QK/PV batch-GEMM shapes and SRAM gather.
            // The existing product consumer performs both BF16 roundings in
            // registers before its inverse-RoPE and group quantization.
            auto product_row = ReshapeHelper(graph, product.at(0).get(), {heads, 512}, at::kFloat);
            const int64_t groups = heads / 8;
            auto quantized = BuildNode(this, graph, {"custom_deepseek_v41_mla_product_rope_quant_gaudi2",
                {product_row.get(), syn_in(10), syn_in(11)},
                {{{groups, 1, 4096}, at::ScalarType::Float8_e4m3fn}, {{groups, 1, 1}, at::kFloat}}});
            synGEMMParams woa_params{false, false}, wob_params{false, true};
            auto woa_product = BuildNode(this, graph, {"batch_gemm", {quantized.at(0).get(), syn_in(8)},
                {{{groups, 1, 1024}, at::kFloat}}, &woa_params, sizeof(woa_params)});
            auto scaled = BuildNode(this, graph, {"custom_deepseek_v41_woa_scale_roundtrip_gaudi2",
                {woa_product.at(0).get(), syn_in(9), quantized.at(1).get()},
                {{{1, groups, 1024}, at::kBFloat16}}});
            auto row = ReshapeHelper(graph, scaled.at(0).get(), {1, heads * 128}, at::kBFloat16);
            auto dense_q = BuildNode(this, graph, {"custom_deepseek_v41_dense_quant_gaudi2", {row.get()},
                {{{1, heads * 128}, at::ScalarType::Float8_e4m3fn}, {{1, 1}, at::kFloat}}});
            auto wob_product = BuildNode(this, graph, {"gemm", {dense_q.at(0).get(), syn_in(12)},
                {{{1, 5120}, at::kFloat}}, &wob_params, sizeof(wob_params)});
            syn_out(0) = std::move(BuildNode(this, graph, {"custom_deepseek_v41_dense_scale_gaudi2",
                {wob_product.at(0).get(), syn_in(13), dense_q.at(1).get()},
                {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
            return;
        }
        syn_out(0) = std::move(BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()},
            {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
    }
};

const bool registered = [] {
    for (int kind : {0, 1, 2, 3, 4, 5, 6}) {
        const bool direct_packed = kind >= 4, sram = kind >= 5, vector = kind == 6;
        const bool paged = kind == 1 || kind == 3 || direct_packed, prefill = kind >= 2;
        const auto schema = vector ? kBatchPackedVector : sram ? kBatchPackedSram : direct_packed ? kBatchPacked :
            (paged && prefill ? kBatchPaged : (prefill ? kPrefill : (paged ? kPaged : kSelected)));
        habana::custom_op::registerUserCustomOp(schema, "batch_gemm", [paged, prefill, sram](const at::Stack& s) {
            const auto out = meta(s, paged, prefill, sram);
            return habana::PartialOutputMetaDataVector{{out.at(0).dtype, out.at(0).shape}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [paged, prefill, direct_packed, sram, vector](synDeviceId d, c10::ScalarType t) {
            return std::make_shared<SelectedMla>(d, t, paged, prefill, direct_packed, sram, vector);
        });
    }
    habana::custom_op::registerUserCustomOp(kPagedDirect, "batch_gemm", [](const at::Stack& s) {
        TORCH_CHECK(s.at(0).toTensor().size(0) == 1, "Direct packed gather requires C1");
        const auto out = meta(s, true);
        return habana::PartialOutputMetaDataVector{{out.at(0).dtype, out.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kPagedDirect, [](synDeviceId d, c10::ScalarType t) {
        return std::make_shared<SelectedMla>(d, t, true, false, false, false, false, false, true);
    });
    habana::custom_op::registerUserCustomOp(kPagedProjection, "batch_gemm", [](const at::Stack& s) {
        const auto out = projection_meta(s);
        return habana::PartialOutputMetaDataVector{{out.at(0).dtype, out.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kPagedProjection, [](synDeviceId d, c10::ScalarType t) {
        return std::make_shared<SelectedMla>(d, t, true, false, false, false, false, true);
    });
    return true;
}();

template<bool Meta> at::Tensor execute(const at::Stack& s, bool paged, bool prefill = false,
                                      bool direct_packed = false, bool sram = false, bool vector = false,
                                      bool projection = false, bool packed_gather = false) {
    TORCH_CHECK(!packed_gather || s.at(0).toTensor().size(0) == 1, "Direct packed gather requires C1");
    const auto output = projection ? projection_meta(s) : meta(s, paged, prefill, sram);
    if (Meta) return at::empty(output.at(0).shape, s.at(0).toTensor().options());
    TORCH_CHECK(registered && s.at(0).toTensor().device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        packed_gather ? kPagedDirect : projection ? kPagedProjection :
        vector ? kBatchPackedVector : sram ? kBatchPackedSram : direct_packed ? kBatchPacked :
            (paged && prefill ? kBatchPaged : (prefill ? kPrefill : (paged ? kPaged : kSelected))));
    return descriptor.execute(s).at(0);
}
template<bool Meta> at::Tensor paged(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) { return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true); }
template<bool Meta> at::Tensor batch_paged(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) { return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true, true); }
template<bool Meta> at::Tensor batch_packed(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) { return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true, true, true); }
template<bool Meta> at::Tensor batch_packed_sram(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) { return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true, true, true, true); }
template<bool Meta> at::Tensor batch_packed_vector(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) { return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true, true, true, true, true); }
template<bool Meta> at::Tensor paged_direct(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) {
    return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true, false, false, false, false, false, true);
}
template<bool Meta> at::Tensor selected(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& ids,
    const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    return execute<Meta>({q, cache, ids, sink, scale, lengths}, false);
}
template<bool Meta> at::Tensor prefill(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& ids,
    const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    return execute<Meta>({q, cache, ids, sink, scale, lengths}, false, true);
}
template<bool Meta> at::Tensor paged_projection(const at::Tensor& q, const at::Tensor& swa,
    const at::Tensor& main, const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink,
    const at::Tensor& scale, const at::Tensor& lengths, const at::Tensor& woa, const at::Tensor& woa_scale,
    const at::Tensor& positions, const at::Tensor& phase, const at::Tensor& wob, const at::Tensor& wob_scale) {
    return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths, woa, woa_scale,
                         positions, phase, wob, wob_scale}, true, false, false, false, false, true);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_batch_packed_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_batch_paged_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_mla_direct_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, "
          "Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_mla_woa_wob_fp8_roundtrip_gaudi2(Tensor q, Tensor swa, Tensor main, "
          "Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths, Tensor woa, "
          "Tensor woa_scale, Tensor positions, Tensor phase, Tensor wob, Tensor wob_scale) -> Tensor");
    m.def("custom_deepseek_v41_prefill_mla_mme_gaudi2(Tensor q, Tensor cache, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_selected_mla_mme_gaudi2(Tensor q, Tensor cache, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2", batch_packed_vector<false>);
    m.impl("custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2", batch_packed_sram<false>);
    m.impl("custom_deepseek_v41_batch_packed_mla_mme_gaudi2", batch_packed<false>);
    m.impl("custom_deepseek_v41_batch_paged_mla_mme_gaudi2", batch_paged<false>);
    m.impl("custom_deepseek_v41_paged_mla_direct_mme_gaudi2", paged_direct<false>);
    m.impl("custom_deepseek_v41_paged_mla_woa_wob_fp8_roundtrip_gaudi2", paged_projection<false>);
    m.impl("custom_deepseek_v41_prefill_mla_mme_gaudi2", prefill<false>);
    m.impl("custom_deepseek_v41_paged_mla_mme_gaudi2", paged<false>);
    m.impl("custom_deepseek_v41_selected_mla_mme_gaudi2", selected<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_batch_packed_vector_mla_mme_gaudi2", batch_packed_vector<true>);
    m.impl("custom_deepseek_v41_batch_packed_sram_mla_mme_gaudi2", batch_packed_sram<true>);
    m.impl("custom_deepseek_v41_batch_packed_mla_mme_gaudi2", batch_packed<true>);
    m.impl("custom_deepseek_v41_batch_paged_mla_mme_gaudi2", batch_paged<true>);
    m.impl("custom_deepseek_v41_paged_mla_direct_mme_gaudi2", paged_direct<true>);
    m.impl("custom_deepseek_v41_paged_mla_woa_wob_fp8_roundtrip_gaudi2", paged_projection<true>);
    m.impl("custom_deepseek_v41_prefill_mla_mme_gaudi2", prefill<true>);
    m.impl("custom_deepseek_v41_paged_mla_mme_gaudi2", paged<true>);
    m.impl("custom_deepseek_v41_selected_mla_mme_gaudi2", selected<true>);
}
