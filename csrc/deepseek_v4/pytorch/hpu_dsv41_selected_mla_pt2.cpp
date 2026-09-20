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
constexpr auto kSelected = "custom_op::custom_deepseek_v41_selected_mla_mme_gaudi2";
constexpr auto kPrefill = "custom_op::custom_deepseek_v41_prefill_mla_mme_gaudi2";

habana::OutputMetaDataVector meta(const at::Stack& s, bool paged, bool prefill = false) {
    const auto q = s.at(0).toTensor();
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= (prefill ? 64 : 6) &&
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
        TORCH_CHECK(cache.dim() == 2 && cache.size(0) > 0 && cache.size(0) <= 512 && cache.size(1) == 528 &&
                    main.dim() == 2 && main.size(0) > 0 && main.size(0) <= 0x7ffffdffLL && main.size(1) == 288 &&
                    rows.dim() == 2 && rows.size(0) == 1 && rows.size(1) > 0 && rows.size(1) <= 4096,
                    "Paged MLA requires SWA528, FP4 main288 and bounded selected row IDs");
    } else {
        TORCH_CHECK(cache.dim() == 2 && cache.size(0) > 0 &&
                    cache.size(0) <= (prefill ? 131072 : 4096) && cache.size(1) == 512,
                    "Selected MLA requires a bounded BF16 KV working set");
    }
    return {{at::kBFloat16, q.sizes().vec()}};
}

class SelectedMla final : public habana::OpBackend {
    bool paged_;
    bool prefill_;
public:
    SelectedMla(int device, c10::ScalarType dtype, bool paged, bool prefill = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_selected_mla_mme"), dtype, {0}, {}, {}, false),
          paged_(paged), prefill_(prefill) {
        SetOutputMetaFn([paged, prefill](const at::Stack& s) { return meta(s, paged, prefill); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta(s, paged_, prefill_);
        const auto q = s.at(0).toTensor();
        const int index = paged_ ? 4 : 2;
        const int64_t tokens = q.size(0), heads = q.size(1), width = s.at(index).toTensor().size(1);
        std::vector<synapse_helpers::tensor> selected;
        synTensor cache = syn_in(1);
        if (paged_) {
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
        auto kv = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_gather_gaudi2",
            {cache, syn_in(index), syn_in(index + 3)},
            {{{tokens, width, 512}, at::kBFloat16}, {{tokens, width, 512}, at::kFloat},
             {{tokens, width}, at::kFloat}}});
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = BuildNode(this, graph, {"batch_gemm", {syn_in(0), kv.at(0).get()},
            {{{tokens, heads, width}, at::kFloat}}, &qk, sizeof(qk)});
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_softmax_gaudi2",
            {scores.at(0).get(), kv.at(2).get(), syn_in(index + 1), syn_in(index + 2)},
            {{{tokens, heads, width}, at::kFloat}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(), kv.at(1).get()},
            {{{tokens, heads, 512}, at::kFloat}}, &pv, sizeof(pv)});
        syn_out(0) = std::move(BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()},
            {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
    }
};

const bool registered = [] {
    for (int kind : {0, 1, 2}) {
        const bool paged = kind == 1, prefill = kind == 2;
        const auto schema = prefill ? kPrefill : (paged ? kPaged : kSelected);
        habana::custom_op::registerUserCustomOp(schema, "batch_gemm", [paged, prefill](const at::Stack& s) {
            const auto out = meta(s, paged, prefill);
            return habana::PartialOutputMetaDataVector{{out.at(0).dtype, out.at(0).shape}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [paged, prefill](synDeviceId d, c10::ScalarType t) {
            return std::make_shared<SelectedMla>(d, t, paged, prefill);
        });
    }
    return true;
}();

template<bool Meta> at::Tensor execute(const at::Stack& s, bool paged, bool prefill = false) {
    const auto output = meta(s, paged, prefill);
    if (Meta) return at::empty(output.at(0).shape, s.at(0).toTensor().options());
    TORCH_CHECK(registered && s.at(0).toTensor().device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        prefill ? kPrefill : (paged ? kPaged : kSelected));
    return descriptor.execute(s).at(0);
}
template<bool Meta> at::Tensor paged(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& rows, const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) { return execute<Meta>({q, swa, main, rows, ids, sink, scale, lengths}, true); }
template<bool Meta> at::Tensor selected(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& ids,
    const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    return execute<Meta>({q, cache, ids, sink, scale, lengths}, false);
}
template<bool Meta> at::Tensor prefill(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& ids,
    const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    return execute<Meta>({q, cache, ids, sink, scale, lengths}, false, true);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_mla_mme_gaudi2(Tensor q, Tensor cache, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_paged_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor row_ids, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
    m.def("custom_deepseek_v41_selected_mla_mme_gaudi2(Tensor q, Tensor cache, Tensor indices, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_mla_mme_gaudi2", prefill<false>);
    m.impl("custom_deepseek_v41_paged_mla_mme_gaudi2", paged<false>);
    m.impl("custom_deepseek_v41_selected_mla_mme_gaudi2", selected<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_mla_mme_gaudi2", prefill<true>);
    m.impl("custom_deepseek_v41_paged_mla_mme_gaudi2", paged<true>);
    m.impl("custom_deepseek_v41_selected_mla_mme_gaudi2", selected<true>);
}
