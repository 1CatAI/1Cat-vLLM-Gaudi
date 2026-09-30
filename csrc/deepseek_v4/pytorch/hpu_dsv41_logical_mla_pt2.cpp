// SPDX-License-Identifier: Apache-2.0
// Preserve selected MLA's QK/softmax/PV while absorbing its indirect layout producer.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto name = "custom_op::custom_deepseek_v41_logical_mla_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto q = s.at(0).toTensor(), swa = s.at(1).toTensor(), main = s.at(2).toTensor();
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= 6 && q.size(1) >= 1 &&
                q.size(1) <= 64 && q.size(2) == 512, "Logical MLA requires Q [T,H,512], T<=6");
    for (int i = 0; i < 9; ++i) {
        const auto t = s.at(i).toTensor();
        const auto type = i == 0 ? at::kBFloat16 : i <= 2 ? at::kByte : (i == 6 || i == 7) ? at::kFloat : at::kInt;
        TORCH_CHECK(t.scalar_type() == type && t.device() == q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Logical MLA requires matching contiguous inference operands");
    }
    const auto selected = s.at(3).toTensor(), positions = s.at(4).toTensor(), pages = s.at(5).toTensor();
    TORCH_CHECK(swa.sizes() == at::IntArrayRef({256, 528}) && main.dim() == 2 && main.size(1) == 288 &&
        main.size(0) > 0 && main.size(0) <= 0x7ffffdffLL &&
        selected.sizes() == at::IntArrayRef({q.size(0), 512}) && positions.sizes() == at::IntArrayRef({q.size(0)}) &&
        pages.dim() == 1 && pages.numel() > 0 && pages.numel() <= 8192 &&
        s.at(6).toTensor().sizes() == at::IntArrayRef({q.size(1)}) && s.at(7).toTensor().sizes() == at::IntArrayRef({1}) &&
        s.at(8).toTensor().sizes() == at::IntArrayRef({q.size(0)}) && (s.at(9).toInt() == 1 || s.at(9).toInt() == 2),
        "Invalid logical MLA cache, page, selection or scale contract");
    return {{at::kBFloat16, q.sizes().vec()}};
}
class LogicalMla final : public habana::OpBackend {
public:
    LogicalMla(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string("dsv41_logical_mla"), type, {0}, {}, {}, false) {
        SetOutputMetaFn(meta);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta(s);
        const auto q = s.at(0).toTensor();
        const int64_t tokens = q.size(0), heads = q.size(1), width = 640;
        int ratio = s.at(9).toInt();
        auto kv = BuildNode(this, graph, {"custom_deepseek_v41_logical_mla_gather_gaudi2",
            {syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(5), syn_in(8)},
            {{{tokens, width, 512}, at::kBFloat16}, {{tokens, width, 512}, at::kFloat}, {{tokens, width}, at::kFloat}},
            &ratio, sizeof(ratio)});
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = BuildNode(this, graph, {"batch_gemm", {syn_in(0), kv.at(0).get()},
            {{{tokens, heads, width}, at::kFloat}}, &qk, sizeof(qk)});
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_softmax_gaudi2",
            {scores.at(0).get(), kv.at(2).get(), syn_in(6), syn_in(7)}, {{{tokens, heads, width}, at::kFloat}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(), kv.at(1).get()},
            {{{tokens, heads, 512}, at::kFloat}}, &pv, sizeof(pv)});
        syn_out(0) = std::move(BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()},
            {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(name, "batch_gemm", [](const at::Stack& s) {
        const auto out = meta(s);
        return habana::PartialOutputMetaDataVector{{out.at(0).dtype, out.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId d, c10::ScalarType t) { return std::make_shared<LogicalMla>(d, t); });
    return true;
}();
template<bool Meta> at::Tensor execute(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& selected, const at::Tensor& positions, const at::Tensor& pages, const at::Tensor& sink,
    const at::Tensor& scale, const at::Tensor& lengths, int64_t ratio) {
    const at::Stack stack{q, swa, main, selected, positions, pages, sink, scale, lengths, ratio};
    const auto out = meta(stack);
    if (Meta) return at::empty(out.at(0).shape, q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_logical_mla_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("custom_deepseek_v41_logical_mla_gaudi2", execute<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("custom_deepseek_v41_logical_mla_gaudi2", execute<true>); }
