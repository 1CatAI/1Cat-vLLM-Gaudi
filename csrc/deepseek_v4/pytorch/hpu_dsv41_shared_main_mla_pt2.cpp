// SPDX-License-Identifier: Apache-2.0
// Share only selected main rows within one C1 layer group; SWA stays per layer.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto publish_name = "custom_op::custom_deepseek_v41_main_publish_mla_gaudi2";
constexpr auto reuse_name = "custom_op::custom_deepseek_v41_main_reuse_mla_gaudi2";
template<bool Reuse> habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto q = s.at(0).toTensor(), swa = s.at(1).toTensor(), main = s.at(2).toTensor();
    TORCH_CHECK(q.dim() == 3 && q.size(0) == 1 && q.size(1) >= 1 && q.size(1) <= 64 && q.size(2) == 512,
                "Shared main MLA requires C1 Q [1,H,512]");
    for (int i = 0; i < (Reuse ? 8 : 9); ++i) {
        const auto t = s.at(i).toTensor();
        const auto type = Reuse ? (i == 0 || i == 2 ? at::kBFloat16 : i == 1 ? at::kByte :
                                  i == 3 || i == 5 || i == 6 ? at::kFloat : at::kInt) :
                                (i == 0 ? at::kBFloat16 : i <= 2 ? at::kByte :
                                  i == 6 || i == 7 ? at::kFloat : at::kInt);
        TORCH_CHECK(t.scalar_type() == type && t.device() == q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Shared main MLA requires matching contiguous inference operands");
    }
    TORCH_CHECK(swa.sizes() == at::IntArrayRef({256, 528}), "Invalid SWA ring");
    TORCH_CHECK(s.at(4).toTensor().sizes() == at::IntArrayRef({1}) &&
                s.at(Reuse ? 5 : 6).toTensor().sizes() == at::IntArrayRef({q.size(1)}) &&
                s.at(Reuse ? 6 : 7).toTensor().sizes() == at::IntArrayRef({1}) &&
                s.at(Reuse ? 7 : 8).toTensor().sizes() == at::IntArrayRef({1}),
                "Invalid position, sink, scale or length");
    if constexpr (Reuse) {
        TORCH_CHECK(main.sizes() == at::IntArrayRef({1, 640, 512}) &&
                    s.at(3).toTensor().sizes() == at::IntArrayRef({1, 640}), "Invalid shared main rows or mask");
        return {{at::kBFloat16, q.sizes().vec()}};
    } else {
        const auto pages = s.at(5).toTensor();
        TORCH_CHECK(main.dim() == 2 && main.size(1) == 288 && main.size(0) > 0 && main.size(0) <= 0x7ffffdffLL &&
                    s.at(3).toTensor().sizes() == at::IntArrayRef({1, 512}) && pages.dim() == 1 &&
                    pages.numel() > 0 && pages.numel() <= 8192 && (s.at(9).toInt() == 1 || s.at(9).toInt() == 2),
                    "Invalid packed main, selection, page table or ratio");
        return {{at::kBFloat16, q.sizes().vec()}, {at::kBFloat16, {1, 640, 512}}, {at::kFloat, {1, 640}}};
    }
}
template<bool Reuse> class SharedMainMla final : public habana::OpBackend {
public:
    SharedMainMla(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string(Reuse ? "dsv41_main_reuse" : "dsv41_main_publish"),
                    type, {0}, {}, {}, false) { SetOutputMetaFn(meta<Reuse>); }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta<Reuse>(s);
        const int64_t heads = s.at(0).toTensor().size(1);
        auto kv = [&] {
            if constexpr (Reuse) {
                return BuildNode(this, graph, {"custom_deepseek_v41_main_reuse_gather_gaudi2",
                    {syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(7)},
                    {{{1, 640, 512}, at::kBFloat16}, {{1, 640, 512}, at::kFloat}, {{1, 640}, at::kFloat}}});
            } else {
                int ratio = s.at(9).toInt();
                return BuildNode(this, graph, {"custom_deepseek_v41_main_publish_gather_gaudi2",
                    {syn_in(1), syn_in(2), syn_in(3), syn_in(4), syn_in(5), syn_in(8)},
                    {{{1, 640, 512}, at::kBFloat16}, {{1, 640, 512}, at::kFloat},
                     {{1, 640}, at::kFloat, 2}, {{1, 640, 512}, at::kBFloat16, 1}}, &ratio, sizeof(ratio)});
            }
        }();
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = BuildNode(this, graph, {"batch_gemm", {syn_in(0), kv.at(0).get()},
            {{{1, heads, 640}, at::kFloat}}, &qk, sizeof(qk)});
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_selected_mla_softmax_gaudi2",
            {scores.at(0).get(), kv.at(2).get(), syn_in(Reuse ? 5 : 6), syn_in(Reuse ? 6 : 7)},
            {{{1, heads, 640}, at::kFloat}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {probabilities.at(0).get(), kv.at(1).get()},
            {{{1, heads, 512}, at::kFloat}}, &pv, sizeof(pv)});
        syn_out(0) = std::move(BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()},
            {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
        if constexpr (!Reuse) {
            syn_out(1) = std::move(kv.at(3));
            syn_out(2) = std::move(kv.at(2));
        }
    }
};
template<bool Reuse> bool register_op() {
    const auto name = Reuse ? reuse_name : publish_name;
    habana::custom_op::registerUserCustomOp(name, "batch_gemm", [](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for (const auto& item : meta<Reuse>(s)) out.push_back({item.dtype, item.shape});
        return out;
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId d, c10::ScalarType t) {
        return std::make_shared<SharedMainMla<Reuse>>(d, t);
    });
    return true;
}
const bool registered = register_op<false>() && register_op<true>();
template<bool Meta> std::tuple<at::Tensor, at::Tensor, at::Tensor> publish(
    const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main, const at::Tensor& selected,
    const at::Tensor& positions, const at::Tensor& pages, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths, int64_t ratio) {
    const at::Stack stack{q, swa, main, selected, positions, pages, sink, scale, lengths, ratio};
    const auto out = meta<false>(stack);
    if (Meta) return {at::empty(out[0].shape, q.options()), at::empty(out[1].shape, q.options()),
                      at::empty(out[2].shape, q.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(publish_name);
    const auto result = descriptor.execute(stack);
    return {result.at(0), result.at(1), result.at(2)};
}
template<bool Meta> at::Tensor reuse(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& main,
    const at::Tensor& mask, const at::Tensor& positions, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths) {
    const at::Stack stack{q, swa, main, mask, positions, sink, scale, lengths};
    const auto out = meta<true>(stack);
    if (Meta) return at::empty(out[0].shape, q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(reuse_name);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_main_publish_mla_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_main_reuse_mla_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_main_publish_mla_gaudi2", publish<false>);
    m.impl("custom_deepseek_v41_main_reuse_mla_gaudi2", reuse<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_main_publish_mla_gaudi2", publish<true>);
    m.impl("custom_deepseek_v41_main_reuse_mla_gaudi2", reuse<true>);
}
