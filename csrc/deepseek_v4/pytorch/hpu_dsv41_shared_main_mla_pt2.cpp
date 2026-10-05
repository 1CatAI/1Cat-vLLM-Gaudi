// SPDX-License-Identifier: Apache-2.0
// Share only selected main rows within one C1 layer group; SWA stays per layer.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto publish_name = "custom_op::custom_deepseek_v41_main_publish_mla_gaudi2";
constexpr auto reuse_name = "custom_op::custom_deepseek_v41_main_reuse_mla_gaudi2";
constexpr auto publish_projection_name = "custom_op::custom_deepseek_v41_main_publish_projection_gaudi2";
constexpr auto reuse_projection_name = "custom_op::custom_deepseek_v41_main_reuse_projection_gaudi2";
template<bool Reuse, bool Projection = false> habana::OutputMetaDataVector meta(const at::Stack& s) {
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
    if constexpr (Projection) {
        constexpr int base = Reuse ? 8 : 10;
        const int64_t groups = q.size(1) / 8;
        const auto wa = s.at(base).toTensor(), sa = s.at(base+1).toTensor();
        const auto phase = s.at(base+2).toTensor(), wb = s.at(base+3).toTensor(), sb = s.at(base+4).toTensor();
        TORCH_CHECK((groups == 2 || groups == 4) && q.size(1) == groups * 8 &&
            wa.scalar_type() == at::ScalarType::Float8_e4m3fn && wa.sizes() == at::IntArrayRef({groups,4096,1024}) &&
            sa.scalar_type() == at::kFloat && sa.sizes() == at::IntArrayRef({groups,1,1024}) &&
            phase.scalar_type() == at::kFloat && phase.dim() == 2 && phase.size(1) == 64 && phase.size(0) > 0 &&
            wb.scalar_type() == at::ScalarType::Float8_e4m3fn && wb.sizes() == at::IntArrayRef({5120,groups*1024}) &&
            sb.scalar_type() == at::kFloat && sb.sizes() == at::IntArrayRef({1,5120}),
            "Shared MLA projection requires prepared FP8 WO weights and an FP32 rotary table");
        for (int i=base; i<base+5; ++i) {
            const auto t=s.at(i).toTensor();
            TORCH_CHECK(t.device()==q.device() && t.is_contiguous() && !t.requires_grad(),
                        "Shared MLA projection operands must be contiguous inference tensors on one device");
        }
    }
    const std::vector<int64_t> shape = Projection ? std::vector<int64_t>{1,5120} : q.sizes().vec();
    if constexpr (Reuse) {
        TORCH_CHECK(main.sizes() == at::IntArrayRef({1, 640, 512}) &&
                    s.at(3).toTensor().sizes() == at::IntArrayRef({1, 640}), "Invalid shared main rows or mask");
        return {{at::kBFloat16, shape}};
    } else {
        const auto pages = s.at(5).toTensor();
        TORCH_CHECK(main.dim() == 2 && main.size(1) == 288 && main.size(0) > 0 && main.size(0) <= 0x7ffffdffLL &&
                    s.at(3).toTensor().sizes() == at::IntArrayRef({1, 512}) && pages.dim() == 1 &&
                    pages.numel() > 0 && pages.numel() <= 8192 && (s.at(9).toInt() == 1 || s.at(9).toInt() == 2),
                    "Invalid packed main, selection, page table or ratio");
        return {{at::kBFloat16, shape}, {at::kBFloat16, {1, 640, 512}}, {at::kFloat, {1, 640}}};
    }
}
template<bool Reuse, bool Projection = false> class SharedMainMla final : public habana::OpBackend {
public:
    SharedMainMla(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string(Reuse ? "dsv41_main_reuse" : "dsv41_main_publish"),
                    type, {0}, {}, {}, false) { SetOutputMetaFn(meta<Reuse, Projection>); }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta<Reuse, Projection>(s);
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
        if constexpr (Projection) {
            // Synapse operand indices omit the publish schema's integer ratio.
            constexpr int base = Reuse ? 8 : 9;
            const int64_t groups = heads / 8;
            auto product_row = ReshapeHelper(graph, product.at(0).get(), {heads,512}, at::kFloat);
            // The hand-written consumer preserves PV->BF16 and inverse-RoPE->BF16
            // roundings in registers before emitting WOa's FP8 input.
            auto quantized = BuildNode(this, graph, {"custom_deepseek_v41_mla_product_rope_quant_gaudi2",
                {product_row.get(), syn_in(4), syn_in(base+2)},
                {{{groups,1,4096},at::ScalarType::Float8_e4m3fn},{{groups,1,1},at::kFloat}}});
            synGEMMParams wa_params{false,false}, wb_params{false,true};
            auto wa = BuildNode(this, graph, {"batch_gemm",{quantized.at(0).get(),syn_in(base)},
                {{{groups,1,1024},at::kFloat}},&wa_params,sizeof(wa_params)});
            auto rounded = BuildNode(this, graph, {"custom_deepseek_v41_woa_scale_roundtrip_gaudi2",
                {wa.at(0).get(),syn_in(base+1),quantized.at(1).get()},{{{1,groups,1024},at::kBFloat16}}});
            auto row = ReshapeHelper(graph, rounded.at(0).get(), {1,groups*1024}, at::kBFloat16);
            auto dense = BuildNode(this, graph, {"custom_deepseek_v41_dense_quant_gaudi2",{row.get()},
                {{{1,groups*1024},at::ScalarType::Float8_e4m3fn},{{1,1},at::kFloat}}});
            auto wb = BuildNode(this, graph, {"gemm",{dense.at(0).get(),syn_in(base+3)},
                {{{1,5120},at::kFloat}},&wb_params,sizeof(wb_params)});
            syn_out(0)=std::move(BuildNode(this,graph,{"custom_deepseek_v41_dense_scale_gaudi2",
                {wb.at(0).get(),syn_in(base+4),dense.at(1).get()},{{output.at(0).shape,at::kBFloat16,0}}}).at(0));
        } else {
        syn_out(0) = std::move(BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()},
            {{output.at(0).shape, at::kBFloat16, 0}}}).at(0));
        }
        if constexpr (!Reuse) {
            syn_out(1) = std::move(kv.at(3));
            syn_out(2) = std::move(kv.at(2));
        }
    }
};
template<bool Reuse, bool Projection = false> bool register_op() {
    const auto name = Projection ? (Reuse ? reuse_projection_name : publish_projection_name) : (Reuse ? reuse_name : publish_name);
    habana::custom_op::registerUserCustomOp(name, "batch_gemm", [](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for (const auto& item : meta<Reuse, Projection>(s)) out.push_back({item.dtype, item.shape});
        return out;
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId d, c10::ScalarType t) {
        return std::make_shared<SharedMainMla<Reuse, Projection>>(d, t);
    });
    return true;
}
const bool registered = register_op<false>() && register_op<true>() && register_op<false,true>() && register_op<true,true>();
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
template<bool Meta, bool Reuse> std::vector<at::Tensor> projection_execute(const at::Stack& stack) {
    const auto out=meta<Reuse,true>(stack); const auto q=stack.at(0).toTensor();
    if constexpr (Meta) {
        std::vector<at::Tensor> result;
        for (const auto& m:out) result.push_back(at::empty(m.shape,q.options().dtype(m.dtype)));
        return result;
    }
    TORCH_CHECK(registered && q.device().type()==at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        Reuse ? reuse_projection_name : publish_projection_name);
    return descriptor.execute(stack);
}
template<bool Meta> at::Tensor reuse_projection(
    const at::Tensor& q,const at::Tensor& swa,const at::Tensor& main,const at::Tensor& mask,
    const at::Tensor& pos,const at::Tensor& sink,const at::Tensor& scale,const at::Tensor& lengths,
    const at::Tensor& wa,const at::Tensor& sa,const at::Tensor& phase,const at::Tensor& wb,const at::Tensor& sb) {
    return projection_execute<Meta,true>({q,swa,main,mask,pos,sink,scale,lengths,wa,sa,phase,wb,sb}).at(0);
}
template<bool Meta> std::tuple<at::Tensor,at::Tensor,at::Tensor> publish_projection(
    const at::Tensor& q,const at::Tensor& swa,const at::Tensor& main,const at::Tensor& selected,
    const at::Tensor& pos,const at::Tensor& pages,const at::Tensor& sink,const at::Tensor& scale,
    const at::Tensor& lengths,int64_t ratio,const at::Tensor& wa,const at::Tensor& sa,
    const at::Tensor& phase,const at::Tensor& wb,const at::Tensor& sb) {
    auto out=projection_execute<Meta,false>({q,swa,main,selected,pos,pages,sink,scale,lengths,ratio,wa,sa,phase,wb,sb});
    return {out.at(0),out.at(1),out.at(2)};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_main_publish_projection_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio, Tensor wa, Tensor sa, Tensor phase, Tensor wb, Tensor sb) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_main_reuse_projection_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths, Tensor wa, Tensor sa, Tensor phase, Tensor wb, Tensor sb) -> Tensor");
    m.def("custom_deepseek_v41_main_publish_mla_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor selected, Tensor positions, Tensor pages, Tensor sink, Tensor scale, Tensor lengths, int ratio) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_main_reuse_mla_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor mask, Tensor positions, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_main_publish_projection_gaudi2",publish_projection<false>);
    m.impl("custom_deepseek_v41_main_reuse_projection_gaudi2",reuse_projection<false>);
    m.impl("custom_deepseek_v41_main_publish_mla_gaudi2", publish<false>);
    m.impl("custom_deepseek_v41_main_reuse_mla_gaudi2", reuse<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_main_publish_projection_gaudi2",publish_projection<true>);
    m.impl("custom_deepseek_v41_main_reuse_projection_gaudi2",reuse_projection<true>);
    m.impl("custom_deepseek_v41_main_publish_mla_gaudi2", publish<true>);
    m.impl("custom_deepseek_v41_main_reuse_mla_gaudi2", reuse<true>);
}
