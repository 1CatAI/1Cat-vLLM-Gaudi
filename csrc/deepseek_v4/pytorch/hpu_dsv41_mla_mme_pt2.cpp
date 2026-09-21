// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_mla_mme_gaudi2";
constexpr auto kBf16Schema = "custom_op::custom_deepseek_v41_mla_bf16_pv_gaudi2";
constexpr auto kMlaWoaSchema = "custom_op::custom_deepseek_v41_mla_woa_fp8_roundtrip_gaudi2";
constexpr auto kMlaWoaWobSchema =
    "custom_op::custom_deepseek_v41_mla_woa_wob_fp8_roundtrip_gaudi2";
constexpr auto kSelectedMlaWoaWobSchema =
    "custom_op::custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2";
constexpr auto kProductRopeQuant = "custom_deepseek_v41_mla_product_rope_quant_gaudi2";
constexpr auto kWoaScaleRoundtrip = "custom_deepseek_v41_woa_scale_roundtrip_gaudi2";
constexpr auto kDenseQuant = "custom_deepseek_v41_dense_quant_gaudi2";
constexpr auto kDenseScale = "custom_deepseek_v41_dense_scale_gaudi2";
struct Params { int32_t offset, rows, prefix_rows; };
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto q = s.at(0).toTensor(), swa = s.at(1).toTensor(), kv = s.at(2).toTensor();
    const auto ids = s.at(3).toTensor();
    for (int i = 0; i < 9; ++i) {
        const auto x = s.at(i).toTensor();
        const auto type = i < 3 ? at::kBFloat16 : (i == 4 || i == 5 ? at::kFloat : at::kInt);
        TORCH_CHECK(x.scalar_type() == type && x.device() == q.device() && x.is_contiguous() && !x.requires_grad(),
                    "MME MLA requires matching contiguous inference tensors");
    }
    const auto offset = s.at(9).toInt(), rows = s.at(10).toInt(), prefix_rows = s.at(11).toInt();
    TORCH_CHECK(q.dim() == 3 && q.size(0) == 1 && q.size(1) > 0 && q.size(1) <= 64 && q.size(2) == 512 &&
                swa.dim() == 2 && swa.size(1) == 512 && kv.dim() == 2 && kv.size(1) == 512 &&
                offset >= 0 && offset % 512 == 0 && offset <= swa.size(0) - 512 &&
                rows >= 0 && rows <= kv.size(0) && rows <= 2560 &&
                (prefix_rows == 256 || prefix_rows == 512) &&
                ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 && ids.size(1) <= 640 && ids.size(1) % 64 == 0 &&
                s.at(4).toTensor().sizes() == at::IntArrayRef({q.size(1)}) &&
                s.at(5).toTensor().sizes() == at::IntArrayRef({1}) &&
                s.at(6).toTensor().sizes() == at::IntArrayRef({1}), "Invalid C1 MME MLA shape or cache range");
    for (int i : {7,8}) TORCH_CHECK(s.at(i).toTensor().dim() == 1 && s.at(i).toTensor().numel() > 0,
                                    "MME MLA requires actual cache writer completions");
    return {{at::kBFloat16, q.sizes().vec()}};
}
habana::OutputMetaDataVector mla_woa_meta(const at::Stack& s) {
    (void)meta(s);
    const auto q = s.at(0).toTensor(), w = s.at(12).toTensor();
    const auto channel_scale = s.at(13).toTensor();
    const auto positions = s.at(14).toTensor(), phase = s.at(15).toTensor();
    TORCH_CHECK(q.size(0) == 1,
                "Fused MLA/wo_a is currently the C1 production contract");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                w.sizes() == at::IntArrayRef({4,4096,1024}) &&
                channel_scale.scalar_type() == at::kFloat &&
                channel_scale.sizes() == at::IntArrayRef({4,1,1024}),
                "Fused MLA/wo_a requires prepared FP8 wo_a weights and channel scales");
    TORCH_CHECK(positions.scalar_type() == at::kInt &&
                positions.sizes() == at::IntArrayRef({1}) &&
                phase.scalar_type() == at::kFloat && phase.dim() == 2 &&
                phase.size(1) == 64,
                "Fused MLA/wo_a requires I32 [1] position and F32 [L,64] inverse phase");
    for (int i = 12; i < 16; ++i) {
        const auto x = s.at(i).toTensor();
        TORCH_CHECK(x.device() == q.device() && x.is_contiguous() && !x.requires_grad(),
                    "Fused MLA/wo_a requires matching contiguous inference tensors");
    }
    return {{at::kBFloat16, {1,4096}}};
}
habana::OutputMetaDataVector mla_woa_wob_meta(const at::Stack& s) {
    (void)mla_woa_meta(s);
    const auto w = s.at(16).toTensor();
    const auto scale = s.at(17).toTensor();
    const auto q = s.at(0).toTensor();
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                w.sizes() == at::IntArrayRef({5120,4096}) &&
                scale.scalar_type() == at::kFloat &&
                scale.sizes() == at::IntArrayRef({1,5120}) &&
                w.device() == q.device() && scale.device() == q.device() &&
                w.is_contiguous() && scale.is_contiguous() &&
                !w.requires_grad() && !scale.requires_grad(),
                "Fused MLA/wo_a/wo_b requires prepared FP8 wo_b weights "
                "and F32 channel scales");
    return {{at::kBFloat16, {1,5120}}};
}
habana::OutputMetaDataVector selected_mla_woa_wob_meta(
    const at::Stack& s) {
    const auto q = s.at(0).toTensor();
    const auto swa = s.at(1).toTensor();
    const auto kv = s.at(2).toTensor();
    const auto selected = s.at(3).toTensor();
    const auto sink = s.at(4).toTensor();
    const auto scale = s.at(5).toTensor();
    const auto swa_done = s.at(6).toTensor();
    const auto main_done = s.at(7).toTensor();
    const auto offset = s.at(8).toInt();
    const auto rows = s.at(9).toInt();
    TORCH_CHECK(
        q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(0) == 1 &&
        q.size(1) > 0 && q.size(1) <= 64 && q.size(2) == 512 &&
        swa.scalar_type() == at::kBFloat16 && swa.dim() == 2 &&
        swa.size(1) == 512 && kv.scalar_type() == at::kBFloat16 &&
        kv.dim() == 2 && kv.size(1) == 512 &&
        selected.scalar_type() == at::kInt &&
        selected.sizes() == at::IntArrayRef({1, 512}) &&
        sink.scalar_type() == at::kFloat && sink.sizes() == at::IntArrayRef({q.size(1)}) &&
        scale.scalar_type() == at::kFloat && scale.sizes() == at::IntArrayRef({1}) &&
        swa_done.scalar_type() == at::kInt && swa_done.dim() == 1 &&
        swa_done.numel() > 0 && main_done.scalar_type() == at::kInt &&
        main_done.dim() == 1 && main_done.numel() > 0 && offset >= 0 &&
        offset % 512 == 0 && offset <= swa.size(0) - 512 && rows >= 0 &&
        rows <= kv.size(0) && rows <= 2560,
        "Selected-prefix MLA requires C1 decoded cache and [1,512] selections");
    for (int i = 0; i < 8; ++i) {
        const auto value = s.at(i).toTensor();
        TORCH_CHECK(value.device() == q.device() && value.is_contiguous() &&
                        !value.requires_grad(),
                    "Selected-prefix MLA requires contiguous inference tensors");
    }
    const auto woa = s.at(10).toTensor();
    const auto woa_scale = s.at(11).toTensor();
    const auto positions = s.at(12).toTensor();
    const auto phase = s.at(13).toTensor();
    const auto wob = s.at(14).toTensor();
    const auto wob_scale = s.at(15).toTensor();
    TORCH_CHECK(
        woa.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        woa.sizes() == at::IntArrayRef({4,4096,1024}) &&
        woa_scale.scalar_type() == at::kFloat &&
        woa_scale.sizes() == at::IntArrayRef({4,1,1024}) &&
        positions.scalar_type() == at::kInt &&
        positions.sizes() == at::IntArrayRef({1}) &&
        phase.scalar_type() == at::kFloat && phase.dim() == 2 &&
        phase.size(1) == 64 &&
        wob.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        wob.sizes() == at::IntArrayRef({5120,4096}) &&
        wob_scale.scalar_type() == at::kFloat &&
        wob_scale.sizes() == at::IntArrayRef({1,5120}),
        "Selected-prefix MLA requires prepared FP8 output projections");
    for (int i = 10; i < 16; ++i) {
        const auto value = s.at(i).toTensor();
        TORCH_CHECK(value.device() == q.device() && value.is_contiguous() &&
                        !value.requires_grad(),
                    "Selected-prefix MLA projection tensors must share the device");
    }
    return {{at::kBFloat16, {1,5120}}};
}
class Mla final : public habana::OpBackend {
    bool bf16_;
public:
    Mla(int device, c10::ScalarType dtype, bool bf16)
        : OpBackend(device, NO_TPC + std::string(bf16 ? "dsv41_mla_bf16_pv" : "dsv41_mla_mme"),
                    dtype, {0}, {}, {}, false), bf16_(bf16) { SetOutputMetaFn(meta); }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta(s);
        const auto heads = s.at(0).toTensor().size(1), width = s.at(3).toTensor().size(1);
        Params params{int32_t(s.at(9).toInt()), int32_t(s.at(10).toInt()), int32_t(s.at(11).toInt())};
        auto kv = bf16_ ? BuildNode(this, graph, {"custom_deepseek_v41_mla_shared_kv_gaudi2",
            {syn_in(1),syn_in(2),syn_in(3),syn_in(6),syn_in(7),syn_in(8)},
            {{{width,512},at::kBFloat16}, {{width},at::kFloat}}, &params,sizeof(params)}) :
            BuildNode(this, graph, {"custom_deepseek_v41_mla_gather_gaudi2",
            {syn_in(1),syn_in(2),syn_in(3),syn_in(6),syn_in(7),syn_in(8)},
            {{{width,512},at::kBFloat16}, {{width,512},at::kFloat}, {{width},at::kFloat}}, &params,sizeof(params)});
        auto q = ReshapeHelper(graph, syn_in(0), {heads,512}, at::kBFloat16);
        synGEMMParams qk_params{false,true};
        auto logits = BuildNode(this, graph, {"gemm", {q.get(),kv.at(0).get()},
            {{{heads,width},at::kFloat}}, &qk_params,sizeof(qk_params)});
        auto probabilities = bf16_ ? BuildNode(this, graph, {"custom_deepseek_v41_mla_exp_bf16_gaudi2",
            {logits.at(0).get(),kv.at(1).get(),syn_in(4),syn_in(5)},
            {{{heads,width},at::kBFloat16},{{heads,1},at::kFloat}}}) :
            BuildNode(this, graph, {"custom_deepseek_v41_mla_softmax_gaudi2",
            {logits.at(0).get(),kv.at(2).get(),syn_in(4),syn_in(5)}, {{{heads,width},at::kFloat}}});
        synGEMMParams pv_params{false,false};
        auto product = BuildNode(this, graph, {"gemm", {probabilities.at(0).get(),kv.at(bf16_ ? 0 : 1).get()},
            {{{heads,512},at::kFloat}}, &pv_params,sizeof(pv_params)});
        auto converted = bf16_ ? BuildNode(this, graph, {"custom_deepseek_v41_mla_normalize_bf16_gaudi2",
            {product.at(0).get(),probabilities.at(1).get()}, {{{heads,512},at::kBFloat16}}}) :
            BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()}, {{{heads,512},at::kBFloat16}}});
        syn_out(0) = ReshapeHelper(graph, converted.at(0).get(), output.at(0).shape, at::kBFloat16, 0);
    }
};
class MlaWoa final : public habana::OpBackend {
    bool wob_;
    bool selected_prefix_;
public:
    MlaWoa(int device, c10::ScalarType dtype, bool wob = false,
           bool selected_prefix = false)
        : OpBackend(device,
                    NO_TPC + std::string(
                        selected_prefix
                            ? "dsv41_mla_selected_woa_wob_fp8_roundtrip"
                            : (wob ? "dsv41_mla_woa_wob_fp8_roundtrip"
                                   : "dsv41_mla_woa_fp8_roundtrip")),
                    dtype, {0}, {}, {}, false),
          wob_(wob), selected_prefix_(selected_prefix) {
        SetOutputMetaFn([wob, selected_prefix](const at::Stack& s) {
            return selected_prefix ? selected_mla_woa_wob_meta(s)
                                   : (wob ? mla_woa_wob_meta(s)
                                          : mla_woa_meta(s));
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = selected_prefix_ ? selected_mla_woa_wob_meta(s)
                                             : (wob_ ? mla_woa_wob_meta(s)
                                                     : mla_woa_meta(s));
        const auto heads = s.at(0).toTensor().size(1);
        const int64_t width = selected_prefix_ ? 640 : s.at(3).toTensor().size(1);
        Params params{int32_t(s.at(selected_prefix_ ? 8 : 9).toInt()),
                      int32_t(s.at(selected_prefix_ ? 9 : 10).toInt()),
                      int32_t(selected_prefix_ ? 256 : s.at(11).toInt())};
        const int sink_input = 4, scale_input = 5;
        const int woa_input = selected_prefix_ ? 8 : 9;
        const int woa_scale_input = selected_prefix_ ? 9 : 10;
        const int position_input = selected_prefix_ ? 10 : 11;
        const int phase_input = selected_prefix_ ? 11 : 12;
        const int wob_input = selected_prefix_ ? 12 : 13;
        const int wob_scale_input = selected_prefix_ ? 13 : 14;
        std::vector<synapse_helpers::tensor> kv;
        if (selected_prefix_) {
            kv = BuildNode(this, graph,
                {"custom_deepseek_v41_mla_selected_prefix_gather_gaudi2",
                 {syn_in(1),syn_in(2),syn_in(3),syn_in(position_input),
                  syn_in(6),syn_in(7)},
                 {{{width,512},at::kBFloat16},
                  {{width,512},at::kFloat},{{width},at::kFloat}},
                 &params,sizeof(params)});
        } else {
            kv = BuildNode(this, graph,
                {"custom_deepseek_v41_mla_gather_gaudi2",
                 {syn_in(1),syn_in(2),syn_in(3),syn_in(6),syn_in(7),syn_in(8)},
                 {{{width,512},at::kBFloat16},
                  {{width,512},at::kFloat},{{width},at::kFloat}},
                 &params,sizeof(params)});
        }
        auto q = ReshapeHelper(graph, syn_in(0), {heads,512}, at::kBFloat16);
        synGEMMParams qk_params{false,true};
        auto logits = BuildNode(this, graph, {"gemm", {q.get(),kv.at(0).get()},
            {{{heads,width},at::kFloat}}, &qk_params,sizeof(qk_params)});
        auto probabilities = BuildNode(this, graph,
            {"custom_deepseek_v41_mla_softmax_gaudi2",
             {logits.at(0).get(),kv.at(2).get(),syn_in(sink_input),
              syn_in(scale_input)},
             {{{heads,width},at::kFloat}}});
        synGEMMParams pv_params{false,false};
        auto pv = BuildNode(this, graph, {"gemm",
            {probabilities.at(0).get(),kv.at(1).get()},
            {{{heads,512},at::kFloat}}, &pv_params,sizeof(pv_params)});
        // syn_in() indexes tensor operands only; the three scalar shape
        // arguments at stack positions 9..11 are not represented there.
        auto quantized = BuildNode(this, graph, {kProductRopeQuant,
            {pv.at(0).get(),syn_in(position_input),syn_in(phase_input)},
            {{{4,1,4096},at::ScalarType::Float8_e4m3fn},
             {{4,1,1},at::kFloat}}});
        synGEMMParams woa_params{false,false};
        auto product = BuildNode(this, graph, {"batch_gemm",
            {quantized.at(0).get(),syn_in(woa_input)},
            {{{4,1,1024},at::kFloat}}, &woa_params,sizeof(woa_params)});
        auto scaled = BuildNode(this, graph, {kWoaScaleRoundtrip,
            {product.at(0).get(),syn_in(woa_scale_input),quantized.at(1).get()},
            {{{1,4,1024},at::kBFloat16}}});
        if (!wob_) {
            syn_out(0) = ReshapeHelper(graph, scaled.at(0).get(),
                                       output.at(0).shape, at::kBFloat16, 0);
            return;
        }
        // Preserve both numerical boundaries exactly, but make the wo_a
        // group-32 BF16 result an internal compiler temporary. The dense
        // quantizer still observes the same BF16 values and scale selection;
        // wo_b receives the same FP8 operand without forcing the 8 KiB row
        // through a graph output between the two native compound operators.
        auto woa = ReshapeHelper(graph, scaled.at(0).get(), {1,4096},
                                 at::kBFloat16);
        auto dense_q = BuildNode(this, graph, {kDenseQuant, {woa.get()},
            {{{1,4096},at::ScalarType::Float8_e4m3fn},
             {{1,1},at::kFloat}}});
        synGEMMParams wob_params{false,true};
        auto wob_product = BuildNode(this, graph, {"gemm",
            {dense_q.at(0).get(),syn_in(wob_input)},
            {{{1,5120},at::kFloat}}, &wob_params,sizeof(wob_params)});
        auto wob_scaled = BuildNode(this, graph, {kDenseScale,
            {wob_product.at(0).get(),syn_in(wob_scale_input),dense_q.at(1).get()},
            {{{1,5120},at::kBFloat16,0}}});
        syn_out(0) = std::move(wob_scaled.at(0));
    }
};
const bool registered = [] {
    for (const bool bf16 : {false,true}) {
    const auto schema = bf16 ? kBf16Schema : kSchema;
    habana::custom_op::registerUserCustomOp(schema, "gemm", [](const at::Stack& s) {
        const auto m = meta(s); return habana::PartialOutputMetaDataVector{{m.at(0).dtype,m.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [bf16](synDeviceId d, c10::ScalarType t) { return std::make_shared<Mla>(d,t,bf16); });
    }
    habana::custom_op::registerUserCustomOp(kMlaWoaSchema, "gemm", [](const at::Stack& s) {
        const auto m = mla_woa_meta(s);
        return habana::PartialOutputMetaDataVector{{m.at(0).dtype,m.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kMlaWoaSchema,
        [](synDeviceId d, c10::ScalarType t) { return std::make_shared<MlaWoa>(d,t); });
    habana::custom_op::registerUserCustomOp(kMlaWoaWobSchema, "gemm", [](const at::Stack& s) {
        const auto m = mla_woa_wob_meta(s);
        return habana::PartialOutputMetaDataVector{{m.at(0).dtype,m.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kMlaWoaWobSchema,
        [](synDeviceId d, c10::ScalarType t) { return std::make_shared<MlaWoa>(d,t,true); });
    habana::custom_op::registerUserCustomOp(
        kSelectedMlaWoaWobSchema, "gemm", [](const at::Stack& s) {
            const auto m = selected_mla_woa_wob_meta(s);
            return habana::PartialOutputMetaDataVector{
                {m.at(0).dtype,m.at(0).shape}};
        }, nullptr);
    habana::KernelRegistry().add(
        kSelectedMlaWoaWobSchema,
        [](synDeviceId d, c10::ScalarType t) {
            return std::make_shared<MlaWoa>(d,t,true,true);
        });
    return true;
}();
template<bool Meta, bool Bf16 = false> at::Tensor run(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& kv,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths,
    const at::Tensor& swa_done, const at::Tensor& main_done, int64_t offset, int64_t rows, int64_t prefix_rows) {
    at::Stack s{q,swa,kv,ids,sink,scale,lengths,swa_done,main_done,offset,rows,prefix_rows};
    const auto m = meta(s);
    if (Meta) return at::empty(m.at(0).shape,q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Bf16 ? kBf16Schema : kSchema);
    return descriptor.execute(s).at(0);
}
template<bool Meta> at::Tensor mla_woa_run(
    const at::Tensor& q, const at::Tensor& swa, const at::Tensor& kv,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths, const at::Tensor& swa_done,
    const at::Tensor& main_done, int64_t offset, int64_t rows,
    int64_t prefix_rows, const at::Tensor& woa,
    const at::Tensor& woa_scale, const at::Tensor& positions,
    const at::Tensor& phase) {
    at::Stack s{q,swa,kv,ids,sink,scale,lengths,swa_done,main_done,
                offset,rows,prefix_rows,woa,woa_scale,positions,phase};
    const auto m = mla_woa_meta(s);
    if (Meta) return at::empty(m.at(0).shape,q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMlaWoaSchema);
    return descriptor.execute(s).at(0);
}
template<bool Meta> at::Tensor mla_woa_wob_run(
    const at::Tensor& q, const at::Tensor& swa, const at::Tensor& kv,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale,
    const at::Tensor& lengths, const at::Tensor& swa_done,
    const at::Tensor& main_done, int64_t offset, int64_t rows,
    int64_t prefix_rows, const at::Tensor& woa,
    const at::Tensor& woa_scale, const at::Tensor& positions,
    const at::Tensor& phase, const at::Tensor& wob,
    const at::Tensor& wob_scale) {
    at::Stack s{q,swa,kv,ids,sink,scale,lengths,swa_done,main_done,
                offset,rows,prefix_rows,woa,woa_scale,positions,phase,
                wob,wob_scale};
    const auto m = mla_woa_wob_meta(s);
    if (Meta) return at::empty(m.at(0).shape,q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::
        getUserCustomOpDescriptor(kMlaWoaWobSchema);
    return descriptor.execute(s).at(0);
}
template<bool Meta> at::Tensor selected_mla_woa_wob_run(
    const at::Tensor& q, const at::Tensor& swa, const at::Tensor& kv,
    const at::Tensor& selected, const at::Tensor& sink,
    const at::Tensor& scale, const at::Tensor& swa_done,
    const at::Tensor& main_done, int64_t offset, int64_t rows,
    const at::Tensor& woa, const at::Tensor& woa_scale,
    const at::Tensor& positions, const at::Tensor& phase,
    const at::Tensor& wob, const at::Tensor& wob_scale) {
    at::Stack s{q,swa,kv,selected,sink,scale,swa_done,main_done,
                offset,rows,woa,woa_scale,positions,phase,wob,wob_scale};
    const auto m = selected_mla_woa_wob_meta(s);
    if (Meta) return at::empty(m.at(0).shape,q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::
        getUserCustomOpDescriptor(kSelectedMlaWoaWobSchema);
    return descriptor.execute(s).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, "
          "Tensor scale, Tensor lengths, Tensor swa_completion, Tensor main_completion, int swa_offset, int main_rows, "
          "int prefix_rows) -> Tensor");
    m.def("custom_deepseek_v41_mla_bf16_pv_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, "
          "Tensor scale, Tensor lengths, Tensor swa_completion, Tensor main_completion, int swa_offset, int main_rows, "
          "int prefix_rows) -> Tensor");
    m.def("custom_deepseek_v41_mla_woa_fp8_roundtrip_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, "
          "Tensor sink, Tensor scale, Tensor lengths, Tensor swa_completion, Tensor main_completion, "
          "int swa_offset, int main_rows, int prefix_rows, Tensor woa_weight, Tensor woa_channel_scale, "
          "Tensor positions, Tensor inverse_phase) -> Tensor");
    m.def("custom_deepseek_v41_mla_woa_wob_fp8_roundtrip_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, "
          "Tensor sink, Tensor scale, Tensor lengths, Tensor swa_completion, Tensor main_completion, "
          "int swa_offset, int main_rows, int prefix_rows, Tensor woa_weight, Tensor woa_channel_scale, "
          "Tensor positions, Tensor inverse_phase, Tensor wob_weight, Tensor wob_channel_scale) -> Tensor");
    m.def("custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2(Tensor q, Tensor swa, Tensor main, "
          "Tensor selected, Tensor sink, Tensor scale, Tensor swa_completion, Tensor main_completion, "
          "int swa_offset, int main_rows, Tensor woa_weight, Tensor woa_channel_scale, Tensor positions, "
          "Tensor inverse_phase, Tensor wob_weight, Tensor wob_channel_scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_mla_mme_gaudi2",run<false>);
    m.impl("custom_deepseek_v41_mla_bf16_pv_gaudi2",run<false,true>);
    m.impl("custom_deepseek_v41_mla_woa_fp8_roundtrip_gaudi2",mla_woa_run<false>);
    m.impl("custom_deepseek_v41_mla_woa_wob_fp8_roundtrip_gaudi2",mla_woa_wob_run<false>);
    m.impl("custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2",
           selected_mla_woa_wob_run<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_mla_mme_gaudi2",run<true>);
    m.impl("custom_deepseek_v41_mla_bf16_pv_gaudi2",run<true,true>);
    m.impl("custom_deepseek_v41_mla_woa_fp8_roundtrip_gaudi2",mla_woa_run<true>);
    m.impl("custom_deepseek_v41_mla_woa_wob_fp8_roundtrip_gaudi2",mla_woa_wob_run<true>);
    m.impl("custom_deepseek_v41_mla_selected_woa_wob_fp8_roundtrip_gaudi2",
           selected_mla_woa_wob_run<true>);
}
