// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_mla_mme_gaudi2";
struct Params { int32_t offset, rows; };
habana::OutputMetaDataVector meta(const at::Stack& s) {
    const auto q = s.at(0).toTensor(), swa = s.at(1).toTensor(), kv = s.at(2).toTensor();
    const auto ids = s.at(3).toTensor();
    for (int i = 0; i < 9; ++i) {
        const auto x = s.at(i).toTensor();
        const auto type = i < 3 ? at::kBFloat16 : (i == 4 || i == 5 ? at::kFloat : at::kInt);
        TORCH_CHECK(x.scalar_type() == type && x.device() == q.device() && x.is_contiguous() && !x.requires_grad(),
                    "MME MLA requires matching contiguous inference tensors");
    }
    const auto offset = s.at(9).toInt(), rows = s.at(10).toInt();
    TORCH_CHECK(q.dim() == 3 && q.size(0) == 1 && q.size(1) > 0 && q.size(1) <= 64 && q.size(2) == 512 &&
                swa.dim() == 2 && swa.size(1) == 512 && kv.dim() == 2 && kv.size(1) == 512 &&
                offset >= 0 && offset % 512 == 0 && offset <= swa.size(0) - 512 &&
                rows >= 0 && rows <= kv.size(0) && rows <= 512 &&
                ids.dim() == 2 && ids.size(0) == 1 && ids.size(1) > 0 && ids.size(1) <= 640 && ids.size(1) % 64 == 0 &&
                s.at(4).toTensor().sizes() == at::IntArrayRef({q.size(1)}) &&
                s.at(5).toTensor().sizes() == at::IntArrayRef({1}) &&
                s.at(6).toTensor().sizes() == at::IntArrayRef({1}), "Invalid C1 MME MLA shape or cache range");
    for (int i : {7,8}) TORCH_CHECK(s.at(i).toTensor().dim() == 1 && s.at(i).toTensor().numel() > 0,
                                    "MME MLA requires actual cache writer completions");
    return {{at::kBFloat16, q.sizes().vec()}};
}
class Mla final : public habana::OpBackend {
public:
    Mla(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_mla_mme"), dtype, {0}, {}, {}, false) { SetOutputMetaFn(meta); }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = meta(s);
        const auto heads = s.at(0).toTensor().size(1), width = s.at(3).toTensor().size(1);
        Params params{int32_t(s.at(9).toInt()), int32_t(s.at(10).toInt())};
        auto kv = BuildNode(this, graph, {"custom_deepseek_v41_mla_gather_gaudi2",
            {syn_in(1),syn_in(2),syn_in(3),syn_in(6),syn_in(7),syn_in(8)},
            {{{width,512},at::kBFloat16}, {{width,512},at::kFloat}, {{width},at::kFloat}}, &params,sizeof(params)});
        auto q = ReshapeHelper(graph, syn_in(0), {heads,512}, at::kBFloat16);
        synGEMMParams qk_params{false,true};
        auto logits = BuildNode(this, graph, {"gemm", {q.get(),kv.at(0).get()},
            {{{heads,width},at::kFloat}}, &qk_params,sizeof(qk_params)});
        auto probabilities = BuildNode(this, graph, {"custom_deepseek_v41_mla_softmax_gaudi2",
            {logits.at(0).get(),kv.at(2).get(),syn_in(4),syn_in(5)}, {{{heads,width},at::kFloat}}});
        synGEMMParams pv_params{false,false};
        auto product = BuildNode(this, graph, {"gemm", {probabilities.at(0).get(),kv.at(1).get()},
            {{{heads,512},at::kFloat}}, &pv_params,sizeof(pv_params)});
        auto converted = BuildNode(this, graph, {"cast_f32_to_bf16", {product.at(0).get()}, {{{heads,512},at::kBFloat16}}});
        syn_out(0) = ReshapeHelper(graph, converted.at(0).get(), output.at(0).shape, at::kBFloat16, 0);
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kSchema, "gemm", [](const at::Stack& s) {
        const auto m = meta(s); return habana::PartialOutputMetaDataVector{{m.at(0).dtype,m.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kSchema, [](synDeviceId d, c10::ScalarType t) { return std::make_shared<Mla>(d,t); });
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& q, const at::Tensor& swa, const at::Tensor& kv,
    const at::Tensor& ids, const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths,
    const at::Tensor& swa_done, const at::Tensor& main_done, int64_t offset, int64_t rows) {
    at::Stack s{q,swa,kv,ids,sink,scale,lengths,swa_done,main_done,offset,rows};
    const auto m = meta(s);
    if (Meta) return at::empty(m.at(0).shape,q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    return descriptor.execute(s).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mla_mme_gaudi2(Tensor q, Tensor swa, Tensor main, Tensor indices, Tensor sink, "
          "Tensor scale, Tensor lengths, Tensor swa_completion, Tensor main_completion, int swa_offset, int main_rows) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) { m.impl("custom_deepseek_v41_mla_mme_gaudi2",run<false>); }
TORCH_LIBRARY_IMPL(custom_op,Meta,m) { m.impl("custom_deepseek_v41_mla_mme_gaudi2",run<true>); }
