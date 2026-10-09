// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_fp8_qkv_prologue_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_fp8_qkv_prologue_gaudi2";
constexpr auto kSplit = "custom_op::custom_deepseek_v41_fp8_qkv_prologue_split_gaudi2";
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size() == 11, "FP8 QKV prologue requires ten tensors and epsilon");
    const auto q = s.at(0).toTensor(), sx = s.at(1).toTensor();
    const auto w = s.at(2).toTensor(), channel = s.at(3).toTensor();
    const auto qnorm = s.at(4).toTensor(), kvnorm = s.at(5).toTensor();
    const auto query = s.at(6).toTensor(), query_channel = s.at(7).toTensor();
    const auto positions = s.at(8).toTensor(), phase = s.at(9).toTensor();
    const auto rows = q.dim() == 2 ? q.size(0) : 0;
    const auto width = query.dim() == 2 ? query.size(0) : 0;
    TORCH_CHECK(rows >= 1 && rows <= 6 && q.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        q.sizes() == at::IntArrayRef({rows, 5120}) && sx.scalar_type() == at::kFloat &&
        sx.sizes() == at::IntArrayRef({rows, 1}) && w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        w.sizes() == at::IntArrayRef({1792, 5120}) && channel.scalar_type() == at::kFloat &&
        channel.sizes() == at::IntArrayRef({1, 1792}),
        "FP8 QKV prologue requires prepared FP8 [C,5120]/[1792,5120] and F32 row/channel scales");
    TORCH_CHECK(qnorm.scalar_type() == at::kBFloat16 && qnorm.sizes() == at::IntArrayRef({1280}) &&
        kvnorm.scalar_type() == at::kBFloat16 && kvnorm.sizes() == at::IntArrayRef({512}) &&
        query.scalar_type() == at::ScalarType::Float8_e4m3fn && (width == 8192 || width == 16384) &&
        query.size(1) == 1280 && query_channel.scalar_type() == at::kFloat &&
        query_channel.sizes() == at::IntArrayRef({1, width}),
        "FP8 QKV prologue requires BF16 norm weights and TP-local prepared FP8 query projection");
    const auto epsilon = static_cast<float>(s.at(10).toDouble());
    TORCH_CHECK(positions.scalar_type() == at::kInt && positions.sizes() == at::IntArrayRef({rows}) &&
        phase.scalar_type() == at::kFloat && phase.dim() == 2 && phase.size(1) == 64 &&
        phase.size(0) > 0 && phase.size(0) <= 1048576 && epsilon > 0 && std::isnormal(epsilon),
        "FP8 QKV prologue requires absolute I32 positions, F32 RoPE table and positive normal epsilon");
    for (int i = 0; i < 10; ++i) {
        const auto t = s.at(i).toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == q.device(),
            "FP8 QKV prologue requires contiguous inference tensors on the same device");
    }
    return {{at::kBFloat16, {rows, 1280}}, {at::kBFloat16, {rows, width}}, {at::kBFloat16, {rows, 512}}};
}
class Prologue final : public habana::OpBackend {
    bool split_;
 public:
    Prologue(int device, c10::ScalarType dtype, bool split)
        : OpBackend(device, NO_TPC + std::string("dsv41_fp8_qkv_prologue"), dtype, {0, 1, 2}, {}, {}, false),
          split_(split) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto m = metadata(s);
        const auto rows = m[0].shape[0];
        synGEMMParams params{false, true};
        auto product = BuildNode(this, graph, {"gemm", {syn_in(0), syn_in(2)},
            {{{rows, 1792}, at::kFloat}}, &params, sizeof(params)});
        auto epsilon = static_cast<float>(s.at(10).toDouble());
        std::vector<synapse_helpers::tensor> prepared, kv;
        if (split_) {
            prepared = BuildNode(this, graph, {"custom_deepseek_v41_fp8_q_prologue_gaudi2",
                {product[0].get(), syn_in(3), syn_in(1), syn_in(4)},
                {{{rows, 1280}, at::ScalarType::Float8_e4m3fn}, {{rows, 1}, at::kFloat},
                 {m[0].shape, at::kBFloat16, 0}}, &epsilon, sizeof(epsilon)});
            kv = BuildNode(this, graph, {"custom_deepseek_v41_fp8_kv_prologue_gaudi2",
                {product[0].get(), syn_in(3), syn_in(1), syn_in(5), syn_in(8), syn_in(9)},
                {{m[2].shape, at::kBFloat16, 2}}, &epsilon, sizeof(epsilon)});
        } else {
        prepared = BuildNode(this, graph, {kGuid,
            {product[0].get(), syn_in(3), syn_in(1), syn_in(4), syn_in(5), syn_in(8), syn_in(9)},
            {{{rows, 1280}, at::ScalarType::Float8_e4m3fn}, {{rows, 1}, at::kFloat},
             {m[0].shape, at::kBFloat16, 0}, {m[2].shape, at::kBFloat16, 2}},
            &epsilon, sizeof(epsilon)});
        }
        auto query_product = BuildNode(this, graph, {"gemm", {prepared[0].get(), syn_in(6)},
            {{m[1].shape, at::kFloat}}, &params, sizeof(params)});
        auto query = BuildNode(this, graph, {"custom_deepseek_v41_q_scale_rope_gaudi2",
            {query_product[0].get(), syn_in(7), prepared[1].get(), syn_in(8), syn_in(9)},
            {{m[1].shape, at::kBFloat16, 1}}});
        syn_out(0) = std::move(prepared[2]);
        syn_out(1) = std::move(query[0]);
        syn_out(2) = split_ ? std::move(kv[0]) : std::move(prepared[3]);
    }
};
const bool registered = [] {
    for (const bool split : {false, true}) {
    const auto schema = split ? kSplit : kSchema;
    habana::custom_op::registerUserCustomOp(schema, kGuid, [](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for (const auto& m : metadata(s)) out.push_back({m.dtype, m.shape});
        return out;
    }, nullptr);
    habana::KernelRegistry().add(schema, [split](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Prologue>(device, dtype, split);
    });
    }
    return true;
}();
template<bool Meta, bool Split = false>
Result run(const at::Tensor& q, const at::Tensor& sx, const at::Tensor& w, const at::Tensor& channel,
           const at::Tensor& qnorm, const at::Tensor& kvnorm, const at::Tensor& query,
           const at::Tensor& query_channel, const at::Tensor& positions, const at::Tensor& phase,
           double epsilon) {
    const at::Stack s{q, sx, w, channel, qnorm, kvnorm, query, query_channel, positions, phase, epsilon};
    const auto m = metadata(s);
    if (Meta) return {at::empty(m[0].shape, q.options().dtype(at::kBFloat16)),
                      at::empty(m[1].shape, q.options().dtype(at::kBFloat16)),
                      at::empty(m[2].shape, q.options().dtype(at::kBFloat16))};
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Split ? kSplit : kSchema);
    const auto out = descriptor.execute(s);
    return {out[0], out[1], out[2]};
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_fp8_qkv_prologue_gaudi2(Tensor q, Tensor sx, Tensor weight, "
          "Tensor channel, Tensor qnorm, Tensor kvnorm, Tensor query_weight, Tensor query_channel, "
          "Tensor positions, Tensor phase, float epsilon) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_fp8_qkv_prologue_split_gaudi2(Tensor q, Tensor sx, Tensor weight, "
          "Tensor channel, Tensor qnorm, Tensor kvnorm, Tensor query_weight, Tensor query_channel, "
          "Tensor positions, Tensor phase, float epsilon) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_fp8_qkv_prologue_gaudi2", run<false>);
    m.impl("custom_deepseek_v41_fp8_qkv_prologue_split_gaudi2", run<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_fp8_qkv_prologue_gaudi2", run<true>);
    m.impl("custom_deepseek_v41_fp8_qkv_prologue_split_gaudi2", run<true, true>);
}
