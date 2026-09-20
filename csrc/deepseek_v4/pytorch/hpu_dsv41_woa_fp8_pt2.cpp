// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_woa_fp8_gaudi2";
constexpr auto kRoundtripSchema = "custom_op::custom_deepseek_v41_woa_fp8_roundtrip_gaudi2";
constexpr auto kRopeSchema = "custom_op::custom_deepseek_v41_rope_woa_fp8_gaudi2";
constexpr auto kRopeRoundtripSchema = "custom_op::custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2";
constexpr auto kQuantSchema = "custom_op::custom_deepseek_v41_woa_quant_gaudi2";
constexpr auto kQuant = "custom_deepseek_v41_woa_quant_gaudi2";
constexpr auto kRopeQuant = "custom_deepseek_v41_woa_rope_quant_gaudi2";
constexpr auto kScale = "custom_deepseek_v41_woa_scale_gaudi2";
constexpr auto kScaleRoundtrip = "custom_deepseek_v41_woa_scale_roundtrip_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;
habana::OutputMetaDataVector meta(const at::Stack& stack, bool quant) {
    const auto x = stack.at(0).toTensor();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 3 && x.size(0) >= 1 && x.size(0) <= 8192 &&
                x.size(1) == 4 && x.size(2) == 4096, "wo_a requires BF16 [T,4,4096]");
    for (const auto& item : stack) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "wo_a requires matching contiguous inference tensors");
    }
    if (quant) return {{at::ScalarType::Float8_e4m3fn, {4, x.size(0), 4096}}, {at::kFloat, {4, x.size(0), 1}}};
    const auto w = stack.at(1).toTensor(), scale = stack.at(2).toTensor();
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.sizes() == at::IntArrayRef({4,4096,1024}) &&
                scale.scalar_type() == at::kFloat && scale.sizes() == at::IntArrayRef({4,1,1024}),
                "wo_a requires prepared Gaudi2 FP8 [4,4096,1024] and F32 [4,1,1024] channel scales");
    return {{at::kBFloat16, {x.size(0), 4096}}};
}
habana::OutputMetaDataVector rope_meta(const at::Stack& stack) {
    const auto x = stack.at(0).toTensor(), w = stack.at(1).toTensor(), scale = stack.at(2).toTensor();
    const auto positions = stack.at(3).toTensor(), phase = stack.at(4).toTensor();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 3 && x.size(0) >= 1 && x.size(0) <= 8192 &&
                x.size(1) == 32 && x.size(2) == 512, "rope wo_a requires BF16 [T,32,512]");
    TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.sizes() == at::IntArrayRef({4,4096,1024}) &&
                scale.scalar_type() == at::kFloat && scale.sizes() == at::IntArrayRef({4,1,1024}),
                "rope wo_a requires prepared Gaudi2 FP8 weights and channel scales");
    TORCH_CHECK(positions.scalar_type() == at::kInt && positions.dim() == 1 && positions.size(0) == x.size(0) &&
                phase.scalar_type() == at::kFloat && phase.dim() == 2 && phase.size(1) == 64,
                "rope wo_a requires I32 [T] positions and F32 [L,64] phase");
    for (const auto& item : stack) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "rope wo_a requires matching contiguous inference tensors");
    }
    return {{at::kBFloat16, {x.size(0), 4096}}};
}
class Woa final : public habana::OpBackend {
    bool quant_;
    bool rope_;
    bool roundtrip_;
public:
    Woa(int device, c10::ScalarType dtype, bool quant, bool rope = false, bool roundtrip = false)
        : OpBackend(device, NO_TPC + std::string(rope ? "dsv41_rope_woa_fp8" : "dsv41_woa_fp8"), dtype,
                    quant ? std::vector<int>{0,1} : std::vector<int>{0}, {}, {}, false),
          quant_(quant), rope_(rope), roundtrip_(roundtrip) {
        SetOutputMetaFn([quant, rope](const at::Stack& stack) {
            return rope ? rope_meta(stack) : meta(stack, quant);
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = rope_ ? rope_meta(stack) : meta(stack, quant_);
        const auto tokens = stack.at(0).toTensor().size(0);
        if (quant_) {
            auto result = BuildNode(this, graph, {kQuant, {syn_in(0)},
                {{{4,tokens,4096}, at::ScalarType::Float8_e4m3fn, 0}, {{4,tokens,1}, at::kFloat, 1}}});
            syn_out(0) = std::move(result.at(0)); syn_out(1) = std::move(result.at(1));
            return;
        }
        auto q = rope_
            ? BuildNode(this, graph, {kRopeQuant, {syn_in(0), syn_in(3), syn_in(4)},
                {{{4,tokens,4096}, at::ScalarType::Float8_e4m3fn}, {{4,tokens,1}, at::kFloat}}})
            : BuildNode(this, graph, {kQuant, {syn_in(0)},
                {{{4,tokens,4096}, at::ScalarType::Float8_e4m3fn}, {{4,tokens,1}, at::kFloat}}});
        // Prepared FP8 is already the final MME weight representation. Let
        // the compiler supply this persistent operand directly; an explicit
        // TPC byte-copy adds a second pass without changing its representation.
        synGEMMParams params{false, false};
        auto product = BuildNode(this, graph, {"batch_gemm", {q.at(0).get(), syn_in(1)},
            {{{4,tokens,1024}, at::kFloat}}, &params, sizeof(params)});
        auto scaled = BuildNode(this, graph, {roundtrip_ ? kScaleRoundtrip : kScale,
            {product.at(0).get(), syn_in(2), q.at(1).get()},
            {{{tokens,4,1024}, at::kBFloat16}}});
        syn_out(0) = ReshapeHelper(graph, scaled.at(0).get(), output.at(0).shape, at::kBFloat16, 0);
    }
};
const bool registered = [] {
    for (bool quant : {false, true}) {
        const auto schema = quant ? kQuantSchema : kSchema;
        habana::custom_op::registerUserCustomOp(schema, kQuant, [quant](const at::Stack& stack) {
            const auto output = meta(stack, quant);
            habana::PartialOutputMetaDataVector result;
            for (const auto& x : output) result.push_back({x.dtype, x.shape});
            return result;
        }, nullptr);
        habana::KernelRegistry().add(schema, [quant](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<Woa>(device, dtype, quant);
        });
    }
    habana::custom_op::registerUserCustomOp(kRoundtripSchema, kQuant, [](const at::Stack& stack) {
        const auto output = meta(stack, false);
        return habana::PartialOutputMetaDataVector{{output.at(0).dtype, output.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kRoundtripSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Woa>(device, dtype, false, false, true);
    });
    habana::custom_op::registerUserCustomOp(kRopeSchema, kRopeQuant, [](const at::Stack& stack) {
        const auto output = rope_meta(stack);
        return habana::PartialOutputMetaDataVector{{output.at(0).dtype, output.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kRopeSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Woa>(device, dtype, false, true);
    });
    habana::custom_op::registerUserCustomOp(kRopeRoundtripSchema, kRopeQuant, [](const at::Stack& stack) {
        const auto output = rope_meta(stack);
        return habana::PartialOutputMetaDataVector{{output.at(0).dtype, output.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kRopeRoundtripSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Woa>(device, dtype, false, true, true);
    });
    return true;
}();
template<bool Meta, bool Roundtrip = false>
at::Tensor run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s) {
    const auto output = meta({x,w,s}, false);
    if (Meta) return at::empty(output.at(0).shape, x.options());
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        Roundtrip ? kRoundtripSchema : kSchema);
    return descriptor.execute({x,w,s}).at(0);
}
template<bool Meta> Pair quant(const at::Tensor& x) {
    const auto output = meta({x}, true);
    if (Meta) return {at::empty(output.at(0).shape, x.options().dtype(at::ScalarType::Float8_e4m3fn)),
                      at::empty(output.at(1).shape, x.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kQuantSchema);
    auto result = descriptor.execute({x});
    return {result.at(0), result.at(1)};
}
template<bool Meta, bool Roundtrip = false>
at::Tensor rope_run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s,
                    const at::Tensor& positions, const at::Tensor& phase) {
    const auto output = rope_meta({x,w,s,positions,phase});
    if (Meta) return at::empty(output.at(0).shape, x.options());
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        Roundtrip ? kRopeRoundtripSchema : kRopeSchema);
    return descriptor.execute({x,w,s,positions,phase}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_woa_fp8_gaudi2(Tensor input, Tensor weight, Tensor channel_scale) -> Tensor");
    m.def("custom_deepseek_v41_woa_fp8_roundtrip_gaudi2(Tensor input, Tensor weight, Tensor channel_scale) -> Tensor");
    m.def("custom_deepseek_v41_rope_woa_fp8_gaudi2(Tensor input, Tensor weight, Tensor channel_scale, Tensor positions, Tensor phase) -> Tensor");
    m.def("custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2(Tensor input, Tensor weight, Tensor channel_scale, Tensor positions, Tensor phase) -> Tensor");
    m.def("custom_deepseek_v41_woa_quant_gaudi2(Tensor input) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_woa_fp8_gaudi2", run<false, false>);
    m.impl("custom_deepseek_v41_woa_fp8_roundtrip_gaudi2", run<false, true>);
    m.impl("custom_deepseek_v41_rope_woa_fp8_gaudi2", rope_run<false, false>);
    m.impl("custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2", rope_run<false, true>);
    m.impl("custom_deepseek_v41_woa_quant_gaudi2", quant<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_woa_fp8_gaudi2", run<true, false>);
    m.impl("custom_deepseek_v41_woa_fp8_roundtrip_gaudi2", run<true, true>);
    m.impl("custom_deepseek_v41_rope_woa_fp8_gaudi2", rope_run<true, false>);
    m.impl("custom_deepseek_v41_rope_woa_fp8_roundtrip_gaudi2", rope_run<true, true>);
    m.impl("custom_deepseek_v41_woa_quant_gaudi2", quant<true>);
}
