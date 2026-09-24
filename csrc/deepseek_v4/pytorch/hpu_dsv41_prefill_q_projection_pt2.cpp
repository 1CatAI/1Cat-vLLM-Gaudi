// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"

namespace {
constexpr auto projection = "custom_op::custom_deepseek_v41_prefill_q_projection_rope_gaudi2";
constexpr auto epilogue = "custom_op::custom_deepseek_v41_prefill_q_scale_rope_gaudi2";
constexpr auto bf16_projection = "custom_op::custom_deepseek_v41_prefill_q_projection_rope_bf16_gaudi2";
constexpr auto bf16_epilogue = "custom_op::custom_deepseek_v41_prefill_q_scale_rope_bf16_gaudi2";
constexpr auto bf16_guid = "custom_deepseek_v41_prefill_q_scale_rope_bf16_gaudi2";
constexpr auto guid = "custom_deepseek_v41_prefill_q_scale_rope_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& stack, bool scale_only, bool bf16_product = false) {
    TORCH_CHECK(stack.size() == 5, "Prefill Q projection requires five tensors");
    const auto x = stack.at(0).toTensor();
    for (const auto& item : stack) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "Prefill Q projection requires contiguous matching inference tensors");
    }
    TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8192,
                "Prefill Q projection requires T1..8192");
    const auto tokens = x.size(0);
    const auto w = stack.at(1).toTensor(), s = stack.at(2).toTensor();
    if (scale_only) {
        TORCH_CHECK(x.scalar_type() == (bf16_product ? at::kBFloat16 : at::kFloat) && x.size(1) == 16384 &&
                    w.scalar_type() == at::kFloat && w.sizes() == at::IntArrayRef({1, 16384}) &&
                    s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({tokens, 1}),
                    "Prefill Q epilogue product dtype must match the selected operator; scales are F32");
    } else {
        TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.size(1) == 1280 &&
                    w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                    w.sizes() == at::IntArrayRef({16384, 1280}) &&
                    s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({1, 16384}),
                    "Prefill Q projection requires BF16 [T,1280], FP8 [16384,1280], F32 [1,16384]");
    }
    const auto pos = stack.at(3).toTensor(), table = stack.at(4).toTensor();
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.sizes() == at::IntArrayRef({tokens}) &&
                table.scalar_type() == at::kFloat && table.dim() == 2 && table.size(0) > 0 &&
                table.size(0) <= 1048576 && table.size(1) == 64,
                "Prefill Q RoPE requires I32 [T] and F32 concatenated cos/sin [L,64]");
    return {{at::kBFloat16, {tokens, 16384}}};
}

class Projection final : public habana::OpBackend {
    bool scale_only_;
    bool bf16_product_;
public:
    Projection(int device, c10::ScalarType dtype, bool scale_only, bool bf16_product)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_q_projection"), dtype, {0}, {}, {}, false),
          scale_only_(scale_only), bf16_product_(bf16_product) {
        SetOutputMetaFn([scale_only, bf16_product](const at::Stack& s) {
            return metadata(s, scale_only, bf16_product);
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = metadata(stack, scale_only_, bf16_product_);
        const auto epilogue_guid = bf16_product_ ? bf16_guid : guid;
        const auto tokens = output[0].shape[0];
        if (scale_only_) {
            auto result = BuildNode(this, graph, {epilogue_guid, {syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4)},
                {{output[0].shape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(result[0]);
            return;
        }
        auto quantized = BuildNode(this, graph, {"custom_deepseek_v41_dense_quant_gaudi2", {syn_in(0)},
            {{{tokens, 1280}, at::ScalarType::Float8_e4m3fn}, {{tokens, 1}, at::kFloat}}});
        synGEMMParams params{false, true};
        auto product = BuildNode(this, graph, {"gemm", {quantized[0].get(), syn_in(1)},
            {{output[0].shape, bf16_product_ ? at::kBFloat16 : at::kFloat}}, &params, sizeof(params)});
        auto result = BuildNode(this, graph, {epilogue_guid,
            {product[0].get(), syn_in(2), quantized[1].get(), syn_in(3), syn_in(4)},
            {{output[0].shape, at::kBFloat16, 0}}});
        syn_out(0) = std::move(result[0]);
    }
};

const char* schema(bool scale_only, bool bf16_product) {
    return bf16_product ? (scale_only ? bf16_epilogue : bf16_projection) : (scale_only ? epilogue : projection);
}
const bool registered = [] {
    for (bool bf16_product : {false, true}) {
        for (bool scale_only : {false, true}) {
            const auto name = schema(scale_only, bf16_product);
            habana::custom_op::registerUserCustomOp(name, bf16_product ? bf16_guid : guid,
                [scale_only, bf16_product](const at::Stack& s) {
                    const auto output = metadata(s, scale_only, bf16_product)[0];
                    return habana::PartialOutputMetaDataVector{{output.dtype, output.shape}};
                }, nullptr);
            habana::KernelRegistry().add(name, [scale_only, bf16_product](synDeviceId device, c10::ScalarType dtype) {
                return std::make_shared<Projection>(device, dtype, scale_only, bf16_product);
            });
        }
    }
    return true;
}();

template<bool Fake, bool ScaleOnly, bool Bf16Product = false>
at::Tensor run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s,
               const at::Tensor& positions, const at::Tensor& table) {
    const auto output = metadata({x, w, s, positions, table}, ScaleOnly, Bf16Product);
    if constexpr (Fake) return at::empty(output[0].shape, x.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema(ScaleOnly, Bf16Product));
    return op.execute({x, w, s, positions, table})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_q_projection_rope_bf16_gaudi2(Tensor input, Tensor weight, Tensor channel_scale, Tensor positions, Tensor table) -> Tensor");
    m.def("custom_deepseek_v41_prefill_q_scale_rope_bf16_gaudi2(Tensor product, Tensor channel_scale, Tensor activation_scale, Tensor positions, Tensor table) -> Tensor");
    m.def("custom_deepseek_v41_prefill_q_projection_rope_gaudi2(Tensor input, Tensor weight, Tensor channel_scale, Tensor positions, Tensor table) -> Tensor");
    m.def("custom_deepseek_v41_prefill_q_scale_rope_gaudi2(Tensor product, Tensor channel_scale, Tensor activation_scale, Tensor positions, Tensor table) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_q_projection_rope_bf16_gaudi2", run<false, false, true>);
    m.impl("custom_deepseek_v41_prefill_q_scale_rope_bf16_gaudi2", run<false, true, true>);
    m.impl("custom_deepseek_v41_prefill_q_projection_rope_gaudi2", run<false, false>);
    m.impl("custom_deepseek_v41_prefill_q_scale_rope_gaudi2", run<false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_q_projection_rope_bf16_gaudi2", run<true, false, true>);
    m.impl("custom_deepseek_v41_prefill_q_scale_rope_bf16_gaudi2", run<true, true, true>);
    m.impl("custom_deepseek_v41_prefill_q_projection_rope_gaudi2", run<true, false>);
    m.impl("custom_deepseek_v41_prefill_q_scale_rope_gaudi2", run<true, true>);
}
