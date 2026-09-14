// SPDX-License-Identifier: Apache-2.0
// Adapted from main 92c82b97 N256 compound. Only N256 registrations live here;
// the retained DSpark BF16 registrations keep their original implementation.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <limits>
#include <cstdlib>
#include <cstring>
#include "perf_lib_layer_params.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kDecode = "custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2";
constexpr auto kDecodeNormal = "custom_deepseek_v41_mxfp4_prepared_dequant_normal_bf16_gaudi2";
constexpr auto kDecodeSchema = "custom_op::custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2";
constexpr auto kMoeSchema = "custom_op::custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2";
constexpr auto kDecodeK128 = "custom_deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2";
constexpr auto kDecodeK128Normal = "custom_deepseek_v41_mxfp4_prepared_dequant_k128n_bf16_gaudi2";
constexpr auto kDecodeK128Schema = "custom_op::custom_deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2";
constexpr auto kMoeK128Schema = "custom_op::custom_deepseek_v41_mxfp4_prepared_moe_k128_bf16_gaudi2";
constexpr auto kMoeFp8Schema = "custom_op::custom_deepseek_v41_mxfp4_prepared_moe_fp8_gaudi2";
constexpr auto kDecodeFp8 = "custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2";
constexpr auto kDynamicQuant = "custom_deepseek_v41_dynamic_quant_bf16_gaudi2";
constexpr auto kN256Fp8 = "custom_deepseek_v41_expert_n256_fp8_gaudi2";
constexpr auto kN256Bf16 = "custom_deepseek_v41_expert_n256_bf16_gaudi2";
constexpr auto kN256Scale = "custom_deepseek_v41_expert_n256_scale_gaudi2";
constexpr auto kN256FusedSchema = "custom_op::custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2";
constexpr auto kN256SiluQuant = "custom_deepseek_v41_expert_n256_silu_quant_gaudi2";
constexpr auto kN256Fp8Schema = "custom_op::custom_deepseek_v41_expert_n256_moe_fp8_gaudi2";
constexpr auto kN256Bf16Schema = "custom_op::custom_deepseek_v41_expert_n256_moe_bf16_gaudi2";
constexpr auto kN256DecodeFp8Schema = "custom_op::custom_deepseek_v41_expert_n256_fp8_gaudi2";
constexpr auto kN256DecodeBf16Schema = "custom_op::custom_deepseek_v41_expert_n256_bf16_gaudi2";
constexpr auto kRound = "custom_deepseek_v41_bf16_identity_gaudi2";

void contract(const at::Tensor& tensor, at::ScalarType type, const at::Device& device) {
    TORCH_CHECK(tensor.scalar_type() == type && tensor.device() == device && tensor.is_contiguous() &&
                !tensor.requires_grad(), "V4.1 prepared MXFP4 requires contiguous inference inputs with matching dtype/device");
}

std::vector<int64_t> decode_shape(const at::Stack& stack, bool n256 = false) {
    const auto& ids = stack.at(0).toTensor();
    const auto& q = stack.at(1).toTensor();
    const auto& s = stack.at(2).toTensor();
    const auto& lut = stack.at(3).toTensor();
    contract(ids, at::kInt, q.device());
    contract(q, at::kShort, q.device());
    contract(s, n256 ? at::kShort : at::kBFloat16, q.device());
    contract(lut, at::kBFloat16, q.device());
    TORCH_CHECK(ids.dim() == 2 && ids.numel() > 0 && ids.numel() <= 3072,
                "V4.1 decode expects [tokens, ordered experts] IDs");
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(0) <= 384 && q.size(1) > 0 && q.size(1) <= (n256 ? 20 : 40) &&
                q.size(2) > 0 && q.size(2) <= (n256 ? 327680 : 163840) && q.size(2) % (n256 ? 8192 : 4096) == 0,
                "V4.1 Q16 expects [E,N/128,(K/2)*64], with K a multiple of 128");
    TORCH_CHECK(s.dim() == 3 && s.size(0) == q.size(0) && s.size(1) == q.size(1) && s.size(2) * 8 == q.size(2),
                "V4.1 S16 must preserve group-32 scale encodings");
    TORCH_CHECK(lut.sizes() == at::IntArrayRef({128}), "V4.1 requires the exact V4 MXFP4 lookup bytes");
    return {ids.numel(), q.size(2) / (n256 ? 64 : 32), q.size(1) * (n256 ? 256 : 128)};
}

std::vector<int64_t> moe_shape(const at::Stack& stack, bool n256 = false) {
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    const auto& router = stack.at(2).toTensor();
    contract(x, at::kBFloat16, ids.device());
    contract(router, at::kFloat, ids.device());
    auto first = decode_shape({ids, stack.at(3), stack.at(5), stack.at(7)}, n256);
    auto second = decode_shape({ids, stack.at(4), stack.at(6), stack.at(7)}, n256);
    TORCH_CHECK(x.dim() == 2 && x.size(0) == ids.size(0) && x.size(1) == first[1] &&
                (ids.size(1) == 3 || ids.size(1) == 6) && router.sizes() == ids.sizes() &&
                first[2] == 2 * second[1] && second[2] == x.size(1) &&
                stack.at(3).toTensor().size(0) == stack.at(4).toTensor().size(0),
                "V4.1 MoE shape must connect W13, W2 and ordered top-3/top-6 routing");
    return {x.size(0), x.size(1)};
}

void fp8_contract(const at::Stack& stack, bool n256 = false) {
    moe_shape(stack, n256);
    const auto x = stack.at(0).toTensor(), ids = stack.at(1).toTensor();
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 6 && ids.size(1) == 6 && stack.back().toBool(),
                "V4.1 N256 FP8 MoE requires C1-C6 top6 and finite normal-scale qualification");
    for (int index : {0, 1}) {
        const auto channel = stack.at(8 + index).toTensor(), q = stack.at(3 + index).toTensor();
        contract(channel, at::kBFloat16, x.device());
        TORCH_CHECK(channel.sizes() == at::IntArrayRef({q.size(0), q.size(1), n256 ? 256 : 128}),
                    "V4.1 FP8 channel scales must match the prepared Q16 rows");
    }
}

class PreparedV41 final : public habana::OpBackend {
    bool moe_;
    bool fp8_;
    bool fused_;
    bool k128_;
    bool n256_;

    const char* decode_guid(bool normal) const {
        if (n256_) return fp8_ ? kN256Fp8 : kN256Bf16;
        // Read only while building the recipe; the immutable launch profile
        // and native binary fingerprint own this experimental selection.
        const char* pipeline = std::getenv("VLLM_HPU_DSV41_EXPERT_COORD_PIPELINE");
        if (k128_ && normal && pipeline && std::strcmp(pipeline, "1") == 0)
            return "custom_deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2";
        return k128_ ? (normal ? kDecodeK128Normal : kDecodeK128) : (normal ? kDecodeNormal : kDecode);
    }
    using Tensor = synapse_helpers::tensor;

    Tensor cast(synapse_helpers::graph& graph, synTensor input, const std::vector<int64_t>& shape, bool toFloat) {
        auto outputs = BuildNode(this, graph, {toFloat ? "cast_bf16_to_f32" : "cast_f32_to_bf16", {input},
            {{shape, toFloat ? at::kFloat : at::kBFloat16}}});
        return std::move(outputs.at(0));
    }

    Tensor round(synapse_helpers::graph& graph, synTensor input, int tokens, int experts, int width) {
        auto flat = ReshapeHelper(graph, input, {1, tokens * experts * width}, at::kBFloat16);
        auto result = BuildNode(this, graph, {kRound, {flat.get()},
            {{{1, tokens * experts * width}, at::kBFloat16}}});
        return ReshapeHelper(graph, result.at(0).get(), {tokens, experts, width}, at::kBFloat16);
    }

    Tensor slice(synapse_helpers::graph& graph, synTensor input, int tokens, int experts, int width,
                 int begin, int end, int axis, at::ScalarType dtype) {
        synSliceParams params{};
        for (unsigned i = 0; i < sizeof(params.axes) / sizeof(params.axes[0]); ++i) {
            params.axes[i] = i;
            params.steps[i] = 1;
        }
        params.ends[0] = width;
        params.ends[1] = experts;
        params.ends[2] = tokens;
        params.starts[axis] = begin;
        params.ends[axis] = end;
        std::vector<int64_t> shape{tokens, experts, width};
        shape[2 - axis] = end - begin;
        auto output = BuildNode(this, graph, {"slice", {input}, {{shape, dtype}}, &params, sizeof(params)});
        return std::move(output.at(0));
    }

    Tensor linear(synapse_helpers::graph& graph, synTensor input, synTensor ids, synTensor q,
                  synTensor s, synTensor lookup, int tokens, int experts, int n, int k,
                  bool broadcast, bool normal, synTensor channel = nullptr) {
        if (fp8_) {
            const int slots = tokens * experts;
            auto decoded = n256_
                ? BuildNode(this, graph, {kN256Fp8, {ids, q, s, lookup},
                    {{{slots, k, n}, at::ScalarType::Float8_e4m3fn}}})
                : BuildNode(this, graph, {kDecodeFp8, {ids, q, s, channel, lookup},
                    {{{experts, k, n}, at::ScalarType::Float8_e4m3fn}, {{experts, 1, n}, at::kFloat}}});
            const int rows = broadcast ? tokens : slots;
            auto inputRows = ReshapeHelper(graph, input, {rows, k}, at::kBFloat16);
            auto quant = BuildNode(this, graph, {kDynamicQuant, {inputRows.get()},
                {{{rows, k}, at::ScalarType::Float8_e4m3fn}, {{rows, 1}, at::kFloat}}});
            auto activation = expand_routes(graph, quant.at(0).get(), tokens, experts, k,
                                            at::ScalarType::Float8_e4m3fn, broadcast);
            auto scaleRows = expand_routes(graph, quant.at(1).get(), tokens, experts, 1, at::kFloat, broadcast);
            auto activationScale = ReshapeHelper(graph, scaleRows.get(), {slots, 1}, at::kFloat);
            synGEMMParams params{false, false};
            const std::vector<int64_t> resultShape{slots, 1, n};
            auto product = BuildNode(this, graph, {"batch_gemm", {activation.get(), decoded.at(0).get()},
                {{resultShape, at::kFloat}}, &params, sizeof(params)});
            if (n256_) {
                auto scaled = BuildNode(this, graph, {kN256Scale,
                    {product.at(0).get(), ids, activationScale.get(), channel}, {{resultShape, at::kBFloat16}}});
                return ReshapeHelper(graph, scaled.at(0).get(), {tokens, experts, n}, at::kBFloat16);
            }
            auto weightScaled = BuildNode(this, graph, {"mult_fwd_f32", {product.at(0).get(), decoded.at(1).get()},
                {{resultShape, at::kFloat}}});
            auto scaled = BuildNode(this, graph, {"mult_fwd_f32", {weightScaled.at(0).get(), activationScale.get()},
                {{resultShape, at::kFloat}}});
            auto rounded = cast(graph, scaled.at(0).get(), resultShape, false);
            return ReshapeHelper(graph, rounded.get(), {1, experts, n}, at::kBFloat16);
        }
        auto decoded = BuildNode(this, graph, {decode_guid(normal), {ids, q, s, lookup},
            {{{tokens * experts, k, n}, at::kBFloat16}}});
        // Keep the TPC and MME batch axes identical. A [T,E,K,N] reshape
        // between them prevents the Gaudi2 slicer from composing the access
        // maps when T > 1 and spills the full decoded weights into HBM.
        // Broadcasting the small activation is bounded (T*E*K BF16 values).
        auto activations = [&]() {
            if (broadcast && tokens > 1) {
                auto tokenInputs = ReshapeHelper(graph, input, {tokens, 1, k}, at::kBFloat16);
                auto expanded = BuildNode(this, graph, {"broadcast", {tokenInputs.get()},
                    {{{tokens, experts, k}, at::kBFloat16}}});
                return ReshapeHelper(graph, expanded.at(0).get(), {tokens * experts, 1, k}, at::kBFloat16);
            }
            return ReshapeHelper(graph, input, {broadcast ? 1 : tokens * experts, 1, k}, at::kBFloat16);
        }();
        synGEMMParams params{false, false};
        auto result = BuildNode(this, graph, {"batch_gemm", {activations.get(), decoded.at(0).get()},
            {{{tokens * experts, 1, n}, at::kBFloat16}}, &params, sizeof(params)});
        return ReshapeHelper(graph, result.at(0).get(), {tokens, experts, n}, at::kBFloat16);
    }

    Tensor expand_routes(synapse_helpers::graph& graph, synTensor input, int tokens, int experts,
                         int width, at::ScalarType dtype, bool broadcast = true) {
        if (broadcast) {
            auto rows = ReshapeHelper(graph, input, {tokens, 1, width}, dtype);
            auto expanded = BuildNode(this, graph, {"broadcast", {rows.get()},
                {{{tokens, experts, width}, dtype}}});
            return ReshapeHelper(graph, expanded.at(0).get(), {tokens * experts, 1, width}, dtype);
        }
        return ReshapeHelper(graph, input, {tokens * experts, 1, width}, dtype);
    }

    Tensor fused_down(synapse_helpers::graph& graph, synTensor ids, int tokens, int experts,
                      int hidden, int intermediate) {
        const int slots = tokens * experts;
        auto quant = BuildNode(this, graph, {kDynamicQuant, {syn_in(0)},
            {{{tokens, hidden}, at::ScalarType::Float8_e4m3fn}, {{tokens, 1}, at::kFloat}}});
        auto x = expand_routes(graph, quant.at(0).get(), tokens, experts, hidden, at::ScalarType::Float8_e4m3fn);
        auto scaleRows = expand_routes(graph, quant.at(1).get(), tokens, experts, 1, at::kFloat);
        auto sx = ReshapeHelper(graph, scaleRows.get(), {slots, 1}, at::kFloat);
        auto router = ReshapeHelper(graph, syn_in(2), {1, slots}, at::kFloat);
        auto w13 = BuildNode(this, graph, {kN256Fp8, {ids, syn_in(3), syn_in(5), syn_in(7)},
            {{{slots, hidden, intermediate * 2}, at::ScalarType::Float8_e4m3fn}}});
        synGEMMParams params{false, false};
        auto p13 = BuildNode(this, graph, {"batch_gemm", {x.get(), w13.at(0).get()},
            {{{slots, 1, intermediate * 2}, at::kFloat}}, &params, sizeof(params)});
        auto middle = BuildNode(this, graph, {kN256SiluQuant,
            {p13.at(0).get(), ids, sx.get(), syn_in(8), router.get()},
            {{{slots, 1, intermediate}, at::ScalarType::Float8_e4m3fn}, {{slots, 1, 1}, at::kFloat}}});
        auto w2 = BuildNode(this, graph, {kN256Fp8, {ids, syn_in(4), syn_in(6), syn_in(7)},
            {{{slots, intermediate, hidden}, at::ScalarType::Float8_e4m3fn}}});
        auto p2 = BuildNode(this, graph, {"batch_gemm", {middle.at(0).get(), w2.at(0).get()},
            {{{slots, 1, hidden}, at::kFloat}}, &params, sizeof(params)});
        auto sx2 = ReshapeHelper(graph, middle.at(1).get(), {slots, 1}, at::kFloat);
        auto scaled = BuildNode(this, graph, {kN256Scale, {p2.at(0).get(), ids, sx2.get(), syn_in(9)},
            {{{slots, 1, hidden}, at::kBFloat16}}});
        return ReshapeHelper(graph, scaled.at(0).get(), {tokens, experts, hidden}, at::kBFloat16);
    }

 public:
    PreparedV41(int device, c10::ScalarType dtype, bool moe, bool fp8 = false, bool k128 = false, bool n256 = false, bool fused = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_prepared_mxfp4"), dtype, {0}, {}, {}, false), moe_(moe), fp8_(fp8), fused_(fused), k128_(k128), n256_(n256) {
        SetOutputMetaFn([moe, fp8, n256](const at::Stack& stack) {
            if (fp8 && moe) fp8_contract(stack, n256);
            const auto dtype = !moe && fp8 ? at::ScalarType::Float8_e4m3fn : at::kBFloat16;
            return habana::OutputMetaDataVector{{dtype, moe ? moe_shape(stack, n256) : decode_shape(stack, n256)}};
        });
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        if (fp8_ && moe_) fp8_contract(stack, n256_);
        const bool normal = stack.back().toBool();
        const auto resultShape = moe_ ? moe_shape(stack, n256_) : decode_shape(stack, n256_);
        const auto& idsTensor = stack.at(moe_ ? 1 : 0).toTensor();
        auto ids = ReshapeHelper(graph, syn_in(moe_ ? 1 : 0), {1, idsTensor.numel()}, at::kInt);
        if (!moe_) {
            auto result = BuildNode(this, graph, {decode_guid(normal),
                {ids.get(), syn_in(1), syn_in(2), syn_in(3)}, {{resultShape, fp8_ ? at::ScalarType::Float8_e4m3fn : at::kBFloat16, 0}}});
            syn_out(0) = std::move(result.at(0));
            return;
        }
        const int tokens = idsTensor.size(0), experts = idsTensor.size(1), hidden = resultShape[1];
        const int intermediate = stack.at(4).toTensor().size(2) / (n256_ ? 64 : 32);
        const std::vector<int64_t> shape{tokens, experts, intermediate};
        auto down = [&]() -> Tensor {
            if (fused_) return fused_down(graph, ids.get(), tokens, experts, hidden, intermediate);
        auto first = linear(graph, syn_in(0), ids.get(), syn_in(3), syn_in(5), syn_in(7),
                            tokens, experts, intermediate * 2, hidden, true, normal, fp8_ ? syn_in(8) : nullptr);
        auto gate = slice(graph, first.get(), tokens, experts, intermediate * 2, 0, intermediate, 0, at::kBFloat16);
        auto up = slice(graph, first.get(), tokens, experts, intermediate * 2, intermediate, intermediate * 2, 0, at::kBFloat16);
        auto gateF32 = cast(graph, gate.get(), shape, true);
        auto upF32 = cast(graph, up.get(), shape, true);
        // Reference inference/model.py Expert.forward: gate <= 10, up in
        // [-10,10]; SwiGLU and routing happen in FP32 before the W2 BF16 input.
        ns_ClampKernel::Params gateClip{}, upClip{};
        gateClip.lowerBound.f = -std::numeric_limits<float>::infinity();
        gateClip.upperBound.f = upClip.upperBound.f = 10.0f;
        upClip.lowerBound.f = -10.0f;
        auto clippedGate = BuildNode(this, graph, {"clamp_pt_fwd_f32", {gateF32.get()}, {{shape, at::kFloat}},
            &gateClip, sizeof(gateClip)});
        auto clippedUp = BuildNode(this, graph, {"clamp_pt_fwd_f32", {upF32.get()}, {{shape, at::kFloat}},
            &upClip, sizeof(upClip)});
        auto silu = BuildNode(this, graph, {"silu_fwd_f32", {clippedGate.at(0).get()}, {{shape, at::kFloat}}});
        auto product = BuildNode(this, graph, {"mult_fwd_f32", {silu.at(0).get(), clippedUp.at(0).get()}, {{shape, at::kFloat}}});
        auto router = ReshapeHelper(graph, syn_in(2), {tokens, experts, 1}, at::kFloat);
        auto weighted = BuildNode(this, graph, {"mult_fwd_f32", {product.at(0).get(), router.get()}, {{shape, at::kFloat}}});
        auto rounded = cast(graph, weighted.at(0).get(), shape, false);
        auto middle = round(graph, rounded.get(), tokens, experts, intermediate);
        return linear(graph, middle.get(), ids.get(), syn_in(4), syn_in(6), syn_in(7),
                           tokens, experts, hidden, intermediate, false, normal, fp8_ ? syn_in(9) : nullptr);
        }();
        auto downF32 = cast(graph, down.get(), {tokens, experts, hidden}, true);
        auto accumulated = slice(graph, downF32.get(), tokens, experts, hidden, 0, 1, 1, at::kFloat);
        for (int expert = 1; expert < experts; ++expert) {
            auto row = slice(graph, downF32.get(), tokens, experts, hidden, expert, expert + 1, 1, at::kFloat);
            auto sum = BuildNode(this, graph, {"add_fwd_f32", {accumulated.get(), row.get()},
                {{{tokens, 1, hidden}, at::kFloat}}});
            accumulated = std::move(sum.at(0));
        }
        auto result = cast(graph, accumulated.get(), {tokens, 1, hidden}, false);
        auto identity = round(graph, result.get(), tokens, 1, hidden);
        syn_out(0) = ReshapeHelper(graph, identity.get(), resultShape, at::kBFloat16, 0);
    }
};

const bool registered = [] {
    for (int mode = 5; mode < 10; ++mode) {
        const bool n256 = mode >= 5;
        const bool moe = mode == 1 || mode == 2 || mode == 4 || mode == 5 || mode == 6 || mode == 9;
        const bool fp8 = mode == 2 || mode == 5 || mode == 7 || mode == 9;
        const bool k128 = mode == 3 || mode == 4;
        const bool fused = mode == 9;
        const char* schema = fused ? kN256FusedSchema : n256 ? (moe ? (fp8 ? kN256Fp8Schema : kN256Bf16Schema)
                                                        : (fp8 ? kN256DecodeFp8Schema : kN256DecodeBf16Schema))
            : k128 ? (moe ? kMoeK128Schema : kDecodeK128Schema)
            : fp8 ? kMoeFp8Schema : moe ? kMoeSchema : kDecodeSchema;
        habana::custom_op::registerUserCustomOp(schema, kDecode, [moe, fp8, n256](const at::Stack& stack) {
            if (fp8 && moe) fp8_contract(stack, n256);
            const auto dtype = !moe && fp8 ? at::ScalarType::Float8_e4m3fn : at::kBFloat16;
            return habana::PartialOutputMetaDataVector{{dtype, moe ? moe_shape(stack, n256) : decode_shape(stack, n256)}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [moe, fp8, k128, n256, fused](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<PreparedV41>(device, dtype, moe, fp8, k128, n256, fused);
        });
    }
    return true;
}();

at::Tensor run(const at::Stack& stack, bool moe, bool meta, bool k128, bool n256 = false, bool fp8 = false) {
    const auto shape = moe ? moe_shape(stack, n256) : decode_shape(stack, n256);
    const auto dtype = !moe && fp8 ? at::ScalarType::Float8_e4m3fn : at::kBFloat16;
    if (moe && k128) {
        TORCH_CHECK(stack.at(0).toTensor().size(0) == 1 && stack.at(1).toTensor().size(1) == 6,
                    "V4.1 K128 MoE requires C1 and top6");
    }
    if (meta) return at::empty(shape, stack.at(0).toTensor().options().dtype(dtype));
    TORCH_CHECK(registered && stack.at(0).toTensor().device().type() == at::kHPU, "V4.1 MXFP4 requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        n256 ? (moe ? kN256Bf16Schema : (fp8 ? kN256DecodeFp8Schema : kN256DecodeBf16Schema)) :
        k128 ? (moe ? kMoeK128Schema : kDecodeK128Schema) : (moe ? kMoeSchema : kDecodeSchema));
    return descriptor.execute(stack).at(0);
}

template<bool Meta, bool N256 = false, bool Fused = false> at::Tensor moe_fp8(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router,
    const at::Tensor& q13, const at::Tensor& q2, const at::Tensor& s13, const at::Tensor& s2,
    const at::Tensor& lookup, const at::Tensor& channel13, const at::Tensor& channel2, bool normal) {
    const at::Stack stack{x, ids, router, q13, q2, s13, s2, lookup, channel13, channel2, normal};
    fp8_contract(stack, N256);
    if (Meta) return at::empty(moe_shape(stack, N256), x.options());
    TORCH_CHECK(registered && x.device().type() == at::kHPU, "V4.1 FP8 MoE requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Fused ? kN256FusedSchema : N256 ? kN256Fp8Schema : kMoeFp8Schema);
    return descriptor.execute(stack).at(0);
}

template<bool Meta, bool K128 = false, bool N256 = false, bool FP8 = false> at::Tensor dequant(const at::Tensor& ids, const at::Tensor& q, const at::Tensor& s,
                                      const at::Tensor& lookup, bool normal) {
    return run({ids, q, s, lookup, normal}, false, Meta, K128, N256, FP8);
}
template<bool Meta, bool K128 = false, bool N256 = false> at::Tensor moe(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router,
                                  const at::Tensor& q13, const at::Tensor& q2, const at::Tensor& s13,
                                  const at::Tensor& s2, const at::Tensor& lookup, bool normal) {
    return run({x, ids, router, q13, q2, s13, s2, lookup, normal}, true, Meta, K128, N256);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2, bool normal) -> Tensor");
    m.def("custom_deepseek_v41_expert_n256_moe_fp8_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2, bool normal) -> Tensor");
    m.def("custom_deepseek_v41_expert_n256_moe_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_expert_n256_fp8_gaudi2(Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal=True) -> Tensor");
    m.def("custom_deepseek_v41_expert_n256_bf16_gaudi2(Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2", moe_fp8<false, true, true>);
    m.impl("custom_deepseek_v41_expert_n256_moe_fp8_gaudi2", moe_fp8<false, true>);
    m.impl("custom_deepseek_v41_expert_n256_moe_bf16_gaudi2", moe<false, false, true>);
    m.impl("custom_deepseek_v41_expert_n256_fp8_gaudi2", dequant<false, false, true, true>);
    m.impl("custom_deepseek_v41_expert_n256_bf16_gaudi2", dequant<false, false, true, false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_expert_n256_moe_fused_fp8_gaudi2", moe_fp8<true, true, true>);
    m.impl("custom_deepseek_v41_expert_n256_moe_fp8_gaudi2", moe_fp8<true, true>);
    m.impl("custom_deepseek_v41_expert_n256_moe_bf16_gaudi2", moe<true, false, true>);
    m.impl("custom_deepseek_v41_expert_n256_fp8_gaudi2", dequant<true, false, true, true>);
    m.impl("custom_deepseek_v41_expert_n256_bf16_gaudi2", dequant<true, false, true, false>);
}
