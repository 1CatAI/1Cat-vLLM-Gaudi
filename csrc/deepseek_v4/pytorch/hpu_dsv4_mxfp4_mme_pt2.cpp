// SPDX-License-Identifier: Apache-2.0
// Reuses 1cat-vllm QPN M1's runtime expert selection and input broadcast.
// Matrix work remains on Gaudi's MME. All decoded weights belong to the
// compiled recipe; actual SRAM placement must be qualified separately.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "perf_lib_layer_params.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kDequant = "custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2";
constexpr auto kDequantNormal = "custom_deepseek_v4_mxfp4_indexed_dequant_normal_bf16_gaudi2";
constexpr auto kDequantSchema = "custom_op::custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2";
constexpr auto kLinearSchema = "custom_op::custom_deepseek_v4_mxfp4_indexed_linear_bf16_gaudi2";
constexpr auto kMoeSchema = "custom_op::custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2";
constexpr auto kPreparedDequant = "custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2";
constexpr auto kPreparedDequantNormal = "custom_deepseek_v4_mxfp4_prepared_dequant_normal_bf16_gaudi2";
constexpr auto kPreparedDequantGate = "custom_deepseek_v4_mxfp4_prepared_gate_bf16_gaudi2";
constexpr auto kPreparedDequantGateNormal =
    "custom_deepseek_v4_mxfp4_prepared_gate_normal_bf16_gaudi2";
constexpr auto kPreparedDequantUp = "custom_deepseek_v4_mxfp4_prepared_up_bf16_gaudi2";
constexpr auto kPreparedDequantUpNormal =
    "custom_deepseek_v4_mxfp4_prepared_up_normal_bf16_gaudi2";
constexpr auto kPreparedDequantSchema = "custom_op::custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2";
constexpr auto kPreparedLinearSchema = "custom_op::custom_deepseek_v4_mxfp4_prepared_linear_bf16_gaudi2";
constexpr auto kPreparedExpertsSchema = "custom_op::custom_deepseek_v4_mxfp4_prepared_experts_bf16_gaudi2";
constexpr auto kPreparedMoeSchema = "custom_op::custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2";
constexpr auto kWeightedSumSchema = "custom_op::custom_deepseek_v4_weighted_sum_bf16_gaudi2";
constexpr auto kBF16Identity = "custom_deepseek_v4_bf16_identity_gaudi2";
enum class Kind { Dequant, Linear, Moe };

void tensor_contract(const at::Tensor& tensor, at::ScalarType dtype, const at::Device& device) {
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.device() == device,
                "indexed MXFP4 dtype/device mismatch");
    TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad(),
                "indexed MXFP4 requires contiguous inference inputs");
}

std::vector<int64_t> dequant_shape(const at::Tensor& ids, const at::Tensor& packed,
                                 const at::Tensor& scales) {
    tensor_contract(ids, at::kInt, packed.device());
    tensor_contract(packed, at::kByte, packed.device());
    tensor_contract(scales, at::kByte, packed.device());
    TORCH_CHECK(ids.sizes() == at::IntArrayRef({1, 6}), "indexed MXFP4 requires six ordered expert IDs");
    TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3 && packed.size(0) > 0 &&
                packed.size(0) <= 256 && packed.size(1) > 0 && packed.size(1) <= 4096 &&
                packed.size(2) > 0 && packed.size(2) <= 2048 && packed.size(2) % 256 == 0,
                "indexed MXFP4 requires [E,N,K/2] packed weights with K divisible by 512");
    TORCH_CHECK(scales.size(0) == packed.size(0) && scales.size(1) == packed.size(1) &&
                scales.size(2) * 16 == packed.size(2), "indexed MXFP4 requires E8M0 groups of 32");
    return {6, packed.size(1), packed.size(2) * 2};
}

std::vector<int64_t> shape(const at::Stack& stack, Kind kind) {
    if (kind == Kind::Dequant)
        return dequant_shape(stack.at(0).toTensor(), stack.at(1).toTensor(), stack.at(2).toTensor());
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    tensor_contract(x, at::kBFloat16, ids.device());
    if (kind == Kind::Linear) {
        auto w = dequant_shape(ids, stack.at(2).toTensor(), stack.at(3).toTensor());
        TORCH_CHECK(x.dim() == 2 && (x.size(0) == 1 || x.size(0) == 6) && x.size(1) == w[2],
                    "indexed MXFP4 linear expects [1,K] or [6,K] activations");
        return {6, w[1]};
    }
    const auto& router = stack.at(2).toTensor();
    const auto& w13 = stack.at(3).toTensor();
    const auto& w2 = stack.at(4).toTensor();
    const auto& s13 = stack.at(5).toTensor();
    const auto& s2 = stack.at(6).toTensor();
    tensor_contract(router, at::kBFloat16, x.device());
    dequant_shape(ids, w13, s13);
    dequant_shape(ids, w2, s2);
    TORCH_CHECK(x.sizes() == at::IntArrayRef({1, 4096}) &&
                router.sizes() == at::IntArrayRef({1, 6}) &&
                w13.sizes() == at::IntArrayRef({256, 2048, 2048}) &&
                w2.sizes() == at::IntArrayRef({256, 4096, 512}),
                "indexed MXFP4 MoE is limited to the DeepSeek V4 TP2 single-token shape");
    return {1, 4096};
}

std::vector<int64_t> prepared_dequant_shape(const at::Tensor& ids, const at::Tensor& q16,
                                            const at::Tensor& s16, const at::Tensor& lookup) {
    tensor_contract(ids, at::kInt, q16.device());
    tensor_contract(q16, at::kShort, q16.device());
    tensor_contract(s16, at::kBFloat16, q16.device());
    tensor_contract(lookup, at::kBFloat16, q16.device());
    TORCH_CHECK(ids.sizes() == at::IntArrayRef({1, 6}), "prepared MXFP4 requires six ordered expert IDs");
    TORCH_CHECK(q16.dim() == 3 && s16.dim() == 3 && q16.size(0) > 0 && q16.size(0) <= 256 &&
                q16.size(1) > 0 && q16.size(1) <= 32 && q16.size(2) > 0 &&
                q16.size(2) <= 131072 && q16.size(2) % 16384 == 0,
                "prepared MXFP4 requires block-major [E,N/128,(K/2)*64] Q16");
    TORCH_CHECK(s16.size(0) == q16.size(0) && s16.size(1) == q16.size(1) &&
                s16.size(2) * 8 == q16.size(2),
                "prepared MXFP4 requires block-major [E,N/128,(K/32)*128] S16");
    TORCH_CHECK(lookup.sizes() == at::IntArrayRef({128}), "prepared MXFP4 requires a 128-value lookup");
    return {6, q16.size(2) / 32, q16.size(1) * 128};
}

std::vector<int64_t> prepared_shape(const at::Stack& stack, Kind kind) {
    if (kind == Kind::Dequant)
        return prepared_dequant_shape(stack.at(0).toTensor(), stack.at(1).toTensor(),
                                      stack.at(2).toTensor(), stack.at(3).toTensor());
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    tensor_contract(x, at::kBFloat16, ids.device());
    if (kind == Kind::Linear) {
        auto weights = prepared_dequant_shape(ids, stack.at(2).toTensor(), stack.at(3).toTensor(),
                                              stack.at(4).toTensor());
        TORCH_CHECK(x.dim() == 2 && (x.size(0) == 1 || x.size(0) == 6) && x.size(1) == weights[1],
                    "prepared MXFP4 linear expects [1,K] or [6,K] activations");
        return {6, weights[2]};
    }
    const auto& q13 = stack.at(2).toTensor();
    const auto& q2 = stack.at(3).toTensor();
    const auto& s13 = stack.at(4).toTensor();
    const auto& s2 = stack.at(5).toTensor();
    const auto& lookup = stack.at(6).toTensor();
    prepared_dequant_shape(ids, q13, s13, lookup);
    prepared_dequant_shape(ids, q2, s2, lookup);
    TORCH_CHECK(x.sizes() == at::IntArrayRef({1, 4096}) &&
                q13.sizes() == at::IntArrayRef({256, 16, 131072}) &&
                q2.sizes() == at::IntArrayRef({256, 32, 32768}),
                "prepared MXFP4 MoE is limited to the DeepSeek V4 TP2 single-token shape");
    return {6, 4096};
}

std::vector<int64_t> prepared_moe_shape(const at::Stack& stack) {
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    const auto& router = stack.at(2).toTensor();
    const auto& q13 = stack.at(3).toTensor();
    const auto& q2 = stack.at(4).toTensor();
    const auto& s13 = stack.at(5).toTensor();
    const auto& s2 = stack.at(6).toTensor();
    const auto& lookup = stack.at(7).toTensor();
    tensor_contract(x, at::kBFloat16, ids.device());
    tensor_contract(router, at::kBFloat16, ids.device());
    prepared_dequant_shape(ids, q13, s13, lookup);
    prepared_dequant_shape(ids, q2, s2, lookup);
    TORCH_CHECK(x.sizes() == at::IntArrayRef({1, 4096}) &&
                router.sizes() == at::IntArrayRef({1, 6}) &&
                q13.sizes() == at::IntArrayRef({256, 16, 131072}) &&
                q2.sizes() == at::IntArrayRef({256, 32, 32768}),
                "prepared MXFP4 MoE is limited to the DeepSeek V4 TP2 single-token shape");
    return {1, 4096};
}

std::vector<int64_t> weighted_sum_shape(const at::Stack& stack) {
    const auto& down = stack.at(0).toTensor();
    const auto& router = stack.at(1).toTensor();
    const auto& token_to_chunk = stack.at(2).toTensor();
    const auto& token_in_chunk = stack.at(3).toTensor();
    tensor_contract(down, at::kBFloat16, down.device());
    tensor_contract(router, at::kBFloat16, down.device());
    tensor_contract(token_to_chunk, at::kInt, down.device());
    tensor_contract(token_in_chunk, at::kInt, down.device());
    TORCH_CHECK(down.sizes() == at::IntArrayRef({6, 4096}) &&
                router.sizes() == at::IntArrayRef({1, 6}) &&
                token_to_chunk.sizes() == at::IntArrayRef({1, 6}) &&
                token_in_chunk.sizes() == at::IntArrayRef({1, 6}),
                "weighted sum is limited to the DeepSeek V4 single-token top-6 shape");
    return {1, 4096};
}

class IndexedMxfp4Mme final : public habana::OpBackend {
    Kind kind_;
 public:
    IndexedMxfp4Mme(int device, c10::ScalarType dtype, Kind kind)
        : OpBackend(device, NO_TPC + std::string("dsv4_indexed_mxfp4_mme"), dtype, {0}, {}, {}, false),
          kind_(kind) {
        SetOutputMetaFn([kind](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16, shape(stack, kind)}};
        });
    }

    synapse_helpers::tensor linear(synapse_helpers::graph& graph, synTensor x, synTensor ids,
                                  synTensor packed, synTensor scales, int batch, int n, int k, bool normal) {
        // No final-result index: neither decoded weights nor BMM outputs
        // become persistent PyTorch allocations at this boundary.
        auto weights = BuildNode(this, graph, {normal ? kDequantNormal : kDequant, {ids, packed, scales},
            {{{6, n, k}, at::kBFloat16}}});
        auto input = ReshapeHelper(graph, x, {batch, 1, k}, at::kBFloat16);
        synGEMMParams params{false, true};
        auto result = BuildNode(this, graph, {"batch_gemm", {input.get(), weights.at(0).get()},
            {{{6, 1, n}, at::kBFloat16}}, &params, sizeof(params)});
        return ReshapeHelper(graph, result.at(0).get(), {6, n}, at::kBFloat16);
    }

    synapse_helpers::tensor half(synapse_helpers::graph& graph, synTensor input, int start) {
        synSliceParams params{};
        for (unsigned i = 0; i < sizeof(params.axes) / sizeof(params.axes[0]); ++i) {
            params.axes[i] = i;
            params.steps[i] = 1;
        }
        params.starts[0] = start;
        params.ends[0] = start + 1024;
        params.ends[1] = 6;
        auto result = BuildNode(this, graph, {"slice", {input},
            {{{6, 1024}, at::kBFloat16}}, &params, sizeof(params)});
        return std::move(result.at(0));
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output_shape = shape(stack, kind_);
        const bool normal = stack.back().toBool();
        if (kind_ == Kind::Dequant) {
            auto result = BuildNode(this, graph, {normal ? kDequantNormal : kDequant,
                {syn_in(0), syn_in(1), syn_in(2)},
                {{output_shape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(result.at(0));
            return;
        }
        if (kind_ == Kind::Linear) {
            const auto& x = stack.at(0).toTensor();
            auto result = linear(graph, syn_in(0), syn_in(1), syn_in(2), syn_in(3),
                                 x.size(0), output_shape[1], x.size(1), normal);
            syn_out(0) = ReshapeHelper(graph, result.get(), output_shape, at::kBFloat16, 0);
            return;
        }
        auto gate_up = linear(graph, syn_in(0), syn_in(1), syn_in(3), syn_in(5), 1, 2048, 4096, normal);
        auto gate = half(graph, gate_up.get(), 0);
        auto up = half(graph, gate_up.get(), 1024);
        auto silu = BuildNode(this, graph, {"silu_fwd_bf16", {gate.get()}, {{{6, 1024}, at::kBFloat16}}});
        auto activated = BuildNode(this, graph, {"mult_fwd_bf16", {silu.at(0).get(), up.get()},
            {{{6, 1024}, at::kBFloat16}}});
        auto down = linear(graph, activated.at(0).get(), syn_in(1), syn_in(4), syn_in(6), 6, 4096, 1024, normal);
        auto router = ReshapeHelper(graph, syn_in(2), {6, 1}, at::kBFloat16);
        auto weighted = BuildNode(this, graph, {"mult_fwd_bf16", {down.get(), router.get()},
            {{{6, 4096}, at::kBFloat16}}});
        // The single-axis GUID uses Params, not the multi-dim mask ABI.
        ns_Reduction::Params params{};
        params.reductionDimension = 1;
        auto result = BuildNode(this, graph, {"reduce_sum_fwd_bf16", {weighted.at(0).get()},
            {{output_shape, at::kBFloat16, 0}}, &params, sizeof(params)});
        syn_out(0) = std::move(result.at(0));
    }
};

class PreparedMxfp4Mme final : public habana::OpBackend {
    Kind kind_;
    bool reduce_;

    synapse_helpers::tensor linear(synapse_helpers::graph& graph, synTensor x, synTensor ids,
                                   synTensor q16, synTensor s16, synTensor lookup,
                                   int batch, int n, int k, bool normal) {
        auto weights = BuildNode(this, graph, {
            normal ? kPreparedDequantNormal : kPreparedDequant,
            {ids, q16, s16, lookup}, {{{6, k, n}, at::kBFloat16}}});
        auto input = ReshapeHelper(graph, x, {batch, 1, k}, at::kBFloat16);
        synGEMMParams params{false, false};
        auto result = BuildNode(this, graph, {"batch_gemm", {input.get(), weights.at(0).get()},
            {{{6, 1, n}, at::kBFloat16}}, &params, sizeof(params)});
        return ReshapeHelper(graph, result.at(0).get(), {6, n}, at::kBFloat16);
    }

    std::pair<synapse_helpers::tensor, synapse_helpers::tensor> gateUp(
            synapse_helpers::graph& graph, synTensor x, synTensor ids, synTensor q16,
            synTensor s16, synTensor lookup, bool normal) {
        constexpr int k = 4096;
        auto gateWeights = BuildNode(this, graph, {
            normal ? kPreparedDequantGateNormal : kPreparedDequantGate,
            {ids, q16, s16, lookup}, {{{6, k, 1024}, at::kBFloat16}}});
        auto upWeights = BuildNode(this, graph, {
            normal ? kPreparedDequantUpNormal : kPreparedDequantUp,
            {ids, q16, s16, lookup}, {{{6, k, 1024}, at::kBFloat16}}});
        auto input = ReshapeHelper(graph, x, {1, 1, k}, at::kBFloat16);
        synGEMMParams params{false, false};
        auto gate = BuildNode(this, graph, {"batch_gemm", {input.get(), gateWeights.at(0).get()},
            {{{6, 1, 1024}, at::kBFloat16}}, &params, sizeof(params)});
        auto up = BuildNode(this, graph, {"batch_gemm", {input.get(), upWeights.at(0).get()},
            {{{6, 1, 1024}, at::kBFloat16}}, &params, sizeof(params)});
        return {
            ReshapeHelper(graph, gate.at(0).get(), {6, 1024}, at::kBFloat16),
            ReshapeHelper(graph, up.at(0).get(), {6, 1024}, at::kBFloat16),
        };
    }

 public:
    PreparedMxfp4Mme(int device, c10::ScalarType dtype, Kind kind, bool reduce = false)
        : OpBackend(device, NO_TPC + std::string("dsv4_prepared_mxfp4_mme"), dtype, {0}, {}, {}, false),
          kind_(kind), reduce_(reduce) {
        SetOutputMetaFn([kind, reduce](const at::Stack& stack) {
            const auto outputShape = reduce ? prepared_moe_shape(stack) : prepared_shape(stack, kind);
            return habana::OutputMetaDataVector{{at::kBFloat16, outputShape}};
        });
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto outputShape = reduce_ ? prepared_moe_shape(stack) : prepared_shape(stack, kind_);
        const bool normal = stack.back().toBool();
        if (kind_ == Kind::Dequant) {
            auto result = BuildNode(this, graph, {
                normal ? kPreparedDequantNormal : kPreparedDequant,
                {syn_in(0), syn_in(1), syn_in(2), syn_in(3)}, {{outputShape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(result.at(0));
            return;
        }
        if (kind_ == Kind::Linear) {
            const auto& x = stack.at(0).toTensor();
            auto result = linear(graph, syn_in(0), syn_in(1), syn_in(2), syn_in(3), syn_in(4),
                                 x.size(0), outputShape[1], x.size(1), normal);
            syn_out(0) = ReshapeHelper(graph, result.get(), outputShape, at::kBFloat16, 0);
            return;
        }
        const int weightOffset = reduce_ ? 1 : 0;
        auto gateAndUp = gateUp(graph, syn_in(0), syn_in(1), syn_in(2 + weightOffset),
                                syn_in(4 + weightOffset), syn_in(6 + weightOffset), normal);
        auto gate = std::move(gateAndUp.first);
        auto up = std::move(gateAndUp.second);
        auto silu = BuildNode(this, graph, {"silu_fwd_bf16", {gate.get()}, {{{6, 1024}, at::kBFloat16}}});
        auto activated = BuildNode(this, graph, {"mult_fwd_bf16", {silu.at(0).get(), up.get()},
            {{{6, 1024}, at::kBFloat16}}});
        auto down = linear(graph, activated.at(0).get(), syn_in(1), syn_in(3 + weightOffset),
                           syn_in(5 + weightOffset), syn_in(6 + weightOffset),
                           6, 4096, 1024, normal);
        if (reduce_) {
            // The stock weighted_sum_reduction_bf16 kernel multiplies BF16
            // inputs and accumulates in FP32, then rounds once to BF16. Keep
            // that exact arithmetic contract inside this compound node so the
            // six-expert [6,4096] intermediate never crosses an HBM boundary.
            auto downF32 = BuildNode(this, graph, {"cast_bf16_to_f32", {down.get()},
                {{{6, 4096}, at::kFloat}}});
            auto router = ReshapeHelper(graph, syn_in(2), {6, 1}, at::kBFloat16);
            auto routerF32 = BuildNode(this, graph, {"cast_bf16_to_f32", {router.get()},
                {{{6, 1}, at::kFloat}}});
            auto weighted = BuildNode(this, graph, {"mult_fwd_f32",
                {downF32.at(0).get(), routerF32.at(0).get()}, {{{6, 4096}, at::kFloat}}});

            // weighted_sum_reduction_bf16 accumulates experts in routing
            // order. A generic reduce_sum may choose a different tree and can
            // change the final BF16 value near cancellation. Materialize the
            // dependency chain explicitly so graph fusion cannot reassociate
            // the six exact BF16*BF16 products.
            auto expertRow = [&](int expert) {
                synSliceParams sliceParams{};
                for (unsigned i = 0; i < sizeof(sliceParams.axes) / sizeof(sliceParams.axes[0]); ++i) {
                    sliceParams.axes[i] = i;
                    sliceParams.steps[i] = 1;
                }
                // Synapse axis 0 is the innermost [4096] dimension and axis 1
                // is the six-expert dimension for this rank-two tensor.
                sliceParams.starts[1] = expert;
                sliceParams.ends[0] = 4096;
                sliceParams.ends[1] = expert + 1;
                auto row = BuildNode(this, graph, {"slice", {weighted.at(0).get()},
                    {{{1, 4096}, at::kFloat}}, &sliceParams, sizeof(sliceParams)});
                return std::move(row.at(0));
            };
            auto accumulated = expertRow(0);
            for (int expert = 1; expert < 6; ++expert) {
                auto next = expertRow(expert);
                auto sum = BuildNode(this, graph, {"add_fwd_f32", {accumulated.get(), next.get()},
                    {{{1, 4096}, at::kFloat}}});
                accumulated = std::move(sum.at(0));
            }
            auto converted = BuildNode(this, graph, {"cast_f32_to_bf16", {accumulated.get()},
                {{outputShape, at::kBFloat16}}});
            // A regular memcpy can be folded and still permit a later node to
            // consume the pre-rounding FP32 producer. The custom TPC boundary
            // performs a real BF16 load/store before exposing the result.
            auto result = BuildNode(this, graph, {kBF16Identity, {converted.at(0).get()},
                {{outputShape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(result.at(0));
            return;
        }
        // Synapse cannot compile the stock weighted-sum GUID directly against
        // this compound op's internal batch_gemm output. Expose only the 48 KiB
        // six-expert result at the native-op boundary; decoded BF16 weights
        // remain internal recipe tensors and are still eligible for SRAM.
        syn_out(0) = ReshapeHelper(graph, down.get(), outputShape, at::kBFloat16, 0);
    }
};

class WeightedSum final : public habana::OpBackend {
 public:
    WeightedSum(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv4_weighted_sum"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16, weighted_sum_shape(stack)}};
        });
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto outputShape = weighted_sum_shape(stack);
        // The stock MoE GUID consumes [chunks, tokens-per-chunk, hidden].
        // Our fixed top-6 case has one token in each of six chunks.
        auto down = ReshapeHelper(graph, syn_in(0), {6, 1, 4096}, at::kBFloat16);
        auto result = BuildNode(this, graph, {"weighted_sum_reduction_bf16",
            {down.get(), syn_in(1), syn_in(2), syn_in(3)},
            {{outputShape, at::kBFloat16, 0}}});
        syn_out(0) = std::move(result.at(0));
    }
};

const char* schema(Kind kind) {
    return kind == Kind::Dequant ? kDequantSchema : kind == Kind::Linear ? kLinearSchema : kMoeSchema;
}
const bool registered = [] {
    for (Kind kind : {Kind::Dequant, Kind::Linear, Kind::Moe}) {
        habana::custom_op::registerUserCustomOp(schema(kind), kDequant, [kind](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, shape(stack, kind)}};
        }, nullptr);
        habana::KernelRegistry().add(schema(kind), [kind](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<IndexedMxfp4Mme>(device, dtype, kind);
        });
    }
    habana::custom_op::registerUserCustomOp(
        kPreparedMoeSchema, kPreparedDequant, [](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, prepared_moe_shape(stack)}};
        }, nullptr);
    habana::KernelRegistry().add(kPreparedMoeSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PreparedMxfp4Mme>(device, dtype, Kind::Moe, true);
    });
    for (Kind kind : {Kind::Dequant, Kind::Linear, Kind::Moe}) {
        const char* preparedSchema = kind == Kind::Dequant ? kPreparedDequantSchema
            : kind == Kind::Linear ? kPreparedLinearSchema : kPreparedExpertsSchema;
        habana::custom_op::registerUserCustomOp(preparedSchema, kPreparedDequant, [kind](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, prepared_shape(stack, kind)}};
        }, nullptr);
        habana::KernelRegistry().add(preparedSchema, [kind](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<PreparedMxfp4Mme>(device, dtype, kind);
        });
    }
    habana::custom_op::registerUserCustomOp(
        kWeightedSumSchema, "weighted_sum_reduction_bf16", [](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, weighted_sum_shape(stack)}};
        }, nullptr);
    habana::KernelRegistry().add(kWeightedSumSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<WeightedSum>(device, dtype);
    });
    return true;
}();

at::Tensor execute(const at::Stack& stack, Kind kind) {
    shape(stack, kind);
    TORCH_CHECK(registered && stack.at(0).toTensor().device().type() == at::kHPU,
                "indexed MXFP4 requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema(kind));
    return descriptor.execute(stack).at(0);
}
at::Tensor meta(const at::Stack& stack, Kind kind) {
    return at::empty(shape(stack, kind), stack.at(0).toTensor().options().dtype(at::kBFloat16));
}
at::Tensor prepared_execute(const at::Stack& stack, Kind kind) {
    prepared_shape(stack, kind);
    TORCH_CHECK(registered && stack.at(0).toTensor().device().type() == at::kHPU,
                "prepared MXFP4 requires HPU");
    const char* preparedSchema = kind == Kind::Dequant ? kPreparedDequantSchema
        : kind == Kind::Linear ? kPreparedLinearSchema : kPreparedExpertsSchema;
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(preparedSchema);
    return descriptor.execute(stack).at(0);
}
at::Tensor prepared_meta(const at::Stack& stack, Kind kind) {
    return at::empty(prepared_shape(stack, kind), stack.at(0).toTensor().options().dtype(at::kBFloat16));
}

at::Tensor dequant(const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, bool normal) {
    return execute({ids, w, s, normal}, Kind::Dequant);
}
at::Tensor dequant_meta(const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, bool normal) {
    return meta({ids, w, s, normal}, Kind::Dequant);
}
at::Tensor linear(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, bool normal) {
    return execute({x, ids, w, s, normal}, Kind::Linear);
}
at::Tensor linear_meta(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& w, const at::Tensor& s, bool normal) {
    return meta({x, ids, w, s, normal}, Kind::Linear);
}
at::Tensor moe(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router, const at::Tensor& w13,
               const at::Tensor& w2, const at::Tensor& s13, const at::Tensor& s2, bool normal) {
    return execute({x, ids, router, w13, w2, s13, s2, normal}, Kind::Moe);
}
at::Tensor moe_meta(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router, const at::Tensor& w13,
                    const at::Tensor& w2, const at::Tensor& s13, const at::Tensor& s2, bool normal) {
    return meta({x, ids, router, w13, w2, s13, s2, normal}, Kind::Moe);
}
at::Tensor prepared_dequant(const at::Tensor& ids, const at::Tensor& q16, const at::Tensor& s16,
                            const at::Tensor& lookup, bool normal) {
    return prepared_execute({ids, q16, s16, lookup, normal}, Kind::Dequant);
}
at::Tensor prepared_dequant_meta(const at::Tensor& ids, const at::Tensor& q16, const at::Tensor& s16,
                                 const at::Tensor& lookup, bool normal) {
    return prepared_meta({ids, q16, s16, lookup, normal}, Kind::Dequant);
}
at::Tensor prepared_linear(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& q16,
                           const at::Tensor& s16, const at::Tensor& lookup, bool normal) {
    return prepared_execute({x, ids, q16, s16, lookup, normal}, Kind::Linear);
}
at::Tensor prepared_linear_meta(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& q16,
                                const at::Tensor& s16, const at::Tensor& lookup, bool normal) {
    return prepared_meta({x, ids, q16, s16, lookup, normal}, Kind::Linear);
}
at::Tensor prepared_experts(const at::Tensor& x, const at::Tensor& ids,
                            const at::Tensor& q13, const at::Tensor& q2,
                            const at::Tensor& s13, const at::Tensor& s2,
                            const at::Tensor& lookup, bool normal) {
    return prepared_execute({x, ids, q13, q2, s13, s2, lookup, normal}, Kind::Moe);
}
at::Tensor prepared_experts_meta(const at::Tensor& x, const at::Tensor& ids,
                                 const at::Tensor& q13, const at::Tensor& q2,
                                 const at::Tensor& s13, const at::Tensor& s2,
                                 const at::Tensor& lookup, bool normal) {
    return prepared_meta({x, ids, q13, q2, s13, s2, lookup, normal}, Kind::Moe);
}
at::Tensor prepared_moe(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router,
                        const at::Tensor& q13, const at::Tensor& q2,
                        const at::Tensor& s13, const at::Tensor& s2,
                        const at::Tensor& lookup, bool normal) {
    auto stack = at::Stack{x, ids, router, q13, q2, s13, s2, lookup, normal};
    prepared_moe_shape(stack);
    TORCH_CHECK(registered && x.device().type() == at::kHPU,
                "prepared MXFP4 MoE requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPreparedMoeSchema);
    return descriptor.execute(stack).at(0);
}
at::Tensor prepared_moe_meta(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router,
                             const at::Tensor& q13, const at::Tensor& q2,
                             const at::Tensor& s13, const at::Tensor& s2,
                             const at::Tensor& lookup, bool normal) {
    auto stack = at::Stack{x, ids, router, q13, q2, s13, s2, lookup, normal};
    return at::empty(prepared_moe_shape(stack), x.options().dtype(at::kBFloat16));
}
at::Tensor weighted_sum(const at::Tensor& down, const at::Tensor& router,
                        const at::Tensor& token_to_chunk, const at::Tensor& token_in_chunk) {
    auto stack = at::Stack{down, router, token_to_chunk, token_in_chunk};
    weighted_sum_shape(stack);
    TORCH_CHECK(registered && down.device().type() == at::kHPU,
                "weighted sum requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kWeightedSumSchema);
    return descriptor.execute(stack).at(0);
}
at::Tensor weighted_sum_meta(const at::Tensor& down, const at::Tensor& router,
                             const at::Tensor& token_to_chunk, const at::Tensor& token_in_chunk) {
    auto stack = at::Stack{down, router, token_to_chunk, token_in_chunk};
    return at::empty(weighted_sum_shape(stack), down.options().dtype(at::kBFloat16));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2("
          "Tensor ids, Tensor w, Tensor s, bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_mxfp4_indexed_linear_bf16_gaudi2("
          "Tensor x, Tensor ids, Tensor w, Tensor s, bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, "
          "Tensor w13, Tensor w2, Tensor s13, Tensor s2, bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2("
          "Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_mxfp4_prepared_linear_bf16_gaudi2("
          "Tensor x, Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_mxfp4_prepared_experts_bf16_gaudi2(Tensor x, Tensor ids, "
          "Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, "
          "bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, "
          "Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, "
          "bool normal_scales=False) -> Tensor");
    m.def("custom_deepseek_v4_weighted_sum_bf16_gaudi2(Tensor down, Tensor router, "
          "Tensor token_to_chunk, Tensor token_in_chunk) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2", dequant);
    m.impl("custom_deepseek_v4_mxfp4_indexed_linear_bf16_gaudi2", linear);
    m.impl("custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2", moe);
    m.impl("custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2", prepared_dequant);
    m.impl("custom_deepseek_v4_mxfp4_prepared_linear_bf16_gaudi2", prepared_linear);
    m.impl("custom_deepseek_v4_mxfp4_prepared_experts_bf16_gaudi2", prepared_experts);
    m.impl("custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2", prepared_moe);
    m.impl("custom_deepseek_v4_weighted_sum_bf16_gaudi2", weighted_sum);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2", dequant_meta);
    m.impl("custom_deepseek_v4_mxfp4_indexed_linear_bf16_gaudi2", linear_meta);
    m.impl("custom_deepseek_v4_mxfp4_indexed_mme_bf16_gaudi2", moe_meta);
    m.impl("custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2", prepared_dequant_meta);
    m.impl("custom_deepseek_v4_mxfp4_prepared_linear_bf16_gaudi2", prepared_linear_meta);
    m.impl("custom_deepseek_v4_mxfp4_prepared_experts_bf16_gaudi2", prepared_experts_meta);
    m.impl("custom_deepseek_v4_mxfp4_prepared_moe_bf16_gaudi2", prepared_moe_meta);
    m.impl("custom_deepseek_v4_weighted_sum_bf16_gaudi2", weighted_sum_meta);
}
