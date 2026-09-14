// SPDX-License-Identifier: Apache-2.0
// V4.1 reuses the V4 Q16 TPC binary and SRAM -> MME dataflow. Its compound
// graph has separate shape/activation contracts; V4 registrations are intact.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <limits>
#include "perf_lib_layer_params.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kDecode = "custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2";
constexpr auto kDecodeNormal = "custom_deepseek_v41_mxfp4_prepared_dequant_normal_bf16_gaudi2";
constexpr auto kDecodeSchema = "custom_op::custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2";
constexpr auto kMoeSchema = "custom_op::custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2";
constexpr auto kTiledMoeSchema = "custom_op::custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2";
constexpr auto kN512MoeSchema = "custom_op::custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2";
constexpr auto kColumnMoeSchema = "custom_op::custom_deepseek_v41_mxfp4_column_moe_bf16_gaudi2";
constexpr auto kRound = "custom_deepseek_v41_bf16_identity_gaudi2";

void contract(const at::Tensor& tensor, at::ScalarType type, const at::Device& device) {
    TORCH_CHECK(tensor.scalar_type() == type && tensor.device() == device && tensor.is_contiguous() &&
                !tensor.requires_grad(), "V4.1 prepared MXFP4 requires contiguous inference inputs with matching dtype/device");
}

std::vector<int64_t> decode_shape(const at::Stack& stack) {
    const auto& ids = stack.at(0).toTensor();
    const auto& q = stack.at(1).toTensor();
    const auto& s = stack.at(2).toTensor();
    const auto& lut = stack.at(3).toTensor();
    contract(ids, at::kInt, q.device());
    contract(q, at::kShort, q.device());
    contract(s, at::kBFloat16, q.device());
    contract(lut, at::kBFloat16, q.device());
    TORCH_CHECK(ids.dim() == 2 && ids.numel() > 0 && ids.numel() <= 3072,
                "V4.1 decode expects [tokens, ordered experts] IDs");
    TORCH_CHECK(q.dim() == 3 && q.size(0) > 0 && q.size(0) <= 384 && q.size(1) > 0 && q.size(1) <= 40 &&
                q.size(2) > 0 && q.size(2) <= 163840 && q.size(2) % 4096 == 0,
                "V4.1 Q16 expects [E,N/128,(K/2)*64], with K a multiple of 128");
    TORCH_CHECK(s.dim() == 3 && s.size(0) == q.size(0) && s.size(1) == q.size(1) && s.size(2) * 8 == q.size(2),
                "V4.1 S16 must preserve group-32 scale encodings");
    TORCH_CHECK(lut.sizes() == at::IntArrayRef({128}), "V4.1 requires the exact V4 MXFP4 lookup bytes");
    return {ids.numel(), q.size(2) / 32, q.size(1) * 128};
}

std::vector<int64_t> moe_shape(const at::Stack& stack) {
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    const auto& router = stack.at(2).toTensor();
    contract(x, at::kBFloat16, ids.device());
    contract(router, at::kFloat, ids.device());
    auto first = decode_shape({ids, stack.at(3), stack.at(5), stack.at(7)});
    auto second = decode_shape({ids, stack.at(4), stack.at(6), stack.at(7)});
    TORCH_CHECK(x.dim() == 2 && x.size(0) == ids.size(0) && x.size(1) == first[1] &&
                (ids.size(1) == 3 || ids.size(1) == 6) && router.sizes() == ids.sizes() &&
                first[2] == 2 * second[1] && second[2] == x.size(1) &&
                stack.at(3).toTensor().size(0) == stack.at(4).toTensor().size(0),
                "V4.1 MoE shape must connect W13, W2 and ordered top-3/top-6 routing");
    return {x.size(0), x.size(1)};
}

class PreparedV41 final : public habana::OpBackend {
    bool moe_;
    bool tiled_;
    bool n512_;
    bool column_;
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

    Tensor column_linear(synapse_helpers::graph& graph, synTensor input, synTensor ids, synTensor q,
                         synTensor s, synTensor lookup, int tokens, int experts, int n, int k, bool normal) {
        TORCH_CHECK(n == 2304 && k == 5120 && experts == 6 && tokens >= 2 && tokens <= 6,
                    "Column W13 requires the V4.1 C2-C6 top-6 shape");
        const int columns = ((tokens * experts * n + 1023) / 1024) * 1024;
        auto decoded = BuildNode(this, graph, {
            normal ? "custom_deepseek_v41_mxfp4_column_dequant_normal_bf16_gaudi2"
                   : "custom_deepseek_v41_mxfp4_column_dequant_bf16_gaudi2",
            {ids, q, s, lookup}, {{{k, columns}, at::kBFloat16}}});
        synGEMMParams params{false, false};
        auto all = BuildNode(this, graph, {"gemm", {input, decoded.at(0).get()},
            {{{tokens, columns}, at::kBFloat16}}, &params, sizeof(params)});
        // Only this diagonal belongs to the routed computation. Other rows
        // are private GEMM outputs and never enter an activation or reduction.
        std::vector<Tensor> parts;
        std::vector<synTensor> inputs;
        for (int token = 0; token < tokens; ++token) {
            synSliceParams p{};
            for (unsigned i = 0; i < sizeof(p.axes) / sizeof(p.axes[0]); ++i) {
                p.axes[i] = i;
                p.steps[i] = 1;
            }
            p.starts[0] = token * experts * n;
            p.ends[0] = (token + 1) * experts * n;
            p.starts[1] = token;
            p.ends[1] = token + 1;
            auto selected = BuildNode(this, graph, {"slice", {all.at(0).get()},
                {{{1, experts * n}, at::kBFloat16}}, &p, sizeof(p)});
            parts.push_back(std::move(selected.at(0)));
            inputs.push_back(parts.back().get());
        }
        synConcatenateParams concat{};
        concat.axis = 1;
        auto result = BuildNode(this, graph, {"concat", std::move(inputs),
            {{{tokens, experts * n}, at::kBFloat16}}, &concat, sizeof(concat)});
        return ReshapeHelper(graph, result.at(0).get(), {tokens, experts, n}, at::kBFloat16);
    }

    Tensor linear(synapse_helpers::graph& graph, synTensor input, synTensor ids, synTensor q,
                  synTensor s, synTensor lookup, int tokens, int experts, int n, int k,
                  bool broadcast, bool normal) {
        if (column_ && broadcast)
            return column_linear(graph, input, ids, q, s, lookup, tokens, experts, n, k, normal);
        const char* guid = tiled_ ? (normal ? "custom_deepseek_v41_mxfp4_k128_dequant_normal_bf16_gaudi2"
                                           : "custom_deepseek_v41_mxfp4_k128_dequant_bf16_gaudi2")
                                 : (normal ? kDecodeNormal : kDecode);
        std::vector<Tensor> decoded;
        if (!(n512_ && broadcast)) {
            decoded = BuildNode(this, graph, {guid, {ids, q, s, lookup},
                {{{tokens * experts, k, n}, at::kBFloat16}}});
        }
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
        if (n512_ && broadcast) {
            // Only independent output channels are split. Every MME retains
            // the original full-K input order and BF16 result boundary.
            std::vector<Tensor> parts;
            std::vector<synTensor> inputs;
            for (int begin = 0; begin < n;) {
                // Do not cross the gate/up split: its two downstream slices
                // otherwise leave one decoded producer outside the SRAM bundle.
                const int width = std::min(512, n / 2 - begin % (n / 2));
                int32_t window[] = {begin / 128, width / 128};
                auto decoded = BuildNode(this, graph, {
                    normal ? "custom_deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2"
                           : "custom_deepseek_v41_mxfp4_n512_dequant_bf16_gaudi2",
                    {ids, q, s, lookup}, {{{tokens * experts, k, width}, at::kBFloat16}},
                    window, sizeof(window)});
                auto result = BuildNode(this, graph, {"batch_gemm", {activations.get(), decoded.at(0).get()},
                    {{{tokens * experts, 1, width}, at::kBFloat16}}, &params, sizeof(params)});
                parts.push_back(std::move(result.at(0)));
                inputs.push_back(parts.back().get());
                begin += width;
            }
            synConcatenateParams concat{};
            concat.axis = 0;
            auto result = BuildNode(this, graph, {"concat", std::move(inputs),
                {{{tokens * experts, 1, n}, at::kBFloat16}}, &concat, sizeof(concat)});
            return ReshapeHelper(graph, result.at(0).get(), {tokens, experts, n}, at::kBFloat16);
        }
        auto result = BuildNode(this, graph, {"batch_gemm", {activations.get(), decoded.at(0).get()},
            {{{tokens * experts, 1, n}, at::kBFloat16}}, &params, sizeof(params)});
        return ReshapeHelper(graph, result.at(0).get(), {tokens, experts, n}, at::kBFloat16);
    }

 public:
    PreparedV41(int device, c10::ScalarType dtype, bool moe, bool tiled = false, bool n512 = false, bool column = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_prepared_mxfp4"), dtype, {0}, {}, {}, false),
          moe_(moe), tiled_(tiled), n512_(n512), column_(column) {
        SetOutputMetaFn([moe](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16, moe ? moe_shape(stack) : decode_shape(stack)}};
        });
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const bool normal = stack.back().toBool();
        const auto resultShape = moe_ ? moe_shape(stack) : decode_shape(stack);
        const auto& idsTensor = stack.at(moe_ ? 1 : 0).toTensor();
        auto ids = ReshapeHelper(graph, syn_in(moe_ ? 1 : 0), {1, idsTensor.numel()}, at::kInt);
        if (!moe_) {
            auto result = BuildNode(this, graph, {normal ? kDecodeNormal : kDecode,
                {ids.get(), syn_in(1), syn_in(2), syn_in(3)}, {{resultShape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(result.at(0));
            return;
        }
        const int tokens = idsTensor.size(0), experts = idsTensor.size(1), hidden = resultShape[1];
        const int intermediate = stack.at(4).toTensor().size(2) / 32;
        const std::vector<int64_t> shape{tokens, experts, intermediate};
        auto first = linear(graph, syn_in(0), ids.get(), syn_in(3), syn_in(5), syn_in(7),
                            tokens, experts, intermediate * 2, hidden, true, normal);
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
        auto down = linear(graph, middle.get(), ids.get(), syn_in(4), syn_in(6), syn_in(7),
                           tokens, experts, hidden, intermediate, false, normal);
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
    for (bool moe : {false, true}) {
        const char* schema = moe ? kMoeSchema : kDecodeSchema;
        habana::custom_op::registerUserCustomOp(schema, kDecode, [moe](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, moe ? moe_shape(stack) : decode_shape(stack)}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [moe](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<PreparedV41>(device, dtype, moe);
        });
    }
    habana::custom_op::registerUserCustomOp(kTiledMoeSchema, kDecode, [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, moe_shape(stack)}};
    }, nullptr);
    habana::KernelRegistry().add(kTiledMoeSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PreparedV41>(device, dtype, true, true);
    });
    habana::custom_op::registerUserCustomOp(kN512MoeSchema, kDecode, [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, moe_shape(stack)}};
    }, nullptr);
    habana::KernelRegistry().add(kN512MoeSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PreparedV41>(device, dtype, true, true, true);
    });
    habana::custom_op::registerUserCustomOp(kColumnMoeSchema, kDecode, [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, moe_shape(stack)}};
    }, nullptr);
    habana::KernelRegistry().add(kColumnMoeSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PreparedV41>(device, dtype, true, true, false, true);
    });
    return true;
}();

at::Tensor run(const at::Stack& stack, bool moe, bool meta, bool tiled = false, bool n512 = false, bool column = false) {
    const auto shape = moe ? moe_shape(stack) : decode_shape(stack);
    if (meta) return at::empty(shape, stack.at(0).toTensor().options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && stack.at(0).toTensor().device().type() == at::kHPU, "V4.1 MXFP4 requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        column ? kColumnMoeSchema : (n512 ? kN512MoeSchema : (tiled ? kTiledMoeSchema : (moe ? kMoeSchema : kDecodeSchema))));
    return descriptor.execute(stack).at(0);
}

template<bool Meta> at::Tensor dequant(const at::Tensor& ids, const at::Tensor& q, const at::Tensor& s,
                                      const at::Tensor& lookup, bool normal) {
    return run({ids, q, s, lookup, normal}, false, Meta);
}
template<bool Meta, bool Tiled = false, bool N512 = false, bool Column = false> at::Tensor moe(const at::Tensor& x, const at::Tensor& ids, const at::Tensor& router,
                                  const at::Tensor& q13, const at::Tensor& q2, const at::Tensor& s13,
                                  const at::Tensor& s2, const at::Tensor& lookup, bool normal) {
    return run({x, ids, router, q13, q2, s13, s2, lookup, normal}, true, Meta, Tiled, N512, Column);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2(Tensor ids, Tensor q16, Tensor s16, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_mxfp4_column_moe_bf16_gaudi2(Tensor x, Tensor ids, Tensor router, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, bool normal=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2", dequant<false>);
    m.impl("custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2", moe<false>);
    m.impl("custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2", moe<false, true>);
    m.impl("custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2", moe<false, true, true>);
    m.impl("custom_deepseek_v41_mxfp4_column_moe_bf16_gaudi2", moe<false, true, false, true>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2", dequant<true>);
    m.impl("custom_deepseek_v41_mxfp4_prepared_moe_bf16_gaudi2", moe<true>);
    m.impl("custom_deepseek_v41_mxfp4_k128_moe_bf16_gaudi2", moe<true, true>);
    m.impl("custom_deepseek_v41_mxfp4_n512_moe_bf16_gaudi2", moe<true, true, true>);
    m.impl("custom_deepseek_v41_mxfp4_column_moe_bf16_gaudi2", moe<true, true, false, true>);
}

namespace {
constexpr auto kSharedLinearSchema = "custom_op::custom_deepseek_v41_mxfp4_shared_linear_bf16_gaudi2";
constexpr auto kBarrierSchema = "custom_op::custom_deepseek_v41_bf16_identity_gaudi2";

std::vector<int64_t> shared_linear_shape(const at::Stack& stack) {
    const auto& x = stack.at(0).toTensor();
    const auto& ids = stack.at(1).toTensor();
    const auto& active = stack.at(2).toTensor();
    auto decoded = decode_shape({ids, stack.at(3), stack.at(4), stack.at(5)});
    contract(x, at::kBFloat16, ids.device());
    contract(active, at::kInt, ids.device());
    TORCH_CHECK(ids.size(0) == 1 && ids.numel() <= 36 && active.sizes() == ids.sizes() &&
                x.dim() == 3 && (x.size(0) == 1 || x.size(0) == ids.numel()) &&
                x.size(1) >= 1 && x.size(1) <= 6 && ids.numel() == x.size(1) * 6 &&
                x.size(2) == decoded[1], "V4.1 shared expert linear contract changed");
    return {decoded[0], x.size(1), decoded[2]};
}

std::vector<int64_t> barrier_shape(const at::Stack& stack) {
    const auto& x = stack.at(0).toTensor();
    contract(x, at::kBFloat16, x.device());
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1 && x.size(1) > 0,
                "V4.1 BF16 barrier expects a nonempty contiguous row");
    return x.sizes().vec();
}

class SharedLinearV41 final : public habana::OpBackend {
    bool barrier_;
    bool tiled_;
public:
    SharedLinearV41(int device, c10::ScalarType dtype, bool barrier, bool tiled = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_shared_expert_linear"), dtype,
                    {0}, {}, {}, false), barrier_(barrier), tiled_(tiled) {
        SetOutputMetaFn([barrier](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16,
                barrier ? barrier_shape(stack) : shared_linear_shape(stack)}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        if (barrier_) {
            auto result = BuildNode(this, graph, {kRound, {syn_in(0)},
                {{barrier_shape(stack), at::kBFloat16, 0}}});
            syn_out(0) = std::move(result.at(0));
            return;
        }
        const auto shape = shared_linear_shape(stack);
        const int64_t k = stack.at(0).toTensor().size(2);
        const bool normal = stack.at(6).toBool();
        auto decoded = BuildNode(this, graph, {
            tiled_ ? (normal ? "custom_deepseek_v41_mxfp4_reuse128_dequant_normal_bf16_gaudi2"
                             : "custom_deepseek_v41_mxfp4_reuse128_dequant_bf16_gaudi2")
                   : (normal ? "custom_deepseek_v41_mxfp4_shared_dequant_normal_bf16_gaudi2"
                             : "custom_deepseek_v41_mxfp4_shared_dequant_bf16_gaudi2"),
            {syn_in(1), syn_in(3), syn_in(4), syn_in(5), syn_in(2)},
            {{{shape[0], k, shape[2]}, at::kBFloat16}}});
        synGEMMParams params{false, false};
        auto result = BuildNode(this, graph, {"batch_gemm", {syn_in(0), decoded.at(0).get()},
            {{shape, at::kBFloat16, 0}}, &params, sizeof(params)});
        syn_out(0) = std::move(result.at(0));
    }
};

constexpr auto kSharedTiledLinearSchema = "custom_op::custom_deepseek_v41_mxfp4_shared_k128_linear_bf16_gaudi2";

const bool sharedRegistered = [] {
    for (bool barrier : {false, true}) {
        const char* schema = barrier ? kBarrierSchema : kSharedLinearSchema;
        habana::custom_op::registerUserCustomOp(schema, kRound, [barrier](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16,
                barrier ? barrier_shape(stack) : shared_linear_shape(stack)}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [barrier](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<SharedLinearV41>(device, dtype, barrier);
        });
    }
    habana::custom_op::registerUserCustomOp(kSharedTiledLinearSchema, kRound, [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, shared_linear_shape(stack)}};
    }, nullptr);
    habana::KernelRegistry().add(kSharedTiledLinearSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<SharedLinearV41>(device, dtype, false, true);
    });
    return true;
}();

template<bool Meta, bool Tiled = false> at::Tensor shared_linear(const at::Tensor& x, const at::Tensor& ids,
    const at::Tensor& active, const at::Tensor& q, const at::Tensor& s,
    const at::Tensor& lut, bool normal) {
    const at::Stack stack{x, ids, active, q, s, lut, normal};
    const auto shape = shared_linear_shape(stack);
    if (Meta) return at::empty(shape, x.options());
    TORCH_CHECK(sharedRegistered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Tiled ? kSharedTiledLinearSchema : kSharedLinearSchema);
    return descriptor.execute(stack).at(0);
}
template<bool Meta> at::Tensor barrier(const at::Tensor& x) {
    const at::Stack stack{x};
    const auto shape = barrier_shape(stack);
    if (Meta) return at::empty(shape, x.options());
    TORCH_CHECK(sharedRegistered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kBarrierSchema);
    return descriptor.execute(stack).at(0);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_mxfp4_shared_linear_bf16_gaudi2(Tensor x, Tensor ids, Tensor active, Tensor q, Tensor s, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_mxfp4_shared_k128_linear_bf16_gaudi2(Tensor x, Tensor ids, Tensor active, Tensor q, Tensor s, Tensor lookup, bool normal=False) -> Tensor");
    m.def("custom_deepseek_v41_bf16_identity_gaudi2(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_mxfp4_shared_linear_bf16_gaudi2", shared_linear<false>);
    m.impl("custom_deepseek_v41_mxfp4_shared_k128_linear_bf16_gaudi2", shared_linear<false, true>);
    m.impl("custom_deepseek_v41_bf16_identity_gaudi2", barrier<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_mxfp4_shared_linear_bf16_gaudi2", shared_linear<true>);
    m.impl("custom_deepseek_v41_mxfp4_shared_k128_linear_bf16_gaudi2", shared_linear<true, true>);
    m.impl("custom_deepseek_v41_bf16_identity_gaudi2", barrier<true>);
}
