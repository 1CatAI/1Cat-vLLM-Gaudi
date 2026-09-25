// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kProjection = "custom_op::custom_deepseek_v41_q_projection_rope_gaudi2";
constexpr auto kNormProjection = "custom_op::custom_deepseek_v41_q_norm_projection_rope_gaudi2";
constexpr auto kEpilogue = "custom_op::custom_deepseek_v41_q_scale_rope_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_q_scale_rope_gaudi2";
constexpr auto kNormQuant = "custom_deepseek_v41_qnorm_quant_gaudi2";
struct NormParams { float epsilon; float inverse_width; };
habana::OutputMetaDataVector meta(const at::Stack& stack, bool epilogue) {
    TORCH_CHECK(stack.size() == 5, "Q projection/RoPE requires five tensors");
    const auto x = stack.at(0).toTensor();
    for (const auto& item : stack) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "Q projection/RoPE requires contiguous matching inference tensors");
    }
    const auto w = stack.at(1).toTensor(), s = stack.at(2).toTensor();
    const auto rows = x.dim() == 2 ? x.size(0) : 0;
    TORCH_CHECK(rows >= 1 && rows <= 64, "Q projection requires 1..64 request rows");
    if (epilogue) {
        TORCH_CHECK(x.scalar_type() == at::kFloat && x.sizes() == at::IntArrayRef({rows,16384}) &&
                    w.scalar_type() == at::kFloat && w.sizes() == at::IntArrayRef({1,16384}) &&
                    s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({rows,1}),
                    "Q epilogue requires F32 product [B,16384], channel scales [1,16384] and activation scale [B,1]");
    } else {
        TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.sizes() == at::IntArrayRef({rows,1280}) &&
                    w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.sizes() == at::IntArrayRef({16384,1280}) &&
                    s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({1,16384}),
                    "Q projection requires BF16 [B,1280], prepared FP8 [16384,1280], F32 scales [1,16384]");
    }
    const auto pos = stack.at(3).toTensor(), table = stack.at(4).toTensor();
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.sizes() == at::IntArrayRef({rows}) &&
                table.scalar_type() == at::kFloat && table.dim() == 2 && table.size(0) > 0 &&
                table.size(0) <= 1048576 && table.size(1) == 64,
                "Q RoPE requires I32 position [B] and F32 cos/sin [1..1048576,64]");
    return {{at::kBFloat16, {rows,16384}}};
}
habana::OutputMetaDataVector norm_meta(const at::Stack& stack) {
    TORCH_CHECK(stack.size() == 7, "Q norm/projection/RoPE requires six tensors and epsilon");
    const auto x = stack.at(0).toTensor(), norm = stack.at(1).toTensor();
    const auto w = stack.at(2).toTensor(), s = stack.at(3).toTensor();
    const auto pos = stack.at(4).toTensor(), table = stack.at(5).toTensor();
    const auto epsilon = stack.at(6).toDouble();
    const auto rows = x.dim() == 2 ? x.size(0) : 0;
    TORCH_CHECK(rows >= 1 && rows <= 64, "Q norm projection requires 1..64 request rows");
    for (int index = 0; index < 6; ++index) {
        const auto t = stack.at(index).toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "Q norm/projection/RoPE requires contiguous matching inference tensors");
    }
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.sizes() == at::IntArrayRef({rows,1280}) &&
                norm.scalar_type() == at::kBFloat16 && norm.sizes() == at::IntArrayRef({1280}) &&
                w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                w.sizes() == at::IntArrayRef({16384,1280}) &&
                s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({1,16384}) &&
                pos.scalar_type() == at::kInt && pos.sizes() == at::IntArrayRef({rows}) &&
                table.scalar_type() == at::kFloat && table.dim() == 2 && table.size(0) > 0 &&
                table.size(0) <= 1048576 && table.size(1) == 64 &&
                std::isnormal(static_cast<float>(epsilon)) && epsilon > 0,
                "Q norm projection requires BF16 [B,1280]/[1280], FP8 [16384,1280], "
                "F32 [1,16384], I32 [B], F32 [L,64], positive normal epsilon");
    return {{at::kBFloat16, {rows,16384}}};
}
class Projection final : public habana::OpBackend {
    bool epilogue_;
    bool fused_norm_;
public:
    Projection(int device, c10::ScalarType dtype, bool epilogue, bool fused_norm = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_q_projection_rope"), dtype, {0}, {}, {}, false),
          epilogue_(epilogue), fused_norm_(fused_norm) {
        SetOutputMetaFn([epilogue, fused_norm](const at::Stack& s) {
            return fused_norm ? norm_meta(s) : meta(s, epilogue);
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = fused_norm_ ? norm_meta(stack) : meta(stack, epilogue_);
        const auto rows = output.at(0).shape.at(0);
        if (epilogue_) {
            auto node = BuildNode(this, graph, {kGuid, {syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4)},
                {{output.at(0).shape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(node.at(0)); return;
        }
        std::vector<synapse_helpers::tensor> q;
        int weight = 1, channel = 2, position = 3, table = 4;
        if (fused_norm_) {
            NormParams scalar{static_cast<float>(stack.at(6).toDouble()), 1.0f / 1280.0f};
            q = BuildNode(this, graph, {kNormQuant, {syn_in(0),syn_in(1)},
                {{{rows,1280}, at::ScalarType::Float8_e4m3fn}, {{rows,1}, at::kFloat}},
                &scalar, sizeof(scalar)});
            weight = 2; channel = 3; position = 4; table = 5;
        } else {
            q = BuildNode(this, graph, {"custom_deepseek_v41_dense_quant_gaudi2", {syn_in(0)},
                {{{rows,1280}, at::ScalarType::Float8_e4m3fn}, {{rows,1}, at::kFloat}}});
        }
        synGEMMParams params{false,true};
        auto product = BuildNode(this, graph, {"gemm", {q.at(0).get(),syn_in(weight)},
            {{output.at(0).shape, at::kFloat}}, &params, sizeof(params)});
        auto node = BuildNode(this, graph, {kGuid,
            {product.at(0).get(),syn_in(channel),q.at(1).get(),syn_in(position),syn_in(table)},
            {{output.at(0).shape, at::kBFloat16, 0}}});
        syn_out(0) = std::move(node.at(0));
    }
};
const bool registered = [] {
    for (bool epilogue : {false,true}) {
        const auto schema = epilogue ? kEpilogue : kProjection;
        habana::custom_op::registerUserCustomOp(schema, kGuid, [epilogue](const at::Stack& stack) {
            const auto x = meta(stack,epilogue).at(0);
            return habana::PartialOutputMetaDataVector{{x.dtype,x.shape}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [epilogue](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<Projection>(device,dtype,epilogue);
        });
    }
    habana::custom_op::registerUserCustomOp(kNormProjection, kNormQuant, [](const at::Stack& stack) {
        const auto x = norm_meta(stack).at(0);
        return habana::PartialOutputMetaDataVector{{x.dtype,x.shape}};
    }, nullptr);
    habana::KernelRegistry().add(kNormProjection, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Projection>(device,dtype,false,true);
    });
    return true;
}();
template<bool Meta, bool Epilogue>
at::Tensor run(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s,
               const at::Tensor& pos, const at::Tensor& table) {
    const auto output = meta({x,w,s,pos,table},Epilogue);
    if (Meta) return at::empty(output.at(0).shape,x.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Epilogue ? kEpilogue : kProjection);
    return descriptor.execute({x,w,s,pos,table}).at(0);
}
template<bool Meta>
at::Tensor norm_run(const at::Tensor& x, const at::Tensor& norm, const at::Tensor& w,
                    const at::Tensor& s, const at::Tensor& pos, const at::Tensor& table,
                    double epsilon) {
    const auto output = norm_meta({x,norm,w,s,pos,table,epsilon});
    if (Meta) return at::empty(output.at(0).shape,x.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kNormProjection);
    return descriptor.execute({x,norm,w,s,pos,table,epsilon}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_q_projection_rope_gaudi2(Tensor input, Tensor weight, Tensor channel_scale, Tensor position, Tensor table) -> Tensor");
    m.def("custom_deepseek_v41_q_scale_rope_gaudi2(Tensor product, Tensor channel_scale, Tensor activation_scale, Tensor position, Tensor table) -> Tensor");
    m.def("custom_deepseek_v41_q_norm_projection_rope_gaudi2(Tensor input, Tensor norm_weight, Tensor weight, Tensor channel_scale, Tensor position, Tensor table, float epsilon) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_q_projection_rope_gaudi2",run<false,false>);
    m.impl("custom_deepseek_v41_q_scale_rope_gaudi2",run<false,true>);
    m.impl("custom_deepseek_v41_q_norm_projection_rope_gaudi2",norm_run<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_q_projection_rope_gaudi2",run<true,false>);
    m.impl("custom_deepseek_v41_q_scale_rope_gaudi2",run<true,true>);
    m.impl("custom_deepseek_v41_q_norm_projection_rope_gaudi2",norm_run<true>);
}
