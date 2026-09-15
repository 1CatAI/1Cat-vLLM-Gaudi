// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
namespace {
constexpr auto kProjection = "custom_op::custom_deepseek_v41_q_projection_rope_gaudi2";
constexpr auto kEpilogue = "custom_op::custom_deepseek_v41_q_scale_rope_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_q_scale_rope_gaudi2";
habana::OutputMetaDataVector meta(const at::Stack& stack, bool epilogue) {
    TORCH_CHECK(stack.size() == 5, "Q projection/RoPE requires five tensors");
    const auto x = stack.at(0).toTensor();
    for (const auto& item : stack) {
        const auto t = item.toTensor();
        TORCH_CHECK(t.is_contiguous() && !t.requires_grad() && t.device() == x.device(),
                    "Q projection/RoPE requires contiguous matching inference tensors");
    }
    const auto w = stack.at(1).toTensor(), s = stack.at(2).toTensor();
    if (epilogue) {
        TORCH_CHECK(x.scalar_type() == at::kFloat && x.sizes() == at::IntArrayRef({1,16384}) &&
                    w.scalar_type() == at::kFloat && w.sizes() == x.sizes() &&
                    s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({1,1}),
                    "Q epilogue requires F32 product/channel scales [1,16384] and activation scale [1,1]");
    } else {
        TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.sizes() == at::IntArrayRef({1,1280}) &&
                    w.scalar_type() == at::ScalarType::Float8_e4m3fn && w.sizes() == at::IntArrayRef({16384,1280}) &&
                    s.scalar_type() == at::kFloat && s.sizes() == at::IntArrayRef({1,16384}),
                    "Q projection requires BF16 [1,1280], prepared FP8 [16384,1280], F32 scales [1,16384]");
    }
    const auto pos = stack.at(3).toTensor(), table = stack.at(4).toTensor();
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.sizes() == at::IntArrayRef({1}) &&
                table.scalar_type() == at::kFloat && table.dim() == 2 && table.size(0) > 0 &&
                table.size(0) <= 1048576 && table.size(1) == 64,
                "Q RoPE requires I32 position [1] and F32 cos/sin [1..1048576,64]");
    return {{at::kBFloat16, {1,16384}}};
}
class Projection final : public habana::OpBackend {
    bool epilogue_;
public:
    Projection(int device, c10::ScalarType dtype, bool epilogue)
        : OpBackend(device, NO_TPC + std::string("dsv41_q_projection_rope"), dtype, {0}, {}, {}, false),
          epilogue_(epilogue) {
        SetOutputMetaFn([epilogue](const at::Stack& s) { return meta(s, epilogue); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = meta(stack, epilogue_);
        if (epilogue_) {
            auto node = BuildNode(this, graph, {kGuid, {syn_in(0),syn_in(1),syn_in(2),syn_in(3),syn_in(4)},
                {{output.at(0).shape, at::kBFloat16, 0}}});
            syn_out(0) = std::move(node.at(0)); return;
        }
        auto q = BuildNode(this, graph, {"custom_deepseek_v41_dense_quant_gaudi2", {syn_in(0)},
            {{{1,1280}, at::ScalarType::Float8_e4m3fn}, {{1,1}, at::kFloat}}});
        synGEMMParams params{false,true};
        auto product = BuildNode(this, graph, {"gemm", {q.at(0).get(),syn_in(1)},
            {{output.at(0).shape, at::kFloat}}, &params, sizeof(params)});
        auto node = BuildNode(this, graph, {kGuid,
            {product.at(0).get(),syn_in(2),q.at(1).get(),syn_in(3),syn_in(4)},
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
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_q_projection_rope_gaudi2(Tensor input, Tensor weight, Tensor channel_scale, Tensor position, Tensor table) -> Tensor");
    m.def("custom_deepseek_v41_q_scale_rope_gaudi2(Tensor product, Tensor channel_scale, Tensor activation_scale, Tensor position, Tensor table) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_q_projection_rope_gaudi2",run<false,false>);
    m.impl("custom_deepseek_v41_q_scale_rope_gaudi2",run<false,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_q_projection_rope_gaudi2",run<true,false>);
    m.impl("custom_deepseek_v41_q_scale_rope_gaudi2",run<true,true>);
}
