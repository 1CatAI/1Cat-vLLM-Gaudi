// SPDX-License-Identifier: Apache-2.0
// Explicit operand/output contract for qualification of BF16-valued projections.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"

namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_bf16_linear_f32_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto input = stack.at(0).toTensor(), weight = stack.at(1).toTensor();
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
                input.dim() == 2 && weight.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 8192 &&
                input.size(1) == 5120 && weight.size(1) == 5120 && weight.size(0) > 0 &&
                weight.size(0) <= 64640 && input.is_contiguous() && weight.is_contiguous() &&
                input.device() == weight.device() && !input.requires_grad() && !weight.requires_grad(),
                "V4.1 BF16 operands require contiguous inference [T,5120], [N,5120]");
    return {{at::kFloat, {input.size(0), weight.size(0)}}};
}

class Linear final : public habana::OpBackend {
public:
    Linear(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_bf16_linear_f32"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto meta = metadata(stack);
        synGEMMParams params{false, true};
        auto result = BuildNode(this, graph, {"gemm", {syn_in(0), syn_in(1)},
            {{meta.at(0).shape, at::kFloat, 0}}, &params, sizeof(params)});
        syn_out(0) = std::move(result.at(0));
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kSchema, "gemm", [](const at::Stack& stack) {
        const auto meta = metadata(stack);
        return habana::PartialOutputMetaDataVector{{at::kFloat, meta.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(kSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Linear>(device, dtype);
    });
    return true;
}();

template<bool Meta> at::Tensor run(const at::Tensor& input, const at::Tensor& weight) {
    const auto meta = metadata({input, weight});
    if (Meta) return at::empty(meta.at(0).shape, input.options().dtype(at::kFloat));
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    return descriptor.execute({input, weight}).at(0);
}

}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_bf16_linear_f32_gaudi2(Tensor input, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_bf16_linear_f32_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_bf16_linear_f32_gaudi2", run<true>);
}
