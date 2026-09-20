// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cmath>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kSchema =
    "custom_op::custom_deepseek_v41_kv_norm_rope_bf16_gaudi2";
constexpr auto kGuid = "custom_deepseek_v41_kv_norm_rope_bf16_gaudi2";
struct Params { float epsilon; float inverse_width; };

std::vector<int64_t> shape(const at::Tensor& input,
                           const at::Tensor& weight,
                           const at::Tensor& positions,
                           const at::Tensor& phase,
                           double epsilon) {
    const auto device = input.device();
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.dim() == 2 &&
                input.size(0) >= 1 && input.size(0) <= 512 &&
                input.size(1) == 512 && input.is_contiguous() &&
                !input.requires_grad(), "KV norm/RoPE requires BF16 [1..512,512]");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16 &&
                weight.sizes() == at::IntArrayRef({512}) &&
                weight.device() == device && weight.is_contiguous() &&
                !weight.requires_grad(), "KV norm/RoPE requires BF16 weight [512]");
    TORCH_CHECK(positions.scalar_type() == at::kInt && positions.dim() == 1 &&
                positions.size(0) == input.size(0) && positions.device() == device &&
                positions.is_contiguous() && !positions.requires_grad(),
                "KV norm/RoPE requires one I32 position per row");
    TORCH_CHECK(phase.scalar_type() == at::kFloat && phase.dim() == 2 &&
                phase.size(0) >= 1 && phase.size(0) <= 1048576 &&
                phase.size(1) == 64 && phase.device() == device &&
                phase.is_contiguous() && !phase.requires_grad(),
                "KV norm/RoPE requires F32 phase [1..1048576,64]");
    TORCH_CHECK(epsilon > 0 && std::isnormal(static_cast<float>(epsilon)),
                "KV norm/RoPE requires a positive normal epsilon");
    return input.sizes().vec();
}

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(
        kSchema, kGuid,
        [](const at::Stack& stack) {
            const auto sizes = shape(stack.at(0).toTensor(), stack.at(1).toTensor(),
                                     stack.at(2).toTensor(), stack.at(3).toTensor(),
                                     stack.at(4).toDouble());
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, sizes}};
        },
        [](const at::Stack& stack, size_t& size) -> std::shared_ptr<void> {
            size = sizeof(Params);
            return std::make_shared<Params>(Params{
                static_cast<float>(stack.at(4).toDouble()), 1.0f / 512.0f});
        });
    return true;
}();

template<bool Meta>
at::Tensor run(const at::Tensor& input, const at::Tensor& weight,
               const at::Tensor& positions, const at::Tensor& phase,
               double epsilon) {
    const auto sizes = shape(input, weight, positions, phase, epsilon);
    if (Meta) return at::empty(sizes, input.options());
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor =
        habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    return descriptor.execute({input, weight, positions, phase, epsilon}).at(0);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_kv_norm_rope_bf16_gaudi2(Tensor input, Tensor weight, Tensor positions, Tensor phase, float epsilon) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_kv_norm_rope_bf16_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_kv_norm_rope_bf16_gaudi2", run<true>);
}
