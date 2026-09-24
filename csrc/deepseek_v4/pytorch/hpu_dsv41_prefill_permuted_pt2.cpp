// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kGuid = "custom_deepseek_v41_prefill_permuted_bf16_gaudi2";
constexpr auto kSchema = "custom_op::custom_deepseek_v41_prefill_permuted_bf16_gaudi2";

std::vector<int64_t> output_shape(const at::Stack& stack) {
    const auto& ids = stack.at(0).toTensor();
    const auto& q = stack.at(1).toTensor();
    const auto& scales = stack.at(2).toTensor();
    const auto& lookup = stack.at(3).toTensor();
    for (const auto& input : {ids, q, scales, lookup})
        TORCH_CHECK(input.device() == q.device() && input.is_contiguous() && !input.requires_grad(),
                    "Permuted prefill weights require contiguous inference tensors on one device");
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 2 && ids.size(0) == 1 &&
                ids.size(1) > 0 && ids.size(1) <= 32, "Expected one to 32 expert descriptors");
    TORCH_CHECK(q.scalar_type() == at::kShort && q.dim() == 3 && q.size(0) > 0 && q.size(0) <= 384 &&
                q.size(1) > 0 && q.size(1) <= 20 && q.size(2) > 0 && q.size(2) <= 327680 &&
                q.size(2) % 8192 == 0, "Expected prepared N256 expert weights with K divisible by 128");
    TORCH_CHECK(scales.scalar_type() == at::kShort && scales.dim() == 3 && scales.size(0) == q.size(0) &&
                scales.size(1) == q.size(1) && scales.size(2) * 8 == q.size(2),
                "Expected prepared group-32 exponent planes");
    TORCH_CHECK(lookup.scalar_type() == at::kBFloat16 && lookup.sizes() == at::IntArrayRef({128}),
                "Expected the prepared MXFP4 lookup table");
    return {ids.numel(), q.size(2) / 64, q.size(1) * 256};
}

class PermutedWeights final : public habana::OpBackend {
 public:
    PermutedWeights(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_permuted"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16, output_shape(stack)}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        syn_out(0) = std::move(BuildNode(this, graph, {kGuid,
            {syn_in(0), syn_in(1), syn_in(2), syn_in(3)}, {{output_shape(stack), at::kBFloat16, 0}}}).at(0));
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(kSchema, kGuid, [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kBFloat16, output_shape(stack)}};
    }, nullptr);
    habana::KernelRegistry().add(kSchema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PermutedWeights>(device, dtype);
    });
    return true;
}();

template<bool Meta> at::Tensor permuted_weights(const at::Tensor& ids, const at::Tensor& q,
        const at::Tensor& scales, const at::Tensor& lookup) {
    const at::Stack stack{ids, q, scales, lookup};
    const auto shape = output_shape(stack);
    if (Meta) return at::empty(shape, q.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
    return op.execute(stack).at(0);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_permuted_bf16_gaudi2("
          "Tensor ids, Tensor q16, Tensor s16, Tensor lookup) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_permuted_bf16_gaudi2", permuted_weights<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_permuted_bf16_gaudi2", permuted_weights<true>);
}
