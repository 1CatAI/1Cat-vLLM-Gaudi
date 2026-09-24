// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_prefill_bmm_f32_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto a = stack[0].toTensor(), b = stack[1].toTensor();
    const bool transpose_b = stack[2].toBool();
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) &&
                a.size(0) >= 1 && a.size(0) <= 2048 && a.size(1) >= 1 && a.size(1) <= 128 &&
                a.size(2) >= 1 && a.size(2) <= 640 && b.size(1) >= 1 && b.size(1) <= 640 &&
                b.size(2) >= 1 && b.size(2) <= 640 && a.size(2) == b.size(transpose_b ? 2 : 1),
                "Prefill BMM requires bounded matching three-dimensional attention operands");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "Prefill BMM requires BF16 operands and returns FP32 accumulators");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a.device() == b.device() &&
                !a.requires_grad() && !b.requires_grad(),
                "Prefill BMM requires contiguous inference operands on one device");
    return {{at::kFloat, {a.size(0), a.size(1), b.size(transpose_b ? 1 : 2)}}};
}

class PrefillBmm final : public habana::OpBackend {
public:
    PrefillBmm(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_bmm"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = metadata(stack)[0];
        synGEMMParams params{false, stack[2].toBool()};
        auto product = BuildNode(this, graph, {"batch_gemm", {syn_in(0), syn_in(1)},
            {{output.shape, at::kFloat, 0}}, &params, sizeof(params)});
        syn_out(0) = std::move(product[0]);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "batch_gemm", [](const at::Stack& stack) {
        const auto output = metadata(stack)[0];
        return habana::PartialOutputMetaDataVector{{output.dtype, output.shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PrefillBmm>(device, dtype);
    });
    return true;
}();

template<bool Fake> at::Tensor run(const at::Tensor& a, const at::Tensor& b, bool transpose_b) {
    const at::Stack stack{a, b, transpose_b};
    const auto output = metadata(stack)[0];
    if constexpr (Fake) return at::empty(output.shape, a.options().dtype(at::kFloat));
    TORCH_CHECK(registered && a.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack)[0];
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_bmm_f32_gaudi2(Tensor a, Tensor b, bool transpose_b) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_bmm_f32_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_bmm_f32_gaudi2", run<true>);
}
