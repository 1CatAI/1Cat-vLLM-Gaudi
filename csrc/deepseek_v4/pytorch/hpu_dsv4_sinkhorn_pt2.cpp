// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* schema = "custom_op::custom_deepseek_v4_sinkhorn4_gaudi2";
void validate(const at::Tensor& input) {
    TORCH_CHECK(input.scalar_type() == at::kFloat && input.dim() == 3 &&
                input.size(0) >= 1 && input.size(0) <= 8192 && input.size(1) == 4 && input.size(2) == 4 &&
                input.is_contiguous(), "V4 Sinkhorn requires contiguous FP32 [T,4,4], T in [1,8192]");
}
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto input = stack.at(0).toTensor();
    validate(input);
    return {{at::kFloat, input.sizes().vec()}};
}

class Sinkhorn final : public habana::OpBackend {
public:
    Sinkhorn(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv4_sinkhorn4"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto meta = metadata(stack);
        const std::vector<int64_t> flat_shape{1, stack.at(0).toTensor().numel()};
        auto input = ReshapeHelper(graph, syn_in(0), flat_shape, at::kFloat);
        auto output = BuildNode(this, graph, {"custom_deepseek_v4_sinkhorn4_gaudi2", {input.get()},
                                             {{flat_shape, at::kFloat}}});
        syn_out(0) = ReshapeHelper(graph, output.at(0).get(), meta.at(0).shape, at::kFloat, 0);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "custom_deepseek_v4_sinkhorn4_gaudi2",
        [](const at::Stack& stack) {
            const auto meta = metadata(stack);
            return habana::PartialOutputMetaDataVector{{at::kFloat, meta.at(0).shape}};
        }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<Sinkhorn>(device, dtype);
    });
    return true;
}();

at::Tensor run(const at::Tensor& input) {
    validate(input);
    TORCH_CHECK(registered && input.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto outputs = descriptor.execute({input});
    TORCH_CHECK(outputs.size() == 1);
    return outputs.at(0);
}
at::Tensor meta(const at::Tensor& input) {
    validate(input);
    return at::empty_like(input);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v4_sinkhorn4_gaudi2(Tensor input) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v4_sinkhorn4_gaudi2", run);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v4_sinkhorn4_gaudi2", meta);
}
