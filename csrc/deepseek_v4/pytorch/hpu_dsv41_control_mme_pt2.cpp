// SPDX-License-Identifier: Apache-2.0
// Explicit control-projection operands and FP32 accumulation/output.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"
#ifndef DSV41_MME_SCHEMA
#define DSV41_MME_SCHEMA "custom_deepseek_v41_control_mme_f32_gaudi2"
#endif
#ifndef DSV41_MME_VARIABLE_SHAPE
#define DSV41_MME_VARIABLE_SHAPE 0
#endif

namespace {
constexpr auto schema = "custom_op::" DSV41_MME_SCHEMA;
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto x = stack.at(0).toTensor(), w = stack.at(1).toTensor();
    TORCH_CHECK((x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16) &&
                w.scalar_type() == x.scalar_type() && x.dim() == 2 && w.dim() == 2 &&
                x.size(0) >= 1 && x.size(0) <= 2048 &&
                (DSV41_MME_VARIABLE_SHAPE ? (x.size(1)==512 && w.size(1)==512 &&
                                            w.size(0)>0 && w.size(0)<=6144)
                                         : (x.size(1)==20480 && w.size(1)==20480 &&
                                            (w.size(0)==24 || w.size(0)==48))) &&
                x.is_contiguous() && w.is_contiguous() &&
                x.device().type() == w.device().type() &&
                (!x.device().has_index() || !w.device().has_index() || x.device().index() == w.device().index()) &&
                !x.requires_grad() && !w.requires_grad(), "Invalid control MME operand contract");
    return {{at::kFloat, {x.size(0), w.size(0)}}};
}
class ControlMME final : public habana::OpBackend {
public:
    ControlMME(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string("dsv41_control_mme"), type, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto meta = metadata(stack);
        synGEMMParams parameters{false, true};
        auto output = BuildNode(this, graph, {"gemm", {syn_in(0), syn_in(1)},
            {{meta.at(0).shape, at::kFloat, 0}}, &parameters, sizeof(parameters)});
        syn_out(0) = std::move(output.at(0));
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "gemm", [](const at::Stack& stack) {
        const auto meta = metadata(stack);
        return habana::PartialOutputMetaDataVector{{at::kFloat, meta.at(0).shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<ControlMME>(device, type);
    });
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x, const at::Tensor& w) {
    const auto meta = metadata({x, w});
    if (Meta) return at::empty(meta.at(0).shape, x.options().dtype(at::kFloat));
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({x, w}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def(DSV41_MME_SCHEMA "(Tensor input, Tensor weight) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl(DSV41_MME_SCHEMA, run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl(DSV41_MME_SCHEMA, run<true>);
}
