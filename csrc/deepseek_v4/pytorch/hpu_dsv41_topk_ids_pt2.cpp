// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "perf_lib_layer_params.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_topk_ids_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    TORCH_CHECK(stack.size() == 3, "Expected scores, row IDs and selection width");
    const auto scores = stack[0].toTensor(), rows = stack[1].toTensor();
    const auto width = stack[2].toInt();
    TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.dim() == 2 &&
                scores.size(0) >= 1 && scores.size(0) <= 8192 &&
                scores.size(1) >= 1 && scores.size(1) <= 4096,
                "Index TopK requires bounded F32 score rows");
    TORCH_CHECK(rows.scalar_type() == at::kInt && rows.sizes() == scores.sizes() &&
                rows.device() == scores.device() && scores.is_contiguous() && rows.is_contiguous() &&
                !scores.requires_grad() && !rows.requires_grad() && width >= 1 && width <= scores.size(1),
                "Index TopK requires contiguous matching I32 IDs and a valid width");
    const std::vector<int64_t> shape{scores.size(0), width};
    return {{at::kFloat, shape}, {at::kInt, shape}};
}

class TopkIds final : public habana::OpBackend {
public:
    TopkIds(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_topk_ids"), dtype, {0, 1}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = metadata(stack);
        ns_TopkNodeV2::ParamsV4 params{};
        params.bsw = static_cast<unsigned>(stack[2].toInt());
        params.axis = 0;
        params.bottomK = false;
        params.isVcData = false;
        auto result = BuildNode(this, graph, {"topk", {syn_in(0), syn_in(1), nullptr},
            {{output[0].shape, at::kFloat, 0}, {output[1].shape, at::kInt, 1}}, &params, sizeof(params)});
        syn_out(0) = std::move(result[0]);
        syn_out(1) = std::move(result[1]);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "topk", [](const at::Stack& stack) {
        const auto output = metadata(stack);
        return habana::PartialOutputMetaDataVector{{output[0].dtype, output[0].shape},
                                                   {output[1].dtype, output[1].shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<TopkIds>(device, dtype);
    });
    return true;
}();

template<bool Fake>
std::tuple<at::Tensor, at::Tensor> execute(const at::Tensor& scores, const at::Tensor& rows, int64_t width) {
    const at::Stack stack{scores, rows, width};
    const auto output = metadata(stack);
    if constexpr (Fake) {
        return {at::empty(output[0].shape, scores.options()),
                at::empty(output[1].shape, rows.options())};
    }
    TORCH_CHECK(registered && scores.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result = op.execute(stack);
    return {result[0], result[1]};
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_topk_ids_gaudi2(Tensor scores, Tensor rows, int width) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_topk_ids_gaudi2", execute<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_topk_ids_gaudi2", execute<true>);
}
