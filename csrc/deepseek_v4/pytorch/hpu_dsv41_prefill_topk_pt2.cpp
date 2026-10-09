// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#ifndef DSV41_PREFILL_TOPK_OP
#define DSV41_PREFILL_TOPK_OP "custom_deepseek_v41_prefill_score_topk_gaudi2"
#define DSV41_PREFILL_THRESHOLD_GUID "custom_deepseek_v41_prefill_topk_threshold_gaudi2"
#define DSV41_PREFILL_EMIT_GUID "custom_deepseek_v41_prefill_topk_emit_gaudi2"
#define DSV41_PREFILL_MAX_COLUMNS 4096
#define DSV41_PREFILL_MAX_ROWS 16384
#define DSV41_PREFILL_MAX_WIDTH 2048
#endif
#ifndef DSV41_PREFILL_BITMAP
#define DSV41_PREFILL_BITMAP 0
#endif
namespace {
constexpr auto schema = "custom_op::" DSV41_PREFILL_TOPK_OP;
constexpr auto threshold_guid = DSV41_PREFILL_THRESHOLD_GUID;
constexpr auto emit_guid = DSV41_PREFILL_EMIT_GUID;
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    TORCH_CHECK(stack.size() == 2, "Prefill score top-k takes scores and width");
    const auto score = stack[0].toTensor();
    const auto width = stack[1].toInt();
    TORCH_CHECK(score.scalar_type() == at::kFloat && score.dim() == 2 && score.is_contiguous() &&
                score.storage_offset() == 0 && !score.requires_grad() &&
                score.size(0) >= 1 && score.size(0) <= DSV41_PREFILL_MAX_ROWS &&
                score.size(1) >= 64 && score.size(1) <= DSV41_PREFILL_MAX_COLUMNS && score.size(1) % 64 == 0 &&
                width >= 1 && width <= score.size(1) && width <= DSV41_PREFILL_MAX_WIDTH,
                "Prefill top-k requires contiguous BF16-valued F32[T1..16384,N64..4096] and K1..2048");
    return {{at::kFloat, {score.size(0), width}}, {at::kInt, {score.size(0), width}}};
}
class Topk final : public habana::OpBackend {
public:
    Topk(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_score_topk"), dtype, {0,1}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto outputs = metadata(stack);
        const auto scores = stack[0].toTensor();
        int params[] = {static_cast<int>(scores.size(1)), static_cast<int>(stack[1].toInt())};
        const int metadata_width = 18 + (DSV41_PREFILL_BITMAP ? (params[0]+127)/128*8 : 0);
        auto counts = BuildNode(this, graph, {threshold_guid, {syn_in(0)},
            {{{scores.size(0), metadata_width}, at::kInt}}, params, sizeof(params)});
        auto selected = BuildNode(this, graph, {emit_guid, {syn_in(0), counts[0].get()},
            {{outputs[0].shape, at::kFloat, 0}, {outputs[1].shape, at::kInt, 1}}, params, sizeof(params)});
        syn_out(0) = std::move(selected[0]);
        syn_out(1) = std::move(selected[1]);
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, threshold_guid, [](const at::Stack& s) {
        const auto m = metadata(s);
        return habana::PartialOutputMetaDataVector{{m[0].dtype, m[0].shape}, {m[1].dtype, m[1].shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId d, c10::ScalarType t) { return std::make_shared<Topk>(d,t); });
    return true;
}();
template<bool Fake> std::tuple<at::Tensor,at::Tensor> run(const at::Tensor& scores, int64_t width) {
    const at::Stack stack{scores,width};
    const auto m = metadata(stack);
    if constexpr (Fake) return {at::empty(m[0].shape,scores.options()),
                               at::empty(m[1].shape,scores.options().dtype(at::kInt))};
    TORCH_CHECK(registered && scores.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result = op.execute(stack);
    return {result[0],result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def(DSV41_PREFILL_TOPK_OP "(Tensor scores, int width) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl(DSV41_PREFILL_TOPK_OP,run<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl(DSV41_PREFILL_TOPK_OP,run<true>); }
