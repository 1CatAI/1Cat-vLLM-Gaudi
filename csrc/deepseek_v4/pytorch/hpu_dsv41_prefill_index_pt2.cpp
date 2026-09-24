// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"

namespace {
constexpr auto score_schema = "custom_op::custom_deepseek_v41_prefill_index_scores_gaudi2";
constexpr auto reduce_schema = "custom_op::custom_deepseek_v41_prefill_index_reduce_gaudi2";
constexpr auto guid = "custom_deepseek_v41_prefill_index_reduce_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& stack, bool reduce_only) {
    TORCH_CHECK(stack.size() == (reduce_only ? 5 : 6), "Invalid prefill index arguments");
    const auto x = stack[0].toTensor(), weights = stack[1].toTensor();
    const auto positions = stack[reduce_only ? 2 : 3].toTensor();
    const auto rows = stack[reduce_only ? 3 : 4].toTensor();
    const auto ratio = stack.back().toInt();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 3 &&
                x.size(0) >= 1 && x.size(0) <= 8192 && x.size(1) == 32,
                "Prefill index requires BF16 [T,32,K] with T1..8192");
    const auto tokens = x.size(0), columns = rows.numel();
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16 &&
                weights.sizes() == at::IntArrayRef({tokens, 32}) &&
                positions.scalar_type() == at::kInt && positions.sizes() == at::IntArrayRef({tokens}) &&
                rows.scalar_type() == at::kInt && rows.dim() == 1 &&
                columns >= 128 && columns <= 2048 && columns % 128 == 0 && (ratio == 1 || ratio == 2),
                "Invalid prefill index weights, positions or source tile");
    if (reduce_only) {
        TORCH_CHECK(x.size(2) == columns, "Index reduction requires one score per source row");
    } else {
        const auto keys = stack[2].toTensor();
        TORCH_CHECK(x.size(2) == 128 && keys.scalar_type() == at::kBFloat16 &&
                    keys.sizes() == at::IntArrayRef({columns, 128}),
                    "Prefill index scoring requires full K128");
    }
    for (size_t i = 0; i + 1 < stack.size(); ++i) {
        const auto tensor = stack[i].toTensor();
        TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad() && tensor.device() == x.device(),
                    "Prefill index tensors must be contiguous inference tensors on one device");
    }
    return {{at::kFloat, {tokens, columns}}};
}

class IndexScore final : public habana::OpBackend {
    bool reduce_only_;
public:
    IndexScore(int device, c10::ScalarType dtype, bool reduce_only)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_index"), dtype, {0}, {}, {}, false),
          reduce_only_(reduce_only) {
        SetOutputMetaFn([reduce_only](const at::Stack& s) { return metadata(s, reduce_only); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = metadata(stack, reduce_only_)[0];
        int ratio = stack.back().toInt();
        if (reduce_only_) {
            auto result = BuildNode(this, graph, {guid, {syn_in(0), syn_in(1), syn_in(2), syn_in(3)},
                {{output.shape, at::kFloat, 0}}, &ratio, sizeof(ratio)});
            syn_out(0) = std::move(result[0]);
            return;
        }
        const auto tokens = output.shape[0], columns = output.shape[1];
        auto query = ReshapeHelper(graph, syn_in(0), {tokens * 32, 128}, at::kBFloat16);
        synGEMMParams params{false, true};
        auto product = BuildNode(this, graph, {"gemm", {query.get(), syn_in(2)},
            {{{tokens * 32, columns}, at::kBFloat16}}, &params, sizeof(params)});
        auto shaped = ReshapeHelper(graph, product[0].get(), {tokens, 32, columns}, at::kBFloat16);
        auto result = BuildNode(this, graph, {guid, {shaped.get(), syn_in(1), syn_in(3), syn_in(4)},
            {{output.shape, at::kFloat, 0}}, &ratio, sizeof(ratio)});
        syn_out(0) = std::move(result[0]);
    }
};

const bool registered = [] {
    for (bool reduce_only : {false, true}) {
        const auto schema = reduce_only ? reduce_schema : score_schema;
        habana::custom_op::registerUserCustomOp(schema, guid, [reduce_only](const at::Stack& s) {
            const auto output = metadata(s, reduce_only)[0];
            return habana::PartialOutputMetaDataVector{{output.dtype, output.shape}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [reduce_only](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<IndexScore>(device, dtype, reduce_only);
        });
    }
    return true;
}();

template<bool Fake, bool ReduceOnly> at::Tensor execute(const at::Stack& stack) {
    const auto output = metadata(stack, ReduceOnly)[0];
    const auto x = stack[0].toTensor();
    if constexpr (Fake) return at::empty(output.shape, x.options().dtype(at::kFloat));
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto op = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        ReduceOnly ? reduce_schema : score_schema);
    return op.execute(stack)[0];
}
template<bool Fake> at::Tensor score(const at::Tensor& q, const at::Tensor& w, const at::Tensor& k,
                                   const at::Tensor& p, const at::Tensor& r, int64_t ratio) {
    return execute<Fake, false>({q, w, k, p, r, ratio});
}
template<bool Fake> at::Tensor reduce(const at::Tensor& s, const at::Tensor& w, const at::Tensor& p,
                                    const at::Tensor& r, int64_t ratio) {
    return execute<Fake, true>({s, w, p, r, ratio});
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_index_scores_gaudi2(Tensor query, Tensor weights, Tensor keys, Tensor positions, Tensor rows, int ratio) -> Tensor");
    m.def("custom_deepseek_v41_prefill_index_reduce_gaudi2(Tensor scores, Tensor weights, Tensor positions, Tensor rows, int ratio) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_index_scores_gaudi2", score<false>);
    m.impl("custom_deepseek_v41_prefill_index_reduce_gaudi2", reduce<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_index_scores_gaudi2", score<true>);
    m.impl("custom_deepseek_v41_prefill_index_reduce_gaudi2", reduce<true>);
}
