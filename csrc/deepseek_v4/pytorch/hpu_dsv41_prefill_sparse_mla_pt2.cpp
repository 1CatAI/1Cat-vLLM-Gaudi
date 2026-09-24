// SPDX-License-Identifier: Apache-2.0
// Experimental Gaudi2 sparse-MLA producer/consumer chain. Query tiles are
// bounded independently of the total prompt length. The compiler decides
// whether each selected-KV slice fits in SRAM; placement must be inspected
// before making a memory-traffic claim.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include "backend/helpers/create_tensor.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_prefill_sparse_mla_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    TORCH_CHECK(stack.size() == 6, "Sparse MLA expects query, cache, IDs, sink, scale and lengths");
    const auto q = stack[0].toTensor(), cache = stack[1].toTensor(), ids = stack[2].toTensor();
    const auto sink = stack[3].toTensor(), scale = stack[4].toTensor(), lengths = stack[5].toTensor();
    TORCH_CHECK(q.dim() == 3 && q.scalar_type() == at::kBFloat16 && q.size(0) >= 1 && q.size(0) <= 32 &&
                q.size(1) >= 1 && q.size(1) <= 64 && q.size(2) == 512 &&
                cache.dim() == 2 && cache.scalar_type() == at::kBFloat16 && cache.size(0) >= 1 &&
                cache.size(0) <= 131072 && cache.size(1) == 512 &&
                ids.dim() == 2 && ids.scalar_type() == at::kInt && ids.size(0) == q.size(0) &&
                ids.size(1) >= 64 && ids.size(1) <= 640 && ids.size(1) % 64 == 0 &&
                sink.scalar_type() == at::kFloat && sink.sizes() == at::IntArrayRef({q.size(1)}) &&
                scale.scalar_type() == at::kFloat && scale.sizes() == at::IntArrayRef({1}) &&
                lengths.scalar_type() == at::kInt && lengths.sizes() == at::IntArrayRef({q.size(0)}),
                "Sparse MLA requires bounded BF16 Q/KV, I32 IDs/lengths, FP32 sink/scale");
    for (int i = 0; i < 6; ++i) {
        const auto t = stack[i].toTensor();
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Sparse MLA inputs must be contiguous inference tensors on one device");
    }
    return {{at::kBFloat16, q.sizes().vec()}};
}

class PrefillSparseMla final : public habana::OpBackend {
public:
    PrefillSparseMla(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_sparse_mla"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }

    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto out = metadata(stack)[0];
        const int64_t tokens = out.shape[0], heads = out.shape[1];
        const int64_t columns = stack[2].toTensor().size(1);
        std::vector<synapse_helpers::tensor> gathered;
        if (!graph.is_dry_run() && !isOutputInfMode()) {
            const int device = SynInput(1).ref().device_id();
            gathered.emplace_back(habana_helpers::create_tensor(
                {tokens, columns, 512}, {columns * 512, 512, 1}, graph,
                false, false, device, at::kBFloat16));
            gathered.emplace_back(habana_helpers::create_tensor(
                {tokens, columns}, {columns, 1}, graph, false, false, device, at::kFloat));
            graph.add_node({syn_in(1), syn_in(2), syn_in(5)},
                           {gathered[0].get(), gathered[1].get()}, nullptr, 0,
                           "custom_deepseek_v41_prefill_sparse_kv_bf16_gaudi2",
                           nullptr, nullptr, nullptr, deterministic, getContextHints());
        } else {
            gathered = BuildNode(this, graph, {"custom_deepseek_v41_prefill_sparse_kv_bf16_gaudi2",
                {syn_in(1), syn_in(2), syn_in(5)},
                {{{tokens, columns, 512}, at::kBFloat16}, {{tokens, columns}, at::kFloat}}});
        }
        synGEMMParams qk{false, true}, pv{false, false};
        auto scores = BuildNode(this, graph, {"batch_gemm", {syn_in(0), gathered[0].get()},
            {{{tokens, heads, columns}, at::kFloat}}, &qk, sizeof(qk)});
        auto exponential = BuildNode(this, graph, {"custom_deepseek_v41_prefill_sparse_exp_bf16_gaudi2",
            {scores[0].get(), gathered[1].get(), syn_in(3), syn_in(4)},
            {{{tokens, heads, columns}, at::kBFloat16}, {{tokens, heads, 1}, at::kFloat}}});
        auto product = BuildNode(this, graph, {"batch_gemm", {exponential[0].get(), gathered[0].get()},
            {{{tokens, heads, 512}, at::kFloat}}, &pv, sizeof(pv)});
        syn_out(0) = std::move(BuildNode(this, graph, {"custom_deepseek_v41_prefill_sparse_normalize_bf16_gaudi2",
            {product[0].get(), exponential[1].get()}, {{out.shape, at::kBFloat16, 0}}})[0]);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "batch_gemm", [](const at::Stack& stack) {
        const auto output = metadata(stack)[0];
        return habana::PartialOutputMetaDataVector{{output.dtype, output.shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PrefillSparseMla>(device, dtype);
    });
    return true;
}();

template<bool Fake>
at::Tensor execute(const at::Tensor& q, const at::Tensor& cache, const at::Tensor& ids,
                   const at::Tensor& sink, const at::Tensor& scale, const at::Tensor& lengths) {
    const at::Stack stack{q, cache, ids, sink, scale, lengths};
    const auto output = metadata(stack)[0];
    if constexpr (Fake) return at::empty(output.shape, q.options());
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack)[0];
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_sparse_mla_gaudi2(Tensor q, Tensor cache, Tensor ids, Tensor sink, Tensor scale, Tensor lengths) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_sparse_mla_gaudi2", execute<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_sparse_mla_gaudi2", execute<true>);
}
