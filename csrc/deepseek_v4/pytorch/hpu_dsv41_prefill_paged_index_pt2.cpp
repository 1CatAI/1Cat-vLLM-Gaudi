// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
// Physical KV pool rows are independent of the per-request logical context bound.
// DeepGEMM-style paged-MQA score chain adapted to Gaudi2: TPC loads one
// compressed page tile, MME reuses its BF16 keys across all query/head rows,
// and TPC performs the original ordered BF16 head reduction.  The decoded
// key tile is a graph-internal tensor; SRAM placement must be checked on the
// compiled recipe before this is promoted to the serving path.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_prefill_paged_index_scores_gaudi2";
constexpr auto key_guid = "custom_deepseek_v41_index_keys_gaudi2";
constexpr auto reduce_guid = "custom_deepseek_v41_prefill_index_reduce_gaudi2";

habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size() == 8, "Paged index scores require Q, weights, compressed keys, page table, positions, rows, ratio");
    const auto q = s[0].toTensor(), w = s[1].toTensor(), packed = s[2].toTensor();
    const auto pages = s[3].toTensor(), positions = s[4].toTensor(), rows = s[5].toTensor();
    const auto ratio = s[6].toInt();
    TORCH_CHECK(s[7].toInt() == 8 || s[7].toInt() == 16, "Index requires TP2/TP4 head groups");
    TORCH_CHECK(q.dim() == 3 && q.scalar_type() == at::kBFloat16 &&
                q.size(0) >= 1 && q.size(0) <= 16384 && q.size(1) == 32 && q.size(2) == 128 &&
                w.scalar_type() == at::kBFloat16 && w.sizes() == at::IntArrayRef({q.size(0), 32}) &&
                packed.dim() == 2 && packed.scalar_type() == at::kByte && packed.size(1) == 68 &&
                packed.size(0) >= 1 && packed.size(0) <= INT32_MAX &&
                pages.dim() == 1 && pages.scalar_type() == at::kInt &&
                pages.numel() >= 1 && pages.numel() <= 8192 &&
                positions.dim() == 1 && positions.scalar_type() == at::kInt && positions.numel() == q.size(0) &&
                rows.dim() == 1 && rows.scalar_type() == at::kInt &&
                rows.numel() >= 128 && rows.numel() <= 2048 && rows.numel() % 128 == 0 &&
                (ratio == 1 || ratio == 2), "Invalid paged index scoring geometry or dtype");
    for (int i = 0; i < 6; ++i) {
        const auto t = s[i].toTensor();
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && !t.requires_grad(),
                    "Paged index inputs must be contiguous inference tensors on one device");
    }
    return {{at::kFloat, {q.size(0), rows.numel()}}};
}

class PagedIndex final : public habana::OpBackend {
public:
    PagedIndex(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_prefill_paged_index"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto output = metadata(s)[0];
        const int64_t tokens = output.shape[0], columns = output.shape[1];
        int ratio = int(s[6].toInt());
        int reduce_parameters[2] = {ratio, int(s[7].toInt())};
        auto row_matrix = ReshapeHelper(graph, syn_in(5), {1, columns}, at::kInt);
        auto key_tensor = BuildNode(this, graph, {key_guid, {syn_in(2), syn_in(3), row_matrix.get()},
            {{{1, columns, 128}, at::kBFloat16}}, &ratio, sizeof(ratio)});
        auto key_matrix = ReshapeHelper(graph, key_tensor[0].get(), {columns, 128}, at::kBFloat16);
        auto query = ReshapeHelper(graph, syn_in(0), {tokens * 32, 128}, at::kBFloat16);
        synGEMMParams params{false, true};
        auto scores = BuildNode(this, graph, {"gemm", {query.get(), key_matrix.get()},
            {{{tokens * 32, columns}, at::kBFloat16}}, &params, sizeof(params)});
        auto shaped = ReshapeHelper(graph, scores[0].get(), {tokens, 32, columns}, at::kBFloat16);
        syn_out(0) = std::move(BuildNode(this, graph, {reduce_guid,
            {shaped.get(), syn_in(1), syn_in(4), syn_in(5)},
            {{output.shape, at::kFloat, 0}}, reduce_parameters, sizeof(reduce_parameters)})[0]);
    }
};

const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, reduce_guid, [](const at::Stack& s) {
        const auto output = metadata(s)[0];
        return habana::PartialOutputMetaDataVector{{output.dtype, output.shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<PagedIndex>(device, dtype);
    });
    return true;
}();

template<bool Fake>
at::Tensor execute(const at::Tensor& q, const at::Tensor& w, const at::Tensor& packed,
                   const at::Tensor& pages, const at::Tensor& positions, const at::Tensor& rows, int64_t ratio, int64_t local_heads) {
    const at::Stack stack{q, w, packed, pages, positions, rows, ratio, local_heads};
    const auto output = metadata(stack)[0];
    if constexpr (Fake) return at::empty(output.shape, q.options().dtype(at::kFloat));
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack)[0];
}
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_prefill_paged_index_scores_gaudi2(Tensor q, Tensor weights, Tensor packed, Tensor pages, Tensor positions, Tensor rows, int ratio, int local_heads=16) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_prefill_paged_index_scores_gaudi2", execute<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_prefill_paged_index_scores_gaudi2", execute<true>);
}
