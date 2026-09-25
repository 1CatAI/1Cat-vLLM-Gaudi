// SPDX-License-Identifier: Apache-2.0
// One independently prepared scoring tile. Row validity and final selection
// remain mandatory consumers outside this optional recipe.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "perf_lib_layer_params.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_reindex_tile_gaudi2";
constexpr auto tiled_schema = "custom_op::custom_deepseek_v41_reindex_tiled_gaudi2";
template<bool Tiled> constexpr const char* key_guid() {
    return Tiled ? "custom_deepseek_v41_index_keys_tiled_gaudi2" : "custom_deepseek_v41_index_keys_gaudi2";
}
std::vector<int64_t> shape(const at::Stack& s) {
    TORCH_CHECK(s.size() == 7, "Reindex tile expects five tensors, ratio and ordinal");
    const auto q = s.at(0).toTensor(), w = s.at(1).toTensor(), cache = s.at(2).toTensor();
    const auto pages = s.at(3).toTensor(), rows = s.at(4).toTensor();
    const auto ratio = s.at(5).toInt(), ordinal = s.at(6).toInt();
    const at::ScalarType types[] = {at::kBFloat16, at::kBFloat16, at::kByte, at::kInt, at::kInt};
    for (size_t i = 0; i < 5; ++i) {
        const auto t = s.at(i).toTensor();
        TORCH_CHECK(t.device() == q.device() && t.is_contiguous() && !t.requires_grad() &&
                    t.scalar_type() == types[i], "Reindex tile tensor contract differs at ", i);
    }
    TORCH_CHECK(q.dim() == 3 && q.size(0) >= 1 && q.size(0) <= 64 && q.size(1) == 32 && q.size(2) == 128 &&
                w.sizes() == at::IntArrayRef({q.size(0),32}) &&
                cache.dim() == 2 && cache.size(0) > 0 && cache.size(0) <= 1048704 && cache.size(1) == 68 &&
                pages.dim() == 2 && pages.size(0) == q.size(0) && pages.size(1) > 0 && pages.size(1) <= 8192 &&
                rows.sizes() == at::IntArrayRef({q.size(0),ordinal == 8 ? 16384 : 2048}) &&
                (ratio == 1 || ratio == 2) && ordinal >= 0 && ordinal <= 8,
                "Reindex tile shape, compression or ordinal differs");
    return {q.size(0),rows.size(1)};
}
template<bool Tiled> class ReindexTile final : public habana::OpBackend {
    using Tensor = synapse_helpers::tensor;
    Tensor tile(synapse_helpers::graph& graph, synTensor rows, int64_t batch, int ratio) {
        auto keys = BuildNode(this, graph, {key_guid<Tiled>(),
            {syn_in(2),syn_in(3),rows}, {{{batch,2048,128},at::kBFloat16}}, &ratio,sizeof(ratio)});
        synGEMMParams params{false,true};
        auto dots = BuildNode(this, graph, {"batch_gemm", {syn_in(0),keys.at(0).get()},
            {{{batch,32,2048},at::kBFloat16}}, &params,sizeof(params)});
        auto flat = ReshapeHelper(graph,dots.at(0).get(),{1,batch*32*2048},at::kBFloat16);
        auto rounded = BuildNode(this,graph,{"custom_deepseek_v41_bf16_identity_gaudi2",{flat.get()},
            {{{1,batch*32*2048},at::kBFloat16}}});
        auto ordered = ReshapeHelper(graph,rounded.at(0).get(),{batch,32,2048},at::kBFloat16);
        auto scores = BuildNode(this, graph, {"custom_deepseek_v41_index_reduce_gaudi2",
            {ordered.get(),syn_in(1)}, {{{batch,2048},at::kFloat}}});
        return std::move(scores.at(0));
    }
 public:
    ReindexTile(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string(Tiled ? "dsv41_reindex_tiled" : "dsv41_reindex_tile"),
                    dtype, {0}, {}, {}, false) {
        SetOutputMetaFn([](const at::Stack& s) { return habana::OutputMetaDataVector{{at::kFloat,shape(s)}}; });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto out = shape(s);
        int ratio = s.at(5).toInt();
        std::vector<Tensor> scores;
        std::vector<synTensor> inputs;
        for (int first = 0; first < out[1]; first += 2048) {
            synTensor rows = syn_in(4);
            std::vector<Tensor> sliced;
            if (out[1] != 2048) {
                synSliceParams params{};
                for (unsigned i = 0; i < sizeof(params.axes)/sizeof(params.axes[0]); ++i) {
                    params.axes[i] = i; params.steps[i] = 1; params.ends[i] = 1;
                }
                params.starts[0] = first; params.ends[0] = first + 2048; params.ends[1] = out[0];
                sliced = BuildNode(this,graph,{"slice",{rows},{{{out[0],2048},at::kInt}},&params,sizeof(params)});
                rows = sliced.at(0).get();
            }
            scores.push_back(tile(graph,rows,out[0],ratio));
            inputs.push_back(scores.back().get());
        }
        synConcatenateParams concat{};
        concat.axis = 0;
        auto result = BuildNode(this,graph,{"concat",std::move(inputs),{{out,at::kFloat,0}},&concat,sizeof(concat)});
        syn_out(0) = std::move(result.at(0));
    }
};
template<bool Tiled> bool register_tile(const char* name) {
    habana::custom_op::registerUserCustomOp(name, key_guid<Tiled>(), [](const at::Stack& s) {
        return habana::PartialOutputMetaDataVector{{at::kFloat,shape(s)}};
    }, nullptr);
    habana::KernelRegistry().add(name, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<ReindexTile<Tiled>>(device,dtype);
    });
    return true;
}
const bool registered = register_tile<false>(schema) && register_tile<true>(tiled_schema);
template<bool Meta, bool Tiled = false> at::Tensor run(const at::Tensor& q, const at::Tensor& w, const at::Tensor& cache,
    const at::Tensor& pages, const at::Tensor& rows, int64_t ratio, int64_t ordinal) {
    const at::Stack stack{q,w,cache,pages,rows,ratio,ordinal};
    const auto out = shape(stack);
    if (Meta) return at::empty(out,q.options().dtype(at::kFloat));
    TORCH_CHECK(registered && q.device().type() == at::kHPU, "Reindex tile requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Tiled ? tiled_schema : schema);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_reindex_tile_gaudi2(Tensor q, Tensor weights, Tensor cache, Tensor pages, Tensor rows, int ratio, int ordinal) -> Tensor");
    m.def("custom_deepseek_v41_reindex_tiled_gaudi2(Tensor q, Tensor weights, Tensor cache, Tensor pages, Tensor rows, int ratio, int ordinal) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_reindex_tile_gaudi2",run<false>);
    m.impl("custom_deepseek_v41_reindex_tiled_gaudi2",run<false,true>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_reindex_tile_gaudi2",run<true>);
    m.impl("custom_deepseek_v41_reindex_tiled_gaudi2",run<true,true>);
}
