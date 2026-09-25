// SPDX-License-Identifier: Apache-2.0
// Reuses the qualified N256 decode, projection rounding, SwiGLU, activation
// quantization and output scaling. Only the matrix M dimension changes.
#include <ATen/ATen.h>
#include <algorithm>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "perf_lib_layer_params.h"

namespace {
constexpr auto kSchema = "custom_op::custom_deepseek_v41_grouped_n256_fp8_gaudi2";
constexpr auto kDecode = "custom_deepseek_v41_expert_n256_fp8_gaudi2";
constexpr auto kSilu = "custom_deepseek_v41_expert_n256_silu_quant_gaudi2";
constexpr auto kReuseSchema = "custom_op::custom_deepseek_v41_reused_n256_fp8_gaudi2";
constexpr auto kReuseDecode = "custom_deepseek_v41_expert_n256_reuse_fp8_gaudi2";
constexpr auto kScale = "custom_deepseek_v41_expert_n256_scale_gaudi2";
constexpr auto kFp8 = at::ScalarType::Float8_e4m3fn;

std::vector<int64_t> shape(const at::Stack& s) {
    const auto x = s.at(0).toTensor();
    const at::ScalarType types[] = {kFp8, at::kFloat, at::kInt, at::kFloat, at::kShort,
                                   at::kShort, at::kShort, at::kShort, at::kBFloat16,
                                   at::kBFloat16, at::kBFloat16};
    TORCH_CHECK(s.size() == 11, "Grouped N256 expects eleven tensor operands");
    for (size_t i = 0; i < s.size(); ++i) {
        auto t = s.at(i).toTensor();
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == types[i] &&
                    t.is_contiguous() && !t.requires_grad(), "Grouped N256 operand contract differs at ", i);
    }
    TORCH_CHECK(x.dim() == 3 && x.size(0) > 0 && x.size(0) <= 384 &&
                (x.size(1) == 1 || x.size(1) == 4 || x.size(1) == 8 || x.size(1) == 16) && x.size(2) > 0 &&
                x.size(2) <= 5120 && x.size(2) % 256 == 0, "Grouped N256 requires [G,M,H], M=1/4/8/16");
    const int64_t g = x.size(0), m = x.size(1), h = x.size(2);
    TORCH_CHECK(s.at(1).toTensor().sizes() == at::IntArrayRef({g,m,1}) &&
                s.at(2).toTensor().sizes() == at::IntArrayRef({1,g}) &&
                s.at(3).toTensor().sizes() == at::IntArrayRef({g,m}), "Grouped N256 route geometry differs");
    const auto q13 = s.at(4).toTensor(), q2 = s.at(5).toTensor();
    TORCH_CHECK(q13.dim() == 3 && q2.dim() == 3 && q13.size(0) > 0 && q13.size(0) <= 384 &&
                q2.size(0) == q13.size(0) && q13.size(1) > 0 && q13.size(1) <= 20 &&
                q13.size(2) == h * 64 && q2.size(1) * 256 == h && q2.size(2) > 0 &&
                q2.size(2) % 8192 == 0 && q13.size(1) * 256 == 2 * (q2.size(2) / 64),
                "Grouped N256 W13/W2 dimensions do not connect");
    for (int j = 0; j < 2; ++j) {
        const auto q = s.at(4+j).toTensor();
        TORCH_CHECK(s.at(6+j).toTensor().sizes() == at::IntArrayRef({q.size(0),q.size(1),q.size(2)/8}) &&
                    s.at(9+j).toTensor().sizes() == at::IntArrayRef({q.size(0),q.size(1),256}),
                    "Grouped N256 scale layout differs");
    }
    TORCH_CHECK(s.at(8).toTensor().sizes() == at::IntArrayRef({128}), "Grouped N256 lookup differs");
    return x.sizes().vec();
}

class Grouped final : public habana::OpBackend {
    using Tensor = synapse_helpers::tensor;
    bool reuse_;
    Tensor slice(synapse_helpers::graph& graph, synTensor input, std::vector<int64_t> dims,
                 int axis, int first, int stop, at::ScalarType type) {
        synSliceParams params{};
        for (unsigned d = 0; d < sizeof(params.axes)/sizeof(params.axes[0]); ++d) {
            params.axes[d] = d; params.steps[d] = 1; params.ends[d] = 1;
        }
        for (size_t d = 0; d < dims.size(); ++d) params.ends[dims.size()-1-d] = dims[d];
        params.starts[dims.size()-1-axis] = first;
        params.ends[dims.size()-1-axis] = stop;
        dims[axis] = stop-first;
        auto result = BuildNode(this, graph, {"slice", {input}, {{dims,type}}, &params,sizeof(params)});
        return std::move(result.at(0));
    }
    Tensor body(synapse_helpers::graph& graph, synTensor x, synTensor activation_scale,
                synTensor group_ids, synTensor routing, int g, int m, int h, int k) {
        const int r = g*m;

        synGEMMParams gemm{false, false};
        auto w13 = BuildNode(this, graph, {reuse_ ? kReuseDecode : kDecode, {group_ids,syn_in(4),syn_in(6),syn_in(8)},
            {{{g,h,2*k}, kFp8}}});
        auto p13 = BuildNode(this, graph, {"batch_gemm", {x,w13.at(0).get()},
            {{{g,m,2*k},at::kFloat}}, &gemm, sizeof(gemm)});
        auto product = ReshapeHelper(graph, p13.at(0).get(), {r,1,2*k}, at::kFloat);
        auto id_rows = ReshapeHelper(graph, group_ids, {g,1}, at::kInt);
        auto id_expand = BuildNode(this, graph, {"broadcast", {id_rows.get()}, {{{g,m},at::kInt}}});
        auto ids = ReshapeHelper(graph, id_expand.at(0).get(), {1,r}, at::kInt);
        auto sx = ReshapeHelper(graph, activation_scale, {r,1}, at::kFloat);
        auto route = ReshapeHelper(graph, routing, {1,r}, at::kFloat);
        auto middle = BuildNode(this, graph, {kSilu,
            {product.get(),ids.get(),sx.get(),syn_in(9),route.get()},
            {{{r,1,k},kFp8},{{r,1,1},at::kFloat}}});
        auto activation = ReshapeHelper(graph, middle.at(0).get(), {g,m,k}, kFp8);
        auto w2 = BuildNode(this, graph, {reuse_ ? kReuseDecode : kDecode, {group_ids,syn_in(5),syn_in(7),syn_in(8)},
            {{{g,k,h},kFp8}}});
        auto p2 = BuildNode(this, graph, {"batch_gemm", {activation.get(),w2.at(0).get()},
            {{{g,m,h},at::kFloat}}, &gemm, sizeof(gemm)});
        auto flat = ReshapeHelper(graph, p2.at(0).get(), {r,1,h}, at::kFloat);
        auto sx2 = ReshapeHelper(graph, middle.at(1).get(), {r,1}, at::kFloat);
        auto scaled = BuildNode(this, graph, {kScale, {flat.get(),ids.get(),sx2.get(),syn_in(10)},
            {{{r,1,h},at::kBFloat16}}});
        return ReshapeHelper(graph, scaled.at(0).get(), {g,m,h}, at::kBFloat16);
    }
 public:
    Grouped(int device, c10::ScalarType dtype, bool reuse = false)
        : OpBackend(device, NO_TPC + std::string("dsv41_grouped_n256"), dtype, {0}, {}, {}, false), reuse_(reuse) {
        SetOutputMetaFn([](const at::Stack& s) { return habana::OutputMetaDataVector{{at::kBFloat16, shape(s)}}; });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto out = shape(s);
        const int g = out[0], m = out[1], h = out[2];
        const int k = s.at(5).toTensor().size(2)/64;
        // Bound each connected decode/BMM island. A single B384 description
        // made the compiler materialize complete FP8 W13 batches in DRAM.
        // This is compile-time expansion, never per-group Python submission.
        std::vector<Tensor> outputs;
        std::vector<synTensor> inputs;
        for (int first = 0; first < g; first += 16) {
            const int stop = std::min(g,first+16), count = stop-first;
            auto x = slice(graph,syn_in(0),{g,m,h},0,first,stop,kFp8);
            auto sx = slice(graph,syn_in(1),{g,m,1},0,first,stop,at::kFloat);
            auto ids = slice(graph,syn_in(2),{1,g},1,first,stop,at::kInt);
            auto route = slice(graph,syn_in(3),{g,m},0,first,stop,at::kFloat);
            outputs.push_back(body(graph,x.get(),sx.get(),ids.get(),route.get(),count,m,h,k));
            inputs.push_back(outputs.back().get());
        }
        synConcatenateParams concat{};
        concat.axis = 2;
        auto result = BuildNode(this,graph,{"concat",std::move(inputs),{{out,at::kBFloat16,0}},&concat,sizeof(concat)});
        syn_out(0) = std::move(result.at(0));
    }
};
const bool registered = [] {
    for (bool reuse : {false, true}) {
        const char* schema = reuse ? kReuseSchema : kSchema;
        habana::custom_op::registerUserCustomOp(schema, reuse ? kReuseDecode : kDecode, [](const at::Stack& s) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16,shape(s)}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [reuse](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<Grouped>(device,dtype,reuse);
        });
    }
    return true;
}();
template<bool Meta, bool Reuse = false> at::Tensor run(const at::Tensor& x, const at::Tensor& sx, const at::Tensor& ids,
    const at::Tensor& route, const at::Tensor& q13, const at::Tensor& q2, const at::Tensor& s13,
    const at::Tensor& s2, const at::Tensor& lookup, const at::Tensor& c13, const at::Tensor& c2) {
    const at::Stack stack{x,sx,ids,route,q13,q2,s13,s2,lookup,c13,c2};
    const auto out = shape(stack);
    TORCH_CHECK(!Reuse || x.size(1) == 1, "Reused route graph requires one row per route");
    if (Meta) return at::empty(out,x.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered && x.device().type() == at::kHPU, "Grouped N256 requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Reuse ? kReuseSchema : kSchema);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_grouped_n256_fp8_gaudi2(Tensor x, Tensor sx, Tensor ids, Tensor route, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2) -> Tensor");
    m.def("custom_deepseek_v41_reused_n256_fp8_gaudi2(Tensor x, Tensor sx, Tensor ids, Tensor route, Tensor q13, Tensor q2, Tensor s13, Tensor s2, Tensor lookup, Tensor channel13, Tensor channel2) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("custom_deepseek_v41_grouped_n256_fp8_gaudi2",run<false>); m.impl("custom_deepseek_v41_reused_n256_fp8_gaudi2",run<false,true>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("custom_deepseek_v41_grouped_n256_fp8_gaudi2",run<true>); m.impl("custom_deepseek_v41_reused_n256_fp8_gaudi2",run<true,true>); }
