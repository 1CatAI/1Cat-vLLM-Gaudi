// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_candidate_coordinates_gaudi2";
constexpr auto guid = "custom_deepseek_v41_candidate_coordinates_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto& b = s.at(0).toTensor();
    TORCH_CHECK(b.scalar_type() == at::kInt && b.dim() == 2 && b.size(0) >= 1 && b.size(0) <= 6 &&
                b.size(1) >= 1 && b.size(1) <= 2048 && b.is_contiguous(), "Candidate coordinates require I32 [C1-C6,1..2048]");
    TORCH_CHECK(s.at(1).toInt() >= 0 && s.at(1).toInt() <= INT32_MAX, "Candidate maximum row must fit nonnegative I32");
    return {{at::kInt, {b.size(0), b.size(1)*8}}, {at::kInt, {b.size(0), b.size(1)*8}}};
}
class Coordinates final : public habana::OpBackend {
public:
    Coordinates(int d, c10::ScalarType t) : OpBackend(d, guid, t, {0,1}, {}, {}, false) { SetOutputMetaFn(metadata); }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& s) override {
        const auto out = metadata(s); int maximum = s.at(1).toInt();
        auto result = BuildNode(this, graph, {s.at(2).toBool() ? "custom_deepseek_v41_candidate_coordinates_global_gaudi2" : guid, {syn_in(0)},
            {{out[0].shape, out[0].dtype, 0}, {out[1].shape, out[1].dtype, 1}}, &maximum, sizeof(maximum)});
        syn_out(0) = std::move(result[0]); syn_out(1) = std::move(result[1]);
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, guid, [](const at::Stack& s) {
        habana::PartialOutputMetaDataVector result;
        for (const auto& o : metadata(s)) result.push_back({o.dtype, o.shape});
        return result;
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId d, c10::ScalarType t) { return std::make_shared<Coordinates>(d,t); });
    return true;
}();
template<bool Meta> std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& blocks, int64_t maximum, bool whole) {
    const at::Stack s{blocks, maximum, whole}; const auto out = metadata(s);
    if constexpr (Meta) return {at::empty(out[0].shape, blocks.options()), at::empty(out[1].shape, blocks.options())};
    TORCH_CHECK(registered && blocks.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    auto result = descriptor.execute(s); return {result[0], result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_candidate_coordinates_gaudi2(Tensor blocks, int maximum_row, bool whole_output=False) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) { m.impl("custom_deepseek_v41_candidate_coordinates_gaudi2",run<false>); }
TORCH_LIBRARY_IMPL(custom_op,Meta,m) { m.impl("custom_deepseek_v41_candidate_coordinates_gaudi2",run<true>); }
