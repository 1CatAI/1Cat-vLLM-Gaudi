// SPDX-License-Identifier: Apache-2.0
// Research-only decomposition; checksum outputs are not model outputs.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "synapse_common_types.h"

namespace {
constexpr auto schema = "custom_op::custom_deepseek_v41_expert_decode_probe_gaudi2";
constexpr auto fp8 = at::ScalarType::Float8_e4m3fn;
habana::OutputMetaDataVector metadata(const at::Stack& stack) {
    const auto ids = stack.at(0).toTensor(), q = stack.at(1).toTensor();
    const auto s = stack.at(2).toTensor(), lut = stack.at(3).toTensor(), x = stack.at(4).toTensor();
    const auto mode = stack.at(5).toInt();
    const auto pack = stack.at(6).toInt();
    TORCH_CHECK(ids.scalar_type() == at::kInt && ids.dim() == 1 && ids.numel() % 2 == 0 &&
                ids.numel() >= 2 && ids.numel() <= 36 && q.scalar_type() == at::kShort && q.dim() == 3 &&
                q.size(2) % 8192 == 0 && s.scalar_type() == at::kShort && s.dim() == 3 &&
                s.size(0) == q.size(0) && s.size(1) == q.size(1) && s.size(2) == q.size(2) / 16 + 128 &&
                lut.scalar_type() == at::kBFloat16 && lut.numel() == 128 && mode >= 0 && mode <= 3 &&
                (pack == 1 || pack == 2) && x.scalar_type() == fp8 &&
                x.sizes() == at::IntArrayRef({ids.numel()/pack, 1, q.size(2)/64}),
                "Invalid SAT diagnostic layout");
    for (const auto& tensor : {ids, q, s, lut, x})
        TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad());
    return {{mode < 2 ? at::kByte : at::kFloat,
             {ids.numel()/pack, mode < 2 ? q.size(2)/8192 : 1, q.size(1)*256*pack}}};
}
class Probe final : public habana::OpBackend {
public:
    Probe(int device, c10::ScalarType type)
        : OpBackend(device, NO_TPC + std::string("dsv41_expert_decode_probe"), type, {0}, {}, {}, false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto meta = metadata(stack);
        const auto mode = stack.at(5).toInt();
        int32_t pack = stack.at(6).toInt();
        const char* guid = mode == 0 ? "custom_deepseek_v41_expert_probe_read_gaudi2" :
                           mode == 1 ? "custom_deepseek_v41_expert_probe_decode_gaudi2" :
                           mode == 2 ? "custom_deepseek_v41_expert_probe_store_gaudi2" :
                           pack == 2 ? "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2" :
                                       "custom_deepseek_v41_expert_n256_sat_fp8_gaudi2";
        const auto q = stack.at(1).toTensor();
        const auto slots = stack.at(0).toTensor().numel()/pack;
        auto ids = ReshapeHelper(graph, syn_in(0), {1, slots*pack}, at::kInt);
        auto decoded = BuildNode(this, graph, {guid, {ids.get(), syn_in(1), syn_in(2), syn_in(3)},
            {{{slots, mode < 2 ? q.size(2)/8192 : q.size(2)/64, q.size(1)*256*pack},
              mode < 2 ? at::kByte : fp8, mode < 2 ? c10::optional<int>{0} : c10::nullopt}},
              mode < 3 ? &pack : nullptr, mode < 3 ? sizeof(pack) : 0});
        if (mode < 2) {
            syn_out(0) = std::move(decoded.at(0));
            return;
        }
        synGEMMParams parameters{false, false};
        auto result = BuildNode(this, graph, {"batch_gemm", {syn_in(4), decoded.at(0).get()},
            {{meta.at(0).shape, at::kFloat, 0}}, &parameters, sizeof(parameters)});
        syn_out(0) = std::move(result.at(0));
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "custom_deepseek_v41_expert_probe_read_gaudi2",
        [](const at::Stack& stack) {
            const auto meta = metadata(stack);
            return habana::PartialOutputMetaDataVector{{meta.at(0).dtype, meta.at(0).shape}};
        }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType type) {
        return std::make_shared<Probe>(device, type);
    });
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& ids, const at::Tensor& q, const at::Tensor& s,
                                  const at::Tensor& lut, const at::Tensor& x, int64_t mode, int64_t pack) {
    const at::Stack stack{ids,q,s,lut,x,mode,pack};
    const auto meta = metadata(stack);
    if (Meta) return at::empty(meta.at(0).shape, q.options().dtype(meta.at(0).dtype));
    TORCH_CHECK(registered && q.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute(stack).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_expert_decode_probe_gaudi2(Tensor ids, Tensor q16, Tensor scales, Tensor lookup, Tensor x, int mode, int route_pack) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_expert_decode_probe_gaudi2", run<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_expert_decode_probe_gaudi2", run<true>);
}
