// SPDX-License-Identifier: Apache-2.0
// Observable operand entrypoints share the exact kernels used by the MoE graph.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kQuant = "custom_deepseek_v41_dynamic_quant_bf16_gaudi2";
constexpr auto kDecode = "custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2";
constexpr auto kQuantSchema = "custom_op::custom_deepseek_v41_dynamic_quant_bf16_gaudi2";
constexpr auto kDecodeSchema = "custom_op::custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2";
using Pair = std::tuple<at::Tensor, at::Tensor>;

habana::OutputMetaDataVector metadata(const at::Stack& stack, bool decode) {
    const auto source = stack.at(0).toTensor();
    for (const auto& value : stack) {
        const auto tensor = value.toTensor();
        TORCH_CHECK(tensor.is_contiguous() && !tensor.requires_grad() && tensor.device() == source.device(),
                    "V4.1 FP8 operands require matching contiguous inference tensors");
    }
    if (!decode) {
        TORCH_CHECK(source.scalar_type() == at::kBFloat16 && source.dim() == 2 &&
                    source.size(0) >= 1 && source.size(0) <= 49152 && source.size(1) >= 256 &&
                    source.size(1) <= 5120 && source.size(1) % 128 == 0,
                    "V4.1 dynamic quantization requires 1..49152 BF16 rows and K128");
        return {{at::ScalarType::Float8_e4m3fn, source.sizes().vec()}, {at::kFloat, {source.size(0), 1}}};
    }
    const auto q = stack.at(1).toTensor(), s = stack.at(2).toTensor();
    const auto channel = stack.at(3).toTensor(), lookup = stack.at(4).toTensor();
    TORCH_CHECK(source.scalar_type() == at::kInt && source.sizes() == at::IntArrayRef({1, 6}) &&
                q.scalar_type() == at::kShort && q.dim() == 3 && q.size(0) > 0 && q.size(0) <= 384 &&
                q.size(1) > 0 && q.size(1) <= 40 && q.size(2) > 0 && q.size(2) <= 163840 && q.size(2) % 4096 == 0 &&
                s.scalar_type() == at::kBFloat16 && s.sizes() == at::IntArrayRef({q.size(0), q.size(1), q.size(2) / 8}) &&
                channel.scalar_type() == at::kBFloat16 && channel.sizes() == at::IntArrayRef({q.size(0), q.size(1), 128}) &&
                lookup.scalar_type() == at::kBFloat16 && lookup.sizes() == at::IntArrayRef({128}),
                "V4.1 FP8 decode requires qualified Q16/S16 and per-channel power-of-two scales");
    return {{at::ScalarType::Float8_e4m3fn, {6, q.size(2) / 32, q.size(1) * 128}},
            {at::kFloat, {6, 1, q.size(1) * 128}}};
}

class Operands final : public habana::OpBackend {
    bool decode_;
public:
    Operands(int device, c10::ScalarType dtype, bool decode)
        : OpBackend(device, NO_TPC + std::string("dsv41_fp8_operands"), dtype, {0, 1}, {}, {}, false), decode_(decode) {
        SetOutputMetaFn([decode](const at::Stack& stack) { return metadata(stack, decode); });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto meta = metadata(stack, decode_);
        std::vector<synTensor> inputs;
        for (size_t i = 0; i < stack.size(); ++i) inputs.push_back(syn_in(i));
        auto result = BuildNode(this, graph, {decode_ ? kDecode : kQuant, inputs,
            {{meta.at(0).shape, at::ScalarType::Float8_e4m3fn, 0}, {meta.at(1).shape, at::kFloat, 1}}});
        syn_out(0) = std::move(result.at(0));
        syn_out(1) = std::move(result.at(1));
    }
};

const bool registered = [] {
    for (bool decode : {false, true}) {
        const auto schema = decode ? kDecodeSchema : kQuantSchema;
        habana::custom_op::registerUserCustomOp(schema, decode ? kDecode : kQuant, [decode](const at::Stack& stack) {
            const auto meta = metadata(stack, decode);
            return habana::PartialOutputMetaDataVector{{at::ScalarType::Float8_e4m3fn, meta.at(0).shape},
                                                       {at::kFloat, meta.at(1).shape}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [decode](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<Operands>(device, dtype, decode);
        });
    }
    return true;
}();

template<bool Meta> Pair run(const at::Stack& stack, bool decode) {
    const auto shape = metadata(stack, decode);
    const auto source = stack.at(0).toTensor();
    if (Meta) return {at::empty(shape.at(0).shape, source.options().dtype(at::ScalarType::Float8_e4m3fn)),
                      at::empty(shape.at(1).shape, source.options().dtype(at::kFloat))};
    TORCH_CHECK(registered && source.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(decode ? kDecodeSchema : kQuantSchema);
    auto result = descriptor.execute(stack);
    return {result.at(0), result.at(1)};
}
template<bool Meta> Pair quant(const at::Tensor& value) { return run<Meta>({value}, false); }
template<bool Meta> Pair decode(const at::Tensor& ids, const at::Tensor& q, const at::Tensor& s,
    const at::Tensor& channel, const at::Tensor& lookup) { return run<Meta>({ids, q, s, channel, lookup}, true); }
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("custom_deepseek_v41_dynamic_quant_bf16_gaudi2(Tensor value) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2(Tensor ids, Tensor q, Tensor s, Tensor channel, Tensor lookup) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("custom_deepseek_v41_dynamic_quant_bf16_gaudi2", quant<false>);
    m.impl("custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2", decode<false>);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("custom_deepseek_v41_dynamic_quant_bf16_gaudi2", quant<true>);
    m.impl("custom_deepseek_v41_mxfp4_prepared_dequant_fp8_gaudi2", decode<true>);
}
