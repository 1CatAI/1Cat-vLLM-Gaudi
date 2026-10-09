// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
#include "perf_lib_layer_params.h"
#include "synapse_api.h"
#include "backend/helpers/create_tensor.h"
#include <cmath>
#ifndef DSV41_HW_FUSED_QUANT
#define DSV41_HW_FUSED_QUANT 0
#endif
#ifndef DSV41_HW_DIRECT_BIAS
#define DSV41_HW_DIRECT_BIAS 0
#endif

namespace {
#if DSV41_HW_FUSED_QUANT
#define DSV41_HW_SCHEMA "custom_deepseek_v41_hw_dense_roundtrip_fp8_gaudi2"
#else
#define DSV41_HW_SCHEMA "custom_deepseek_v41_hw_dense_fp8_gaudi2"
#endif
constexpr auto schema = "custom_op::" DSV41_HW_SCHEMA;
bool aligned(double scale) {
    return scale == 16.0 || scale == 1.0 || scale == 0.0625 || scale == 0.00390625;
}
void set_operand_bias(synTensor tensor, double logical_scale) {
    synFpQuantParam parameter{1.0, unsigned(7 - int(std::log2(logical_scale)))};
    synFpQuantMetadata metadata{syn_type_fp8_143, &parameter, 1};
    const auto status = synTensorSetQuantizationData(tensor, SYN_FP_QUANT_METADATA,
                                                    &metadata, sizeof(metadata));
    TORCH_CHECK(status == synSuccess, "Private FP8 exponent-bias metadata rejected: ", int(status));
}
void* scale_pointer(double scale) {
    // Immutable storage survives graph/backend destruction and replay. The
    // installed complex GUID accepts aligned scales only as H2D metadata,
    // not ordinary constant nodes. No request-time scale data is produced.
    static const at::BFloat16 values[] = {at::BFloat16(16.0f), at::BFloat16(1.0f),
        at::BFloat16(0.0625f), at::BFloat16(0.00390625f)};
    for (const auto& value : values)
        if (scale == float(value)) return const_cast<at::BFloat16*>(&value);
    TORCH_CHECK(false, "Unsupported hardware scale");
}
habana::OutputMetaDataVector meta(const at::Stack& stack) {
    const auto x = stack.at(0).toTensor(), w = stack.at(1).toTensor();
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::ScalarType::Float8_e4m3fn &&
        x.dim() == 2 && w.dim() == 2 && x.size(0) >= 2 && x.size(0) <= 6 && x.size(1) == w.size(1) &&
        ((x.size(1) == 5120 && w.size(0) == 1792) ||
         (x.size(1) == 1280 && (w.size(0) == 8192 || w.size(0) == 16384))) &&
        x.is_contiguous() && w.is_contiguous() && !x.requires_grad() && !w.requires_grad() &&
        x.device() == w.device() && aligned(stack.at(2).toDouble()) && aligned(stack.at(3).toDouble()),
        "Static FP8 dense requires C2-C6 QKV/query operands and hardware-aligned positive scales");
    return {{at::kBFloat16, {x.size(0), w.size(0)}}};
}
class HardwareDense final : public habana::OpBackend {
public:
    HardwareDense(int device, c10::ScalarType dtype)
        : OpBackend(device, NO_TPC + std::string("dsv41_hw_dense"), dtype, {0}, {}, {}, false) {
        SetOutputMetaFn(meta);
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto output = meta(stack);
        const auto sx = stack.at(2).toDouble(), sw = stack.at(3).toDouble();
#if DSV41_HW_DIRECT_BIAS
        // This private capability build bypasses the complex scaled GUID.
        // Only the newly allocated FP8 activation and private prepared weight
        // receive quantization metadata; shared C1 tensors are never modified.
        ns_ConvertToFp8::ParamsV2 cast{};
        cast.round_mode = CAST_ROUND_HALF_NE;
        synGEMMParams params{false, true};
#if DSV41_HW_FUSED_QUANT
        float reciprocal = float(1.0 / sx);
#else
        auto reciprocal = BuildConstantTensor(this, graph, 1.0 / sx, at::kFloat);
#endif
        if (isOutputInfMode()) {
#if DSV41_HW_FUSED_QUANT
            auto q = BuildNode(this, graph, {"custom_deepseek_v41_fixed_dense_quant_gaudi2", {syn_in(0)},
                {{stack.at(0).toTensor().sizes().vec(), at::ScalarType::Float8_e4m3fn}},
                &reciprocal, sizeof(reciprocal)});
#else
            auto q = BuildNode(this, graph, {"convert_to_fp8_bf16", {syn_in(0), reciprocal.get()},
                {{stack.at(0).toTensor().sizes().vec(), at::ScalarType::Float8_e4m3fn}}, &cast, sizeof(cast)});
#endif
            auto product = BuildNode(this, graph, {"gemm", {q.at(0).get(), syn_in(1)},
                {{output.at(0).shape, at::kBFloat16, 0}}, &params, sizeof(params)});
            syn_out(0) = std::move(product.at(0));
            return;
        }
        // Synapse locks tensor attributes at first graph-node insertion.
        // Prepare metadata before either operand participates in a node.
        auto x = stack.at(0).toTensor();
        auto q = habana_helpers::create_tensor(x.sizes(), x.strides(), graph, false, false,
            graph.get_device().id(), at::ScalarType::Float8_e4m3fn);
        set_operand_bias(q.get(), sx);
        set_operand_bias(syn_in(1), sw);
        // The converter does not consume operand exponent bias. Supply its
        // exact reciprocal as recipe-owned data; only MME consumes the bias.
#if DSV41_HW_FUSED_QUANT
        graph.add_node({syn_in(0)}, {q.get()}, &reciprocal, sizeof(reciprocal),
                       "custom_deepseek_v41_fixed_dense_quant_gaudi2", nullptr, nullptr, nullptr, true);
#else
        graph.add_node({syn_in(0), reciprocal.get()}, {q.get()}, &cast, sizeof(cast), "convert_to_fp8_bf16",
                       nullptr, nullptr, nullptr, true);
#endif
        auto product = BuildNode(this, graph, {"gemm", {q.get(), syn_in(1)},
            {{output.at(0).shape, at::kBFloat16, 0}}, &params, sizeof(params)});
#else
        // Constants belong to the recipe. No row amax, H2D scale or TPC
        // product epilogue participates in replay. The complex GEMM lowers
        // aligned scales to Gaudi2's operand exponent biases.
        std::vector<synTensor> scale_inputs;
        CreateH2dTensorInput(graph, at::kBFloat16, scale_pointer(sx), sizeof(at::BFloat16),
                            scale_inputs, HOST_TO_DEVICE_TENSOR, true);
        CreateH2dTensorInput(graph, at::kBFloat16, scale_pointer(sw), sizeof(at::BFloat16),
                            scale_inputs, HOST_TO_DEVICE_TENSOR, true);
        ns_ConvertToFp8::ParamsV2 cast{};
        cast.round_mode = CAST_ROUND_HALF_NE;
        auto q = BuildNode(this, graph, {"convert_to_fp8_bf16", {syn_in(0), scale_inputs.at(0)},
            {{stack.at(0).toTensor().sizes().vec(), at::ScalarType::Float8_e4m3fn}}, &cast, sizeof(cast)});
        ns_Fp8Gemm::ParamsV2 params{};
        params.transpose_a = false;
        params.transpose_b = true;
        params.is_hw_aligned = true;
        auto product = BuildNode(this, graph, {"fp8_gemm_bf16",
            {q.at(0).get(), syn_in(1), scale_inputs.at(0), scale_inputs.at(1), nullptr, nullptr, nullptr},
            {{output.at(0).shape, at::kBFloat16, 0}}, &params, sizeof(params)});
#endif
        syn_out(0) = std::move(product.at(0));
    }
};
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema, "fp8_gemm_bf16", [](const at::Stack& stack) {
        const auto output = meta(stack).at(0);
        return habana::PartialOutputMetaDataVector{{output.dtype, output.shape}};
    }, nullptr);
    habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
        return std::make_shared<HardwareDense>(device, dtype);
    });
    return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& x, const at::Tensor& w, double sx, double sw) {
    const auto output = meta({x, w, sx, sw}).at(0);
    if constexpr (Meta) return at::empty(output.shape, x.options());
    TORCH_CHECK(registered && x.device().type() == at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({x, w, sx, sw}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def(DSV41_HW_SCHEMA "(Tensor value, Tensor weight, float input_scale, float weight_scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl(DSV41_HW_SCHEMA, run<false>); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl(DSV41_HW_SCHEMA, run<true>); }
