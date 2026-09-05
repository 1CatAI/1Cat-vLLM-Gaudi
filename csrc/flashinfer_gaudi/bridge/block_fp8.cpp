// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include <limits>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kDequant = "custom_op::flashinfer_gaudi_block_fp8_dequant";
constexpr auto kLinear = "custom_op::flashinfer_gaudi_block_fp8_linear";
constexpr auto kGuid = "flashinfer_gaudi_block_fp8_dequant_gaudi2";

void validate_weights(const at::Tensor& weight, const at::Tensor& scale) {
    TORCH_CHECK(weight.dim() == 2 && weight.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                scale.dim() == 2 && scale.scalar_type() == at::kFloat,
                "block-FP8 requires rank-2 E4M3 weight and FP32 scale");
    TORCH_CHECK(weight.is_contiguous() && scale.is_contiguous() && !weight.requires_grad() && !scale.requires_grad(),
                "block-FP8 requires contiguous inference inputs");
    TORCH_CHECK(weight.device() == scale.device(), "block-FP8 inputs must share a device");
    for (int dim = 0; dim < 2; ++dim) {
        TORCH_CHECK(weight.size(dim) > 0 && weight.size(dim) <= std::numeric_limits<int32_t>::max() &&
                    weight.size(dim) % 128 == 0 && scale.size(dim) == weight.size(dim) / 128,
                    "block-FP8 requires aligned [N,K] weights and [N/128,K/128] scales");
    }
}

std::vector<int64_t> output_shape(const at::Stack& stack, bool linear) {
    const auto& weight = stack.at(linear ? 1 : 0).toTensor();
    const auto& scale = stack.at(linear ? 2 : 1).toTensor();
    validate_weights(weight, scale);
    if (!linear) return weight.sizes().vec();
    const auto& x = stack.at(0).toTensor();
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && !x.requires_grad(),
                "block-FP8 linear requires contiguous inference BF16 [M,K]");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= std::numeric_limits<int32_t>::max() &&
                x.size(1) == weight.size(1) && x.device() == weight.device(),
                "block-FP8 linear input shape/device mismatch");
    return {x.size(0), weight.size(0)};
}

class BlockFp8 final : public habana::OpBackend {
    bool linear_;
public:
    BlockFp8(int device, c10::ScalarType dtype, bool linear)
        : OpBackend(device, NO_TPC + std::string("flashinfer_gaudi_block_fp8"), dtype, {0}, {}, {}, false),
          linear_(linear) {
        SetOutputMetaFn([linear](const at::Stack& stack) {
            return habana::OutputMetaDataVector{{at::kBFloat16, output_shape(stack, linear)}};
        });
    }
    void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
        const auto shape = output_shape(stack, linear_);
        const int index = linear_ ? 1 : 0;
        const auto weight_shape = stack.at(index).toTensor().sizes().vec();
        // No final-output index for the linear temporary: Synapse owns it.
        habana::NodeAttr::NodeOutputAttr dequant_output{weight_shape, at::kBFloat16};
        if (!linear_) dequant_output.final_result_index = 0;
        auto dequant = BuildNode(this, graph, {kGuid, {syn_in(index), syn_in(index + 1)}, {dequant_output}});
        if (!linear_) {
            syn_out(0) = std::move(dequant.at(0));
            return;
        }
        synGEMMParams params{false, true};
        auto result = BuildNode(this, graph, {"gemm", {syn_in(0), dequant.at(0).get()},
            {{shape, at::kBFloat16, 0}}, &params, sizeof(params)});
        syn_out(0) = std::move(result.at(0));
    }
};

const bool registered = [] {
    for (bool linear : {false, true}) {
        const char* schema = linear ? kLinear : kDequant;
        habana::custom_op::registerUserCustomOp(schema, kGuid, [linear](const at::Stack& stack) {
            return habana::PartialOutputMetaDataVector{{at::kBFloat16, output_shape(stack, linear)}};
        }, nullptr);
        habana::KernelRegistry().add(schema, [linear](synDeviceId device, c10::ScalarType dtype) {
            return std::make_shared<BlockFp8>(device, dtype, linear);
        });
    }
    return true;
}();

at::Tensor execute(const at::Stack& stack, bool linear) {
    output_shape(stack, linear);
    TORCH_CHECK(stack.at(0).toTensor().device().type() == at::kHPU && registered, "block-FP8 native requires HPU");
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(linear ? kLinear : kDequant);
    return descriptor.execute(stack).at(0);
}
at::Tensor dequant(const at::Tensor& weight, const at::Tensor& scale) { return execute({weight, scale}, false); }
at::Tensor linear(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale) {
    return execute({x, weight, scale}, true);
}
at::Tensor dequant_meta(const at::Tensor& weight, const at::Tensor& scale) {
    return at::empty(output_shape({weight, scale}, false), weight.options().dtype(at::kBFloat16));
}
at::Tensor linear_meta(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& scale) {
    return at::empty(output_shape({x, weight, scale}, true), x.options());
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
    m.def("flashinfer_gaudi_block_fp8_dequant(Tensor weight, Tensor scale) -> Tensor");
    m.def("flashinfer_gaudi_block_fp8_linear(Tensor x, Tensor weight, Tensor scale) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
    m.impl("flashinfer_gaudi_block_fp8_dequant", dequant);
    m.impl("flashinfer_gaudi_block_fp8_linear", linear);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
    m.impl("flashinfer_gaudi_block_fp8_dequant", dequant_meta);
    m.impl("flashinfer_gaudi_block_fp8_linear", linear_meta);
}
