// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <limits>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kFunctional = "custom_op::flashinfer_gaudi_gemm_silu";
constexpr auto kOut = "custom_op::flashinfer_gaudi_gemm_silu_out";

std::vector<int64_t> validate(const at::Tensor& x, const at::Tensor& weight) {
  for (const auto& tensor : {x, weight}) {
    TORCH_CHECK(!tensor.requires_grad(), "native GEMM-SiLU is inference-only");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 2 && tensor.is_contiguous(),
                "native GEMM-SiLU requires contiguous rank-2 BF16 inputs");
    for (auto dim : tensor.sizes()) {
      TORCH_CHECK(dim > 0 && dim <= std::numeric_limits<int32_t>::max(), "invalid native GEMM-SiLU dimension");
    }
  }
  TORCH_CHECK(x.device() == weight.device(), "native GEMM-SiLU inputs must share a device");
  TORCH_CHECK(x.size(1) == weight.size(0) && weight.size(1) % 256 == 0,
              "native GEMM-SiLU requires x[M,K], weight[K,2D], D divisible by 128");
  return {x.size(0), weight.size(1) / 2};
}

void validate_out(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& out) {
  const auto shape = validate(x, weight);
  TORCH_CHECK(out.device() == x.device() && out.scalar_type() == x.scalar_type() &&
                  out.sizes().vec() == shape && out.is_contiguous() && !out.requires_grad(),
              "native GEMM-SiLU out must match shape, dtype and device, be contiguous and inference-only");
  // Reject all shared storage, including disjoint views: no hidden copy or resize.
  TORCH_CHECK(!out.is_alias_of(x) && !out.is_alias_of(weight), "native GEMM-SiLU out aliases an input");
}

habana::OutputMetaDataVector output_meta(const at::Stack& stack) {
  return {{at::kBFloat16, validate(stack.at(0).toTensor(), stack.at(1).toTensor())}};
}

class GemmSilu final : public habana::OpBackend {
 public:
  GemmSilu(int device, c10::ScalarType dtype, bool out)
      : OpBackend(device, NO_TPC + std::string("flashinfer_gaudi_gemm_silu"), dtype,
                  out ? std::vector<int>{} : std::vector<int>{0}, {}, {}, out) {
    SetOutputMetaFn(output_meta);
  }

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
    const auto& x = stack.at(0).toTensor();
    const auto& weight = stack.at(1).toTensor();
    const auto shape = validate(x, weight);
    synGEMMParams params{false, false};
    // No final-output index: this BF16 temporary belongs to the recipe.
    auto packed = BuildNode(this, graph, {"gemm", {syn_in(0), syn_in(1)},
        {{{x.size(0), weight.size(1)}, at::kBFloat16}}, &params, sizeof(params)});
    auto result = BuildNode(this, graph, {"flashinfer_gaudi_silu_and_mul_bf16_gaudi2",
        {packed.at(0).get()}, {{shape, at::kBFloat16, 0}}});
    syn_out(0) = std::move(result.at(0));
  }
};

const bool registered = [] {
  habana::custom_op::registerUserCustomOp(kFunctional, "unused_compound_guid",
      [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kBFloat16,
            validate(stack.at(0).toTensor(), stack.at(1).toTensor())}};
      }, nullptr);
  habana::KernelRegistry().add(kFunctional, [](synDeviceId device, c10::ScalarType dtype) {
    return std::make_shared<GemmSilu>(device, dtype, false);
  });
  habana::KernelRegistry().add(kOut, [](synDeviceId device, c10::ScalarType dtype) {
    return std::make_shared<GemmSilu>(device, dtype, true);
  });
  return true;
}();

at::Tensor run(const at::Tensor& x, const at::Tensor& weight) {
  validate(x, weight);
  TORCH_CHECK(x.device().type() == at::kHPU && registered, "native GEMM-SiLU requires HPU");
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kFunctional);
  return descriptor.execute({x, weight}).at(0);
}

at::Tensor& run_out(const at::Tensor& x, const at::Tensor& weight, at::Tensor& out) {
  validate_out(x, weight, out);
  TORCH_CHECK(x.device().type() == at::kHPU && registered, "native GEMM-SiLU requires HPU");
  habana::eager::EagerOp<at::Tensor&> op{kOut, {x, weight, out}, {out.sizes().vec()}, 2};
  op.set_eager_op_info({habana::eager::eagerOpKind::InplaceOut, kOut, 1});
  return op.call(out);
}

at::Tensor meta(const at::Tensor& x, const at::Tensor& weight) {
  return at::empty(validate(x, weight), x.options());
}

at::Tensor& meta_out(const at::Tensor& x, const at::Tensor& weight, at::Tensor& out) {
  validate_out(x, weight, out);
  return out;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("flashinfer_gaudi_gemm_silu(Tensor x, Tensor weight) -> Tensor");
  m.def("flashinfer_gaudi_gemm_silu_out(Tensor x, Tensor weight, Tensor(a!) out) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("flashinfer_gaudi_gemm_silu", run);
  m.impl("flashinfer_gaudi_gemm_silu_out", run_out);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("flashinfer_gaudi_gemm_silu", meta);
  m.impl("flashinfer_gaudi_gemm_silu_out", meta_out);
}
