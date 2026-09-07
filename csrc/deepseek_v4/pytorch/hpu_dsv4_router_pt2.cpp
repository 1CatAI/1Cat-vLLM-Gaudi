#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {

constexpr const char* kSchema =
    "custom_op::custom_deepseek_v4_topk_softplus_sqrt_gaudi2";
constexpr const char* kGuid =
    "custom_deepseek_v4_topk_softplus_sqrt_gaudi2";

bool register_router_topk() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& gating_output = inputs.at(0).toTensor();
    const auto tokens = gating_output.size(0);

    habana::PartialOutputMetaData weights;
    weights.dtype = at::ScalarType::Float;
    weights.shape = {tokens, 6, 1};

    habana::PartialOutputMetaData ids;
    ids.dtype = at::ScalarType::Int;
    ids.shape = {tokens, 6, 1};

    return habana::PartialOutputMetaDataVector{weights, ids};
  };
  habana::custom_op::registerUserCustomOp(
      kSchema, kGuid, output_meta, nullptr);
  return true;
}

const bool kRouterTopkRegistered = register_router_topk();

std::tuple<at::Tensor, at::Tensor> router_topk_hpu(
    const at::Tensor& gating_output,
    const at::Tensor& correction_bias) {
  TORCH_CHECK(
      gating_output.scalar_type() == at::ScalarType::Float &&
      correction_bias.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(
      gating_output.dim() == 2 && gating_output.size(0) > 0 &&
      gating_output.size(1) == 256);
  TORCH_CHECK(
      correction_bias.dim() == 1 && correction_bias.size(0) == 256);
  TORCH_CHECK(
      gating_output.is_contiguous() && correction_bias.is_contiguous());
  TORCH_CHECK(kRouterTopkRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kSchema);
  std::vector<c10::IValue> inputs{gating_output, correction_bias};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> router_topk_meta(
    const at::Tensor& gating_output,
    const at::Tensor& correction_bias) {
  (void)correction_bias;
  const auto tokens = gating_output.size(0);
  return std::make_tuple(
      at::empty({tokens, 6, 1}, gating_output.options()),
      at::empty(
          {tokens, 6, 1},
          gating_output.options().dtype(at::ScalarType::Int)));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def(
      "custom_deepseek_v4_topk_softplus_sqrt_gaudi2("
      "Tensor gating_output, Tensor correction_bias) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl(
      "custom_deepseek_v4_topk_softplus_sqrt_gaudi2",
      router_topk_hpu);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl(
      "custom_deepseek_v4_topk_softplus_sqrt_gaudi2",
      router_topk_meta);
}
