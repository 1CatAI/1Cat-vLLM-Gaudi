#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {

constexpr const char* kPostPrepareSchema =
    "custom_op::custom_deepseek_v4_mhc_post_prepare_gaudi2";
constexpr const char* kPostPrepareGuid =
    "custom_deepseek_v4_mhc_post_prepare_gaudi2";
constexpr const char* kPreEmitSchema =
    "custom_op::custom_deepseek_v4_mhc_pre_emit_gaudi2";
constexpr const char* kPreEmitGuid =
    "custom_deepseek_v4_mhc_pre_emit_gaudi2";
constexpr const char* kPreEmitNormSchema =
    "custom_op::custom_deepseek_v4_mhc_pre_emit_norm_gaudi2";
constexpr const char* kPreEmitNormGuid =
    "custom_deepseek_v4_mhc_pre_emit_norm_gaudi2";

bool register_mhc_post_prepare() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    const auto& residual = inputs.at(1).toTensor();

    habana::PartialOutputMetaData residual_output;
    residual_output.dtype = at::ScalarType::BFloat16;
    residual_output.shape = residual.sizes().vec();

    habana::PartialOutputMetaData residual_f32;
    residual_f32.dtype = at::ScalarType::Float;
    residual_f32.shape = {x.size(0), 4 * 4096};

    habana::PartialOutputMetaData rrms;
    rrms.dtype = at::ScalarType::Float;
    rrms.shape = {x.size(0), 1};

    return habana::PartialOutputMetaDataVector{
        residual_output, residual_f32, rrms};
  };
  habana::custom_op::registerUserCustomOp(
      kPostPrepareSchema,
      kPostPrepareGuid,
      output_meta,
      nullptr);
  return true;
}

bool register_mhc_pre_emit() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& residual = inputs.at(0).toTensor();
    const auto tokens = residual.size(0);

    habana::PartialOutputMetaData post_mix;
    post_mix.dtype = at::ScalarType::Float;
    post_mix.shape = {tokens, 4, 1};

    habana::PartialOutputMetaData comb_mix;
    comb_mix.dtype = at::ScalarType::Float;
    comb_mix.shape = {tokens, 16, 1};

    habana::PartialOutputMetaData layer_input;
    layer_input.dtype = at::ScalarType::BFloat16;
    layer_input.shape = {tokens, 4096};

    return habana::PartialOutputMetaDataVector{
        post_mix, comb_mix, layer_input};
  };
  habana::custom_op::registerUserCustomOp(
      kPreEmitSchema,
      kPreEmitGuid,
      output_meta,
      nullptr);
  return true;
}

bool register_mhc_pre_emit_norm() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& residual = inputs.at(0).toTensor();
    const auto tokens = residual.size(0);

    habana::PartialOutputMetaData post_mix;
    post_mix.dtype = at::ScalarType::Float;
    post_mix.shape = {tokens, 4, 1};

    habana::PartialOutputMetaData comb_mix;
    comb_mix.dtype = at::ScalarType::Float;
    comb_mix.shape = {tokens, 16, 1};

    habana::PartialOutputMetaData layer_input;
    layer_input.dtype = at::ScalarType::BFloat16;
    layer_input.shape = {tokens, 4096};

    return habana::PartialOutputMetaDataVector{
        post_mix, comb_mix, layer_input};
  };
  habana::custom_op::registerUserCustomOp(
      kPreEmitNormSchema,
      kPreEmitNormGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kMHCPostPrepareRegistered = register_mhc_post_prepare();
const bool kMHCPreEmitRegistered = register_mhc_pre_emit();
const bool kMHCPreEmitNormRegistered = register_mhc_pre_emit_norm();

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_post_prepare_hpu(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix) {
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::BFloat16 &&
      residual.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      post_layer_mix.scalar_type() == at::ScalarType::Float &&
      comb_res_mix.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 4096);
  TORCH_CHECK(
      residual.dim() == 3 && residual.size(0) == x.size(0) &&
      residual.size(1) == 4 && residual.size(2) == 4096);
  TORCH_CHECK(
      post_layer_mix.dim() == 3 &&
      post_layer_mix.size(0) == x.size(0) &&
      post_layer_mix.size(1) == 4 && post_layer_mix.size(2) == 1);
  TORCH_CHECK(
      comb_res_mix.dim() == 3 &&
      comb_res_mix.size(0) == x.size(0) &&
      comb_res_mix.size(1) == 4 && comb_res_mix.size(2) == 4);
  TORCH_CHECK(
      x.is_contiguous() && residual.is_contiguous() &&
      post_layer_mix.is_contiguous() && comb_res_mix.is_contiguous());
  TORCH_CHECK(kMHCPostPrepareRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kPostPrepareSchema);
  std::vector<c10::IValue> inputs{
      x, residual, post_layer_mix, comb_res_mix};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 3);
  return std::make_tuple(outputs.at(0), outputs.at(1), outputs.at(2));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_post_prepare_meta(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix) {
  (void)post_layer_mix;
  (void)comb_res_mix;
  const auto float_options = x.options().dtype(at::ScalarType::Float);
  return std::make_tuple(
      at::empty_like(residual),
      at::empty({x.size(0), 4 * 4096}, float_options),
      at::empty({x.size(0), 1}, float_options));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre_emit_hpu(
    const at::Tensor& residual,
    const at::Tensor& raw_mixes,
    const at::Tensor& rrms,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base) {
  TORCH_CHECK(residual.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      raw_mixes.scalar_type() == at::ScalarType::Float &&
      rrms.scalar_type() == at::ScalarType::Float &&
      hc_scale.scalar_type() == at::ScalarType::Float &&
      hc_base.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(
      residual.dim() == 3 && residual.size(1) == 4 &&
      residual.size(2) == 4096);
  TORCH_CHECK(
      raw_mixes.dim() == 2 &&
      raw_mixes.size(0) == residual.size(0) && raw_mixes.size(1) == 24);
  TORCH_CHECK(
      rrms.dim() == 2 && rrms.size(0) == residual.size(0) &&
      rrms.size(1) == 1);
  TORCH_CHECK(hc_scale.dim() == 1 && hc_scale.size(0) == 3);
  TORCH_CHECK(hc_base.dim() == 1 && hc_base.size(0) == 24);
  TORCH_CHECK(
      residual.is_contiguous() && raw_mixes.is_contiguous() &&
      rrms.is_contiguous() && hc_scale.is_contiguous() &&
      hc_base.is_contiguous());
  TORCH_CHECK(kMHCPreEmitRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kPreEmitSchema);
  std::vector<c10::IValue> inputs{
      residual, raw_mixes, rrms, hc_scale, hc_base};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 3);
  return std::make_tuple(outputs.at(0), outputs.at(1), outputs.at(2));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre_emit_meta(
    const at::Tensor& residual,
    const at::Tensor& raw_mixes,
    const at::Tensor& rrms,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base) {
  (void)raw_mixes;
  (void)rrms;
  (void)hc_scale;
  (void)hc_base;
  const auto tokens = residual.size(0);
  const auto float_options = residual.options().dtype(at::ScalarType::Float);
  return std::make_tuple(
      at::empty({tokens, 4, 1}, float_options),
      at::empty({tokens, 16, 1}, float_options),
      at::empty(
          {tokens, 4096},
      residual.options().dtype(at::ScalarType::BFloat16)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre_emit_norm_hpu(
    const at::Tensor& residual,
    const at::Tensor& raw_mixes,
    const at::Tensor& rrms,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& norm_weight) {
  TORCH_CHECK(
      residual.scalar_type() == at::ScalarType::BFloat16 &&
      norm_weight.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      raw_mixes.scalar_type() == at::ScalarType::Float &&
      rrms.scalar_type() == at::ScalarType::Float &&
      hc_scale.scalar_type() == at::ScalarType::Float &&
      hc_base.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(
      residual.dim() == 3 && residual.size(1) == 4 &&
      residual.size(2) == 4096);
  TORCH_CHECK(
      raw_mixes.dim() == 2 &&
      raw_mixes.size(0) == residual.size(0) && raw_mixes.size(1) == 24);
  TORCH_CHECK(
      rrms.dim() == 2 && rrms.size(0) == residual.size(0) &&
      rrms.size(1) == 1);
  TORCH_CHECK(hc_scale.dim() == 1 && hc_scale.size(0) == 3);
  TORCH_CHECK(hc_base.dim() == 1 && hc_base.size(0) == 24);
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == 4096);
  TORCH_CHECK(
      residual.is_contiguous() && raw_mixes.is_contiguous() &&
      rrms.is_contiguous() && hc_scale.is_contiguous() &&
      hc_base.is_contiguous() && norm_weight.is_contiguous());
  TORCH_CHECK(kMHCPreEmitNormRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kPreEmitNormSchema);
  std::vector<c10::IValue> inputs{
      residual, raw_mixes, rrms, hc_scale, hc_base, norm_weight};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 3);
  return std::make_tuple(outputs.at(0), outputs.at(1), outputs.at(2));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre_emit_norm_meta(
    const at::Tensor& residual,
    const at::Tensor& raw_mixes,
    const at::Tensor& rrms,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& norm_weight) {
  (void)raw_mixes;
  (void)rrms;
  (void)hc_scale;
  (void)hc_base;
  (void)norm_weight;
  const auto tokens = residual.size(0);
  const auto float_options = residual.options().dtype(at::ScalarType::Float);
  return std::make_tuple(
      at::empty({tokens, 4, 1}, float_options),
      at::empty({tokens, 16, 1}, float_options),
      at::empty(
          {tokens, 4096},
          residual.options().dtype(at::ScalarType::BFloat16)));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def(
      "custom_deepseek_v4_mhc_post_prepare_gaudi2("
      "Tensor x, Tensor residual, Tensor post_layer_mix, "
      "Tensor comb_res_mix) -> (Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_mhc_pre_emit_gaudi2("
      "Tensor residual, Tensor raw_mixes, Tensor rrms, "
      "Tensor hc_scale, Tensor hc_base) -> (Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_mhc_pre_emit_norm_gaudi2("
      "Tensor residual, Tensor raw_mixes, Tensor rrms, "
      "Tensor hc_scale, Tensor hc_base, Tensor norm_weight) "
      "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl(
      "custom_deepseek_v4_mhc_post_prepare_gaudi2",
      mhc_post_prepare_hpu);
  m.impl(
      "custom_deepseek_v4_mhc_pre_emit_gaudi2",
      mhc_pre_emit_hpu);
  m.impl(
      "custom_deepseek_v4_mhc_pre_emit_norm_gaudi2",
      mhc_pre_emit_norm_hpu);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl(
      "custom_deepseek_v4_mhc_post_prepare_gaudi2",
      mhc_post_prepare_meta);
  m.impl(
      "custom_deepseek_v4_mhc_pre_emit_gaudi2",
      mhc_pre_emit_meta);
  m.impl(
      "custom_deepseek_v4_mhc_pre_emit_norm_gaudi2",
      mhc_pre_emit_norm_meta);
}
