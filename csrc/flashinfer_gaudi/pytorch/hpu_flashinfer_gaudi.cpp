// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* kSchema = "flashinfer_gaudi::gdn_decode_packed";
constexpr const char* kGuid = "flashinfer_gaudi_gdn_decode_packed_f32_gaudi2";

bool register_gdn_decode_packed() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& state = inputs.at(0).toTensor();
    const auto& packed = inputs.at(1).toTensor();

    habana::PartialOutputMetaData state_meta;
    state_meta.dtype = state.scalar_type();
    state_meta.shape = state.sizes().vec();
    state_meta.shape.at(0) = packed.size(0);

    habana::PartialOutputMetaData output_meta;
    output_meta.dtype = state.scalar_type();
    output_meta.shape = state_meta.shape;
    output_meta.shape.pop_back();
    return habana::PartialOutputMetaDataVector{state_meta, output_meta};
  };

  habana::custom_op::registerUserCustomOp(kSchema, kGuid, output_meta, nullptr);
  return true;
}

const bool kRegistered = register_gdn_decode_packed();

std::tuple<at::Tensor, at::Tensor> gdn_decode_packed_hpu(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& decay,
    const at::Tensor& beta,
    const at::Tensor& state_indices) {
  TORCH_CHECK(state_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(packed_qkv.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(decay.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(beta.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(state_cache.dim() == 4 && state_cache.size(-1) == 128 && state_cache.size(-2) == 128);
  TORCH_CHECK(packed_qkv.dim() == 2);

  const auto batch = packed_qkv.size(0);
  const auto value_heads = state_cache.size(1);
  const auto packed_width = packed_qkv.size(1);
  const auto key_heads = (packed_width - value_heads * 128) / 256;
  TORCH_CHECK(key_heads > 0 && value_heads % key_heads == 0);
  TORCH_CHECK(packed_width == 2 * key_heads * 128 + value_heads * 128);
  TORCH_CHECK(decay.sizes() == at::IntArrayRef({batch, value_heads}));
  TORCH_CHECK(beta.sizes() == decay.sizes());
  TORCH_CHECK(state_indices.dim() == 1 && state_indices.size(0) == batch);
  TORCH_CHECK(kRegistered);

  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kSchema);
  std::vector<c10::IValue> inputs{state_cache, packed_qkv, decay, beta, state_indices};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> gdn_decode_packed_meta(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto state_sizes = state_cache.sizes().vec();
  state_sizes.at(0) = packed_qkv.size(0);
  auto output_sizes = state_sizes;
  output_sizes.pop_back();
  return std::make_tuple(
      at::empty(state_sizes, state_cache.options()),
      at::empty(output_sizes, state_cache.options()));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(flashinfer_gaudi, m) {
  m.def(
      "gdn_decode_packed(Tensor(a!) state_cache, Tensor packed_qkv, Tensor decay, "
      "Tensor beta, Tensor state_indices) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flashinfer_gaudi, HPU, m) {
  m.impl("gdn_decode_packed", gdn_decode_packed_hpu);
}

TORCH_LIBRARY_IMPL(flashinfer_gaudi, Meta, m) {
  m.impl("gdn_decode_packed", gdn_decode_packed_meta);
}
