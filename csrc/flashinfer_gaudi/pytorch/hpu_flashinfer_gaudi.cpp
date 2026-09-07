// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* kSchema = "flashinfer_gaudi::gdn_decode_packed";
constexpr const char* kGuid = "flashinfer_gaudi_gdn_decode_packed_bf16_f32state_gaudi2";
constexpr const char* kMtpSchema = "flashinfer_gaudi::gdn_mtp_packed";
constexpr const char* kMtpGuid = "flashinfer_gaudi_gdn_mtp_packed_bf16_f32state_gaudi2";
// SynapseAI's graph compiler recognizes the CustomOp API namespace.
constexpr const char* kMtpPreparedSchema = "custom_op::flashinfer_gaudi_gdn_mtp_prepared";
constexpr const char* kMtpPreparedGuid = "flashinfer_gaudi_gdn_mtp_prepared_f32_gaudi2";
constexpr const char* kSelectPathSchema = "flashinfer_gaudi::dflash2_select_path";
constexpr const char* kSelectPathGuid = "flashinfer_gaudi_dflash2_select_path_i32_f32_gaudi2";
constexpr const char* kScoreSelectSchema = "flashinfer_gaudi::dflash2_score_select";
constexpr const char* kScoreSelectGuid =
    "flashinfer_gaudi_dflash2_score_select_i32_bf16_f32_gaudi2";
constexpr int64_t kMtpTokens = 8;
constexpr int64_t kMtpKeyHeads = 16;
constexpr int64_t kMtpValueHeads = 48;
constexpr int64_t kMtpDim = 128;
constexpr int64_t kMtpPackedWidth =
    (2 * kMtpKeyHeads + kMtpValueHeads) * kMtpDim;
constexpr int64_t kMtpMaxBatch = 16;
constexpr int64_t kSelectPathSteps = 7;
constexpr int64_t kSelectPathTopK = 16;
constexpr int64_t kSelectPathMaxBatch = 16;
constexpr int64_t kScoreSelectVocab = 248320;
constexpr int64_t kScoreSelectRank = 256;

bool register_gdn_decode_packed() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& state = inputs.at(0).toTensor();
    const auto& packed = inputs.at(1).toTensor();

    habana::PartialOutputMetaData output_meta;
    output_meta.dtype = packed.scalar_type();
    output_meta.shape = state.sizes().vec();
    output_meta.shape.at(0) = packed.size(0);
    output_meta.shape.pop_back();
    return habana::PartialOutputMetaDataVector{output_meta};
  };

  habana::custom_op::registerUserCustomOp(kSchema, kGuid, output_meta, nullptr);
  return true;
}

const bool kRegistered = register_gdn_decode_packed();

bool register_gdn_mtp_packed() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& state = inputs.at(0).toTensor();
    const auto& packed = inputs.at(1).toTensor();

    habana::PartialOutputMetaData output_meta;
    output_meta.dtype = packed.scalar_type();
    output_meta.shape = {
        packed.size(0), packed.size(1), state.size(1), state.size(2)};
    return habana::PartialOutputMetaDataVector{output_meta};
  };

  habana::custom_op::registerUserCustomOp(kMtpSchema, kMtpGuid, output_meta, nullptr);
  return true;
}

const bool kMtpRegistered = register_gdn_mtp_packed();

bool register_gdn_mtp_prepared() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& state = inputs.at(0).toTensor();
    const auto& packed = inputs.at(1).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Float;
    output.shape = {packed.size(0), packed.size(1), state.size(1), state.size(2)};
    habana::PartialOutputMetaData checkpoints;
    checkpoints.dtype = at::ScalarType::Float;
    checkpoints.shape = {packed.size(0), packed.size(1), state.size(1), state.size(2), state.size(3)};
    return habana::PartialOutputMetaDataVector{output, checkpoints};
  };
  habana::custom_op::registerUserCustomOp(kMtpPreparedSchema, kMtpPreparedGuid, output_meta, nullptr);
  return true;
}

const bool kMtpPreparedRegistered = register_gdn_mtp_prepared();

bool register_dflash2_select_path() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& candidates = inputs.at(0).toTensor();

    habana::PartialOutputMetaData output_meta;
    output_meta.dtype = candidates.scalar_type();
    output_meta.shape = {candidates.size(0), candidates.size(1)};
    return habana::PartialOutputMetaDataVector{output_meta};
  };

  habana::custom_op::registerUserCustomOp(
      kSelectPathSchema, kSelectPathGuid, output_meta, nullptr);
  return true;
}

const bool kSelectPathRegistered = register_dflash2_select_path();

bool register_dflash2_score_select() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& candidates = inputs.at(2).toTensor();

    habana::PartialOutputMetaData output_meta;
    output_meta.dtype = candidates.scalar_type();
    output_meta.shape = {candidates.size(0), candidates.size(1)};
    return habana::PartialOutputMetaDataVector{output_meta};
  };

  habana::custom_op::registerUserCustomOp(
      kScoreSelectSchema, kScoreSelectGuid, output_meta, nullptr);
  return true;
}

const bool kScoreSelectRegistered = register_dflash2_score_select();

at::Tensor gdn_decode_packed_hpu(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& decay,
    const at::Tensor& beta,
    const at::Tensor& state_indices) {
  TORCH_CHECK(state_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(packed_qkv.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(decay.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(beta.scalar_type() == at::ScalarType::BFloat16);
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
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor gdn_decode_packed_meta(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  auto output_sizes = state_cache.sizes().vec();
  output_sizes.at(0) = packed_qkv.size(0);
  output_sizes.pop_back();
  return at::empty(output_sizes, packed_qkv.options());
}

at::Tensor gdn_mtp_packed_hpu(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& decay,
    const at::Tensor& beta,
    const at::Tensor& state_indices,
    const at::Tensor& num_accepted_tokens,
    const at::Tensor& query_lengths) {
  TORCH_CHECK(state_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(packed_qkv.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(decay.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(beta.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(state_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(num_accepted_tokens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(query_lengths.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      state_cache.dim() == 4 && state_cache.size(1) == kMtpValueHeads &&
      state_cache.size(2) == kMtpDim && state_cache.size(3) == kMtpDim &&
      state_cache.size(0) > 0);
  TORCH_CHECK(
      packed_qkv.dim() == 3 && packed_qkv.size(1) == kMtpTokens &&
      packed_qkv.size(2) == kMtpPackedWidth);

  const auto batch = packed_qkv.size(0);
  const auto tokens = packed_qkv.size(1);
  const auto value_heads = state_cache.size(1);
  const auto packed_width = packed_qkv.size(2);
  const auto key_heads = (packed_width - value_heads * kMtpDim) / (2 * kMtpDim);
  TORCH_CHECK(batch > 0 && batch <= kMtpMaxBatch);
  TORCH_CHECK(key_heads == kMtpKeyHeads && value_heads % key_heads == 0);
  TORCH_CHECK(
      packed_width == 2 * key_heads * kMtpDim + value_heads * kMtpDim);
  TORCH_CHECK(decay.sizes() == at::IntArrayRef({batch, tokens, value_heads}));
  TORCH_CHECK(beta.sizes() == decay.sizes());
  TORCH_CHECK(state_indices.sizes() == at::IntArrayRef({batch, tokens}));
  TORCH_CHECK(num_accepted_tokens.dim() == 1 && num_accepted_tokens.size(0) == batch);
  TORCH_CHECK(query_lengths.dim() == 1 && query_lengths.size(0) == batch);
  TORCH_CHECK(
      state_cache.is_contiguous() && packed_qkv.is_contiguous() &&
      decay.is_contiguous() && beta.is_contiguous() &&
      state_indices.is_contiguous() && num_accepted_tokens.is_contiguous() &&
      query_lengths.is_contiguous());
  TORCH_CHECK(kMtpRegistered);

  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMtpSchema);
  std::vector<c10::IValue> inputs{
      state_cache,
      packed_qkv,
      decay,
      beta,
      state_indices,
      num_accepted_tokens,
      query_lengths};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor gdn_mtp_packed_meta(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  return at::empty(
      {packed_qkv.size(0), packed_qkv.size(1), state_cache.size(1), state_cache.size(2)},
      packed_qkv.options());
}

void check_gdn_mtp_prepared_inputs(
    const at::Tensor& initial_state,
    const at::Tensor& prepared_qkv,
    const at::Tensor& decay,
    const at::Tensor& beta) {
  TORCH_CHECK(initial_state.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(prepared_qkv.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(decay.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(beta.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(initial_state.dim() == 4 && initial_state.size(1) == kMtpValueHeads &&
              initial_state.size(2) == kMtpDim && initial_state.size(3) == kMtpDim);
  TORCH_CHECK(prepared_qkv.dim() == 3 && prepared_qkv.size(1) == kMtpTokens &&
              prepared_qkv.size(2) == kMtpPackedWidth);
  const auto batch = initial_state.size(0);
  TORCH_CHECK(batch > 0 && batch <= kMtpMaxBatch && prepared_qkv.size(0) == batch);
  TORCH_CHECK(decay.sizes() == at::IntArrayRef({batch, kMtpTokens, kMtpValueHeads}));
  TORCH_CHECK(beta.sizes() == decay.sizes());
  TORCH_CHECK(initial_state.is_contiguous() && prepared_qkv.is_contiguous() &&
              decay.is_contiguous() && beta.is_contiguous());
  TORCH_CHECK(initial_state.device() == prepared_qkv.device() &&
              initial_state.device() == decay.device() && initial_state.device() == beta.device());
}

std::tuple<at::Tensor, at::Tensor> gdn_mtp_prepared_hpu(
    const at::Tensor& initial_state,
    const at::Tensor& prepared_qkv,
    const at::Tensor& decay,
    const at::Tensor& beta) {
  check_gdn_mtp_prepared_inputs(initial_state, prepared_qkv, decay, beta);
  TORCH_CHECK(kMtpPreparedRegistered);
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kMtpPreparedSchema);
  std::vector<c10::IValue> inputs{initial_state, prepared_qkv, decay, beta};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> gdn_mtp_prepared_meta(
    const at::Tensor& initial_state,
    const at::Tensor& prepared_qkv,
    const at::Tensor& decay,
    const at::Tensor& beta) {
  check_gdn_mtp_prepared_inputs(initial_state, prepared_qkv, decay, beta);
  const auto batch = initial_state.size(0);
  return {
      at::empty({batch, kMtpTokens, kMtpValueHeads, kMtpDim}, prepared_qkv.options()),
      at::empty({batch, kMtpTokens, kMtpValueHeads, kMtpDim, kMtpDim}, initial_state.options())};
}

at::Tensor dflash2_select_path_hpu(
    const at::Tensor& candidate_ids,
    const at::Tensor& scores) {
  TORCH_CHECK(
      candidate_ids.scalar_type() == at::ScalarType::Long ||
      candidate_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(candidate_ids.dim() == 3);

  const auto batch = candidate_ids.size(0);
  TORCH_CHECK(batch > 0 && batch <= kSelectPathMaxBatch);
  TORCH_CHECK(
      candidate_ids.sizes() ==
      at::IntArrayRef({batch, kSelectPathSteps, kSelectPathTopK}));
  TORCH_CHECK(
      scores.sizes() == at::IntArrayRef(
          {batch, kSelectPathSteps, kSelectPathTopK, kSelectPathTopK}));
  TORCH_CHECK(candidate_ids.device() == scores.device());
  TORCH_CHECK(candidate_ids.is_contiguous() && scores.is_contiguous());
  TORCH_CHECK(kSelectPathRegistered);

  const bool restore_i64 = candidate_ids.scalar_type() == at::ScalarType::Long;
  const auto candidate_ids_i32 =
      restore_i64 ? candidate_ids.to(at::ScalarType::Int) : candidate_ids;
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kSelectPathSchema);
  std::vector<c10::IValue> inputs{candidate_ids_i32, scores};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return restore_i64 ? outputs.at(0).to(at::ScalarType::Long) : outputs.at(0);
}

at::Tensor dflash2_select_path_meta(
    const at::Tensor& candidate_ids,
    const at::Tensor&) {
  return at::empty(
      {candidate_ids.size(0), candidate_ids.size(1)}, candidate_ids.options());
}

at::Tensor dflash2_score_select_hpu(
    const at::Tensor& predecessor_table,
    const at::Tensor& successor_table,
    const at::Tensor& candidate_ids,
    const at::Tensor& unary_logits,
    const at::Tensor& hidden,
    const at::Tensor& anchor_token_ids) {
  TORCH_CHECK(predecessor_table.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(successor_table.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      candidate_ids.scalar_type() == at::ScalarType::Long ||
      candidate_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(unary_logits.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(hidden.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(anchor_token_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(candidate_ids.dim() == 3);

  const auto batch = candidate_ids.size(0);
  TORCH_CHECK(batch > 0 && batch <= kSelectPathMaxBatch);
  TORCH_CHECK(
      predecessor_table.sizes() ==
      at::IntArrayRef({kScoreSelectVocab, kScoreSelectRank}));
  TORCH_CHECK(successor_table.sizes() == predecessor_table.sizes());
  TORCH_CHECK(
      candidate_ids.sizes() ==
      at::IntArrayRef({batch, kSelectPathSteps, kSelectPathTopK}));
  TORCH_CHECK(unary_logits.sizes() == candidate_ids.sizes());
  TORCH_CHECK(
      hidden.sizes() ==
      at::IntArrayRef({batch, kSelectPathSteps, kScoreSelectRank}));
  TORCH_CHECK(
      anchor_token_ids.dim() == 1 && anchor_token_ids.size(0) == batch);
  TORCH_CHECK(
      predecessor_table.device() == candidate_ids.device() &&
      successor_table.device() == candidate_ids.device() &&
      unary_logits.device() == candidate_ids.device() &&
      hidden.device() == candidate_ids.device() &&
      anchor_token_ids.device() == candidate_ids.device());
  TORCH_CHECK(
      predecessor_table.is_contiguous() && successor_table.is_contiguous() &&
      candidate_ids.is_contiguous() && unary_logits.is_contiguous() &&
      hidden.is_contiguous() && anchor_token_ids.is_contiguous());
  TORCH_CHECK(kScoreSelectRegistered);

  const bool restore_i64 = candidate_ids.scalar_type() == at::ScalarType::Long;
  const auto candidate_ids_i32 =
      restore_i64 ? candidate_ids.to(at::ScalarType::Int) : candidate_ids;
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kScoreSelectSchema);
  std::vector<c10::IValue> inputs{
      predecessor_table,
      successor_table,
      candidate_ids_i32,
      unary_logits,
      hidden,
      anchor_token_ids};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return restore_i64 ? outputs.at(0).to(at::ScalarType::Long) : outputs.at(0);
}

at::Tensor dflash2_score_select_meta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& candidate_ids,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  return at::empty(
      {candidate_ids.size(0), candidate_ids.size(1)}, candidate_ids.options());
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(flashinfer_gaudi, m) {
  m.def(
      "gdn_decode_packed(Tensor(a!) state_cache, Tensor packed_qkv, Tensor decay, "
      "Tensor beta, Tensor state_indices) -> Tensor");
  m.def(
      "gdn_mtp_packed(Tensor(a!) state_cache, Tensor packed_qkv, Tensor decay, "
      "Tensor beta, Tensor state_indices, Tensor num_accepted_tokens, "
      "Tensor query_lengths) -> Tensor");
  m.def("dflash2_select_path(Tensor candidate_ids, Tensor scores) -> Tensor");
  m.def(
      "dflash2_score_select(Tensor predecessor_table, Tensor successor_table, "
      "Tensor candidate_ids, Tensor unary_logits, Tensor hidden, "
      "Tensor anchor_token_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(flashinfer_gaudi, HPU, m) {
  m.impl("gdn_decode_packed", gdn_decode_packed_hpu);
  m.impl("gdn_mtp_packed", gdn_mtp_packed_hpu);
  m.impl("dflash2_select_path", dflash2_select_path_hpu);
  m.impl("dflash2_score_select", dflash2_score_select_hpu);
}

TORCH_LIBRARY_IMPL(flashinfer_gaudi, Meta, m) {
  m.impl("gdn_decode_packed", gdn_decode_packed_meta);
  m.impl("gdn_mtp_packed", gdn_mtp_packed_meta);
  m.impl("dflash2_select_path", dflash2_select_path_meta);
  m.impl("dflash2_score_select", dflash2_score_select_meta);
}

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def(
      "flashinfer_gaudi_gdn_mtp_prepared(Tensor initial_state, Tensor prepared_qkv, "
      "Tensor decay, Tensor beta) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("flashinfer_gaudi_gdn_mtp_prepared", gdn_mtp_prepared_hpu);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("flashinfer_gaudi_gdn_mtp_prepared", gdn_mtp_prepared_meta);
}
