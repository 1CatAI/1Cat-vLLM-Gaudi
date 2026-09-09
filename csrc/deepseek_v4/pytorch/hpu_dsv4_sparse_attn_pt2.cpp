#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {

constexpr const char* kSchema =
    "custom_op::custom_deepseek_v4_sparse_attn_bf16_gaudi2";
constexpr const char* kGuid =
    "custom_deepseek_v4_sparse_attn_bf16_gaudi2";
constexpr const char* kLengthsSchema =
    "custom_op::custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2";
constexpr const char* kLengthsGuid =
    "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2";
constexpr const char* kDequantSchema =
    "custom_op::custom_deepseek_v4_dequant_gather_bf16_gaudi2";
constexpr const char* kDequantGuid =
    "custom_deepseek_v4_dequant_gather_bf16_gaudi2";
constexpr const char* kDualDequantSchema =
    "custom_op::custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2";
constexpr const char* kDualDequantGuid =
    "custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2";
constexpr const char* kLocalDualDequantSchema =
    "custom_op::custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2";
constexpr const char* kLocalDualDequantGuid =
    "custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2";
constexpr const char* kSavePartialStatesSchema =
    "custom_op::custom_deepseek_v4_save_partial_states_f32_gaudi2";
constexpr const char* kSavePartialStatesGuid =
    "custom_deepseek_v4_save_partial_states_f32_gaudi2";
constexpr const char* kSaveCompressNormC4Schema =
    "custom_op::custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2";
constexpr const char* kSaveCompressNormC4Guid =
    "custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2";
constexpr const char* kSaveCompressNormC4OrderedSchema =
    "custom_op::custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2";
constexpr const char* kSaveCompressNormC4OrderedGuid =
    "custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2";
constexpr const char* kSaveCompressNormC4NoCloneSchema =
    "custom_op::custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2";
constexpr const char* kSaveCompressNormC4BF16Schema =
    "custom_op::custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2";
constexpr const char* kSaveCompressNormC4BF16Guid =
    "custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2";
constexpr const char* kSaveCompressNormC4MixedSchema =
    "custom_op::custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2";
constexpr const char* kSaveCompressNormC4MixedGuid =
    "custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2";
constexpr const char* kQnormRopeKvPackSchema =
    "custom_op::custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2";
constexpr const char* kQnormRopeKvPackGuid =
    "custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2";
constexpr const char* kHybridQnormRopeKvPackSchema =
    "custom_op::custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2";
constexpr const char* kHybridQnormRopeKvPackGuid =
    "custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2";
constexpr const char* kInsertPackedKvSchema =
    "custom_op::custom_deepseek_v4_insert_packed_kv_u8_gaudi2";
constexpr const char* kInsertPackedKvGuid =
    "custom_deepseek_v4_insert_packed_kv_u8_gaudi2";
constexpr const char* kPagedSparseAttnSchema =
    "custom_op::custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2";
constexpr const char* kPagedSparseAttnGuid =
    "custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2";
constexpr const char* kPagedSparseAttnLocalSchema =
    "custom_op::custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2";
constexpr const char* kPagedSparseAttnLocalGuid =
    "custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2";
constexpr const char* kPagedSparseAttnSequentialSchema =
    "custom_op::custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2";
constexpr const char* kPagedSparseAttnSequentialGuid =
    "custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2";
constexpr const char* kPagedSparseAttnPairSequentialSchema =
    "custom_op::custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2";
constexpr const char* kPagedSparseAttnPairSequentialGuid =
    "custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2";
constexpr const char* kPagedSwaAttnSchema =
    "custom_op::custom_deepseek_v4_paged_swa_attn_fp8_gaudi2";
constexpr const char* kPagedSwaAttnGuid =
    "custom_deepseek_v4_paged_swa_attn_fp8_gaudi2";
constexpr const char* kFlashMLASplitKVPartialSchema =
    "custom_op::custom_deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2";
constexpr const char* kFlashMLASplitKVPartialGuid =
    "custom_deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2";
constexpr const char* kFlashMLASplitKVCombineSchema =
    "custom_op::custom_deepseek_v4_flashmla_splitkv_combine_gaudi2";
constexpr const char* kFlashMLASplitKVCombineGuid =
    "custom_deepseek_v4_flashmla_splitkv_combine_gaudi2";
constexpr const char* kFlashMLASplitKVSchema =
    "custom_op::custom_deepseek_v4_flashmla_splitkv_fp8_gaudi2";
constexpr const char* kFlashMLASplitKVTiledSchema =
    "custom_op::custom_deepseek_v4_flashmla_splitkv_tiled_fp8_gaudi2";
constexpr const char* kFlashMLASplitKVTiledPartialGuid =
    "custom_deepseek_v4_flashmla_splitkv_partial_tiled_fp8_gaudi2";
constexpr const char* kCompressFlashMLAC4Schema =
    "custom_op::custom_deepseek_v4_compress_flashmla_c4_f32_gaudi2";
constexpr const char* kQnormCompressorC4Schema =
    "custom_op::custom_deepseek_v4_qnorm_compressor_c4_f32_gaudi2";
constexpr const char* kQnormCompressorC4NoCloneSchema =
    "custom_op::custom_deepseek_v4_qnorm_compressor_c4_f32_noclone_gaudi2";
constexpr const char* kQnormPagedSparseAttnSequentialSchema =
    "custom_op::custom_deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2";
constexpr const char* kQnormPagedSparseAttnSequentialGuid =
    "custom_deepseek_v4_qnorm_paged_sparse_attn_seq_fp8_gaudi2";
constexpr const char* kFillShortTopkSchema =
    "custom_op::custom_deepseek_v4_fill_short_topk_i32_gaudi2";
constexpr const char* kFillShortTopkGuid =
    "custom_deepseek_v4_fill_short_topk_i32_gaudi2";
constexpr const char* kMxfp4GatherSchema =
    "custom_op::custom_deepseek_v4_mxfp4_gather_u8_gaudi2";
constexpr const char* kMxfp4GatherGuid =
    "custom_deepseek_v4_mxfp4_gather_u8_gaudi2";
constexpr const char* kMxfp4DequantFp8Schema =
    "custom_op::custom_deepseek_v4_mxfp4_dequant_fp8_gaudi2";
constexpr const char* kMxfp4DequantFp8Guid =
    "custom_deepseek_v4_mxfp4_dequant_fp8_gaudi2";
constexpr const char* kMxfp4IndexedMoeSchema =
    "custom_op::custom_deepseek_v4_mxfp4_indexed_moe_gaudi2";
constexpr const char* kMxfp4IndexedFc1Guid =
    "custom_deepseek_v4_mxfp4_indexed_fc1_gaudi2";
constexpr const char* kMxfp4IndexedFc2Guid =
    "custom_deepseek_v4_mxfp4_indexed_fc2_gaudi2";

bool register_sparse_attention() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = q.scalar_type();
    output.shape = q.sizes().vec();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{output, stats, stats};
  };

  habana::custom_op::registerUserCustomOp(
      kSchema,
      kGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kSparseAttentionRegistered = register_sparse_attention();

bool register_sparse_attention_lengths() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = q.scalar_type();
    output.shape = q.sizes().vec();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{output, stats, stats};
  };

  habana::custom_op::registerUserCustomOp(
      kLengthsSchema,
      kLengthsGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kSparseAttentionLengthsRegistered =
    register_sparse_attention_lengths();

bool register_dequant_gather() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& indices = inputs.at(2).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = {indices.size(0), 512};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kDequantSchema,
      kDequantGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kDequantGatherRegistered = register_dequant_gather();

bool register_dual_dequant_gather() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& first_indices = inputs.at(2).toTensor();
    const auto& second_indices = inputs.at(5).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = {first_indices.size(0) + second_indices.size(0), 512};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kDualDequantSchema,
      kDualDequantGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kDualDequantGatherRegistered = register_dual_dequant_gather();

bool register_local_dual_dequant_gather() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& first_shape_buffer = inputs.at(2).toTensor();
    const auto& second_indices = inputs.at(9).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = {
        first_shape_buffer.size(0) + second_indices.size(0), 512};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kLocalDualDequantSchema,
      kLocalDualDequantGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kLocalDualDequantGatherRegistered =
    register_local_dual_dequant_gather();

bool register_save_partial_states() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& kv = inputs.at(2).toTensor();
    const auto& slot_mapping = inputs.at(6).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Float;
    output.shape = {slot_mapping.size(0), kv.size(1) / 64};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kSavePartialStatesSchema,
      kSavePartialStatesGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kSavePartialStatesRegistered = register_save_partial_states();

bool register_save_compress_norm_c4() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& slot_mapping = inputs.at(7).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {slot_mapping.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kSaveCompressNormC4Schema,
      kSaveCompressNormC4Guid,
      output_meta,
      nullptr);
  return true;
}

const bool kSaveCompressNormC4Registered =
    register_save_compress_norm_c4();

bool register_save_compress_norm_c4_ordered() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& slot_mapping = inputs.at(7).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {slot_mapping.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kSaveCompressNormC4OrderedSchema,
      kSaveCompressNormC4Guid,
      output_meta,
      nullptr);
  return true;
}

const bool kSaveCompressNormC4OrderedRegistered =
    register_save_compress_norm_c4_ordered();

bool register_save_compress_norm_c4_bf16() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& slot_mapping = inputs.at(7).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {slot_mapping.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kSaveCompressNormC4BF16Schema,
      kSaveCompressNormC4BF16Guid,
      output_meta,
      nullptr);
  return true;
}

const bool kSaveCompressNormC4BF16Registered =
    register_save_compress_norm_c4_bf16();

bool register_save_compress_norm_c4_mixed() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& slot_mapping = inputs.at(7).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {slot_mapping.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kSaveCompressNormC4MixedSchema,
      kSaveCompressNormC4MixedGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kSaveCompressNormC4MixedRegistered =
    register_save_compress_norm_c4_mixed();

bool register_qnorm_rope_kv_pack() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {q.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kQnormRopeKvPackSchema,
      kQnormRopeKvPackGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kQnormRopeKvPackRegistered =
    register_qnorm_rope_kv_pack();

bool register_hybrid_qnorm_rope_kv_pack() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {q.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kHybridQnormRopeKvPackSchema,
      kHybridQnormRopeKvPackGuid,
      output_meta,
      nullptr);
  for (const auto* schema : {
           "custom_op::custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2",
           "custom_op::custom_deepseek_v4_native_qnorm_rope_kv_pack_ordered_bf16_gaudi2"}) {
    habana::custom_op::registerUserCustomOp(schema, kHybridQnormRopeKvPackGuid, output_meta, nullptr);
  }
  return true;
}

const bool kHybridQnormRopeKvPackRegistered =
    register_hybrid_qnorm_rope_kv_pack();


bool register_insert_packed_kv() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& packed = inputs.at(2).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Byte;
    output.shape = {packed.size(0)};
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::registerUserCustomOp(
      kInsertPackedKvSchema,
      kInsertPackedKvGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kInsertPackedKvRegistered = register_insert_packed_kv();

bool register_paged_sparse_attention() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{stats};
  };

  habana::custom_op::registerUserCustomOp(
      kPagedSparseAttnSchema,
      kPagedSparseAttnGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kPagedSparseAttentionRegistered =
    register_paged_sparse_attention();

bool register_paged_swa_attention() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{stats};
  };

  habana::custom_op::registerUserCustomOp(
      kPagedSwaAttnSchema,
      kPagedSwaAttnGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kPagedSwaAttentionRegistered =
    register_paged_swa_attention();

bool register_paged_sparse_attention_local() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{stats};
  };

  habana::custom_op::registerUserCustomOp(
      kPagedSparseAttnLocalSchema,
      kPagedSparseAttnLocalGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kPagedSparseAttentionLocalRegistered =
    register_paged_sparse_attention_local();

bool register_paged_sparse_attention_sequential() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{stats};
  };

  habana::custom_op::registerUserCustomOp(
      kPagedSparseAttnSequentialSchema,
      kPagedSparseAttnSequentialGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kPagedSparseAttentionSequentialRegistered =
    register_paged_sparse_attention_sequential();

bool register_paged_sparse_attention_pair_sequential() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{stats};
  };

  habana::custom_op::registerUserCustomOp(
      kPagedSparseAttnPairSequentialSchema,
      kPagedSparseAttnPairSequentialGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kPagedSparseAttentionPairSequentialRegistered =
    register_paged_sparse_attention_pair_sequential();

bool register_flashmla_splitkv_partial() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    const auto& split_shape = inputs.at(4).toTensor();
    const auto splits = split_shape.size(0);
    habana::PartialOutputMetaData partial_output;
    partial_output.dtype = at::ScalarType::Float;
    partial_output.shape = {q.size(0), splits, q.size(1), 512};
    habana::PartialOutputMetaData partial_stats;
    partial_stats.dtype = at::ScalarType::Float;
    partial_stats.shape = {q.size(0), splits, q.size(1), 2};
    return habana::PartialOutputMetaDataVector{partial_output, partial_stats};
  };

  habana::custom_op::registerUserCustomOp(
      kFlashMLASplitKVPartialSchema,
      kFlashMLASplitKVPartialGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kFlashMLASplitKVPartialRegistered =
    register_flashmla_splitkv_partial();

bool register_flashmla_splitkv_combine() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& partial_output = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = {
        partial_output.size(0), partial_output.size(2), 512};
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {partial_output.size(0), partial_output.size(2)};
    return habana::PartialOutputMetaDataVector{output, stats};
  };

  habana::custom_op::registerUserCustomOp(
      kFlashMLASplitKVCombineSchema,
      kFlashMLASplitKVCombineGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kFlashMLASplitKVCombineRegistered =
    register_flashmla_splitkv_combine();

class FlashMLASplitKVBackend final : public habana::OpBackend {
 public:
  FlashMLASplitKVBackend(
      int device_id,
      const habana::custom_op::UserCustomOpDescriptor& descriptor,
      const char* partial_guid)
      : OpBackend(
            device_id,
            partial_guid,
            c10::ScalarType::Undefined,
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(descriptor.getOutputMetaFn());
    partial_guid_ = partial_guid;
  }

 private:
  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto& q = stack.at(0).toTensor();
    const auto& split_shape = stack.at(4).toTensor();
    const auto batch_size = q.size(0);
    const auto split_count = split_shape.size(0);
    const auto head_count = q.size(1);
    const auto& output_template = stack.at(14).toTensor();
    const auto output_head_count = output_template.size(1);

    auto partial = BuildOp(
        graph,
        partial_guid_,
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         syn_in(3),
         syn_in(4),
         syn_in(5),
         syn_in(6),
         syn_in(7),
         syn_in(8),
         syn_in(9),
         syn_in(10),
         syn_in(11),
         syn_in(12)},
        {{{batch_size, split_count, head_count, 512},
          at::ScalarType::Float,
          2},
         {{batch_size, split_count, head_count, 2},
          at::ScalarType::Float,
          3}});

    auto combined = BuildOp(
        graph,
        kFlashMLASplitKVCombineGuid,
        {partial.at(0).get(),
         partial.at(1).get(),
         syn_in(13)},
        {{{batch_size, output_head_count, 512},
          at::ScalarType::BFloat16,
          0},
         {{batch_size, head_count}, at::ScalarType::Float, 1}});
    syn_out(0) = std::move(combined.at(0));
    syn_out(1) = std::move(combined.at(1));
    syn_out(2) = std::move(partial.at(0));
    syn_out(3) = std::move(partial.at(1));
  }

  const char* partial_guid_;
};

bool register_flashmla_splitkv_backend(
    const char* schema,
    const char* partial_guid) {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    const auto& output_template = inputs.at(14).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = output_template.sizes().vec();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    const auto split_count = inputs.at(4).toTensor().size(0);
    habana::PartialOutputMetaData partial_output;
    partial_output.dtype = at::ScalarType::Float;
    partial_output.shape = {
        q.size(0), split_count, q.size(1), 512};
    habana::PartialOutputMetaData partial_stats;
    partial_stats.dtype = at::ScalarType::Float;
    partial_stats.shape = {
        q.size(0), split_count, q.size(1), 2};
    return habana::PartialOutputMetaDataVector{
        output, stats, partial_output, partial_stats};
  };

  habana::custom_op::UserCustomOpDescriptor descriptor{
      schema,
      partial_guid,
      output_meta,
      nullptr};
  habana::KernelRegistry().add_user_custom_op(
      [partial_guid](const int device_id, const std::string& schema_name) {
        auto& registered =
            habana::KernelRegistry().get_user_custom_op_desc(schema_name);
        return std::make_shared<FlashMLASplitKVBackend>(
            device_id, registered, partial_guid);
      },
      descriptor);
  return true;
}

const bool kFlashMLASplitKVBackendRegistered =
    register_flashmla_splitkv_backend(
        kFlashMLASplitKVSchema, kFlashMLASplitKVPartialGuid);
const bool kFlashMLASplitKVTiledBackendRegistered =
    register_flashmla_splitkv_backend(
        kFlashMLASplitKVTiledSchema,
        kFlashMLASplitKVTiledPartialGuid);

class CompressFlashMLAC4Backend final : public habana::OpBackend {
 public:
  CompressFlashMLAC4Backend(
      int device_id,
      const habana::custom_op::UserCustomOpDescriptor& descriptor)
      : OpBackend(
            device_id,
            kSaveCompressNormC4OrderedGuid,
            c10::ScalarType::Undefined,
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(descriptor.getOutputMetaFn());
  }

 private:
  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto& q = stack.at(14).toTensor();
    const auto& split_shape = stack.at(16).toTensor();
    const auto batch_size = q.size(0);
    const auto split_count = split_shape.size(0);
    const auto head_count = q.size(1);
    const auto& output_template = stack.at(26).toTensor();
    const auto output_head_count = output_template.size(1);

    // The ordered compressor returns the original int32 validity metadata.
    // Feeding that tensor to FlashMLA gives the cache write and cache read a
    // direct graph edge without a separate conversion/where recipe.
    auto completion = BuildOp(
        graph,
        kSaveCompressNormC4OrderedGuid,
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         syn_in(3),
         syn_in(4),
         syn_in(5),
         syn_in(6),
         syn_in(7),
         syn_in(8),
         syn_in(9),
         syn_in(10),
         syn_in(11),
         syn_in(12),
         syn_in(13),
         syn_in(19)},
        {{{batch_size}, at::ScalarType::Int}});

    auto partial = BuildOp(
        graph,
        kFlashMLASplitKVTiledPartialGuid,
        {syn_in(14),
         syn_in(0),
         syn_in(2),
         syn_in(15),
         syn_in(16),
         syn_in(17),
         syn_in(18),
         completion.at(0).get(),
         syn_in(20),
         syn_in(21),
         syn_in(22),
         syn_in(23),
         syn_in(24)},
        {{{batch_size, split_count, head_count, 512},
          at::ScalarType::Float,
          2},
         {{batch_size, split_count, head_count, 2},
          at::ScalarType::Float,
          3}});

    auto combined = BuildOp(
        graph,
        kFlashMLASplitKVCombineGuid,
        {partial.at(0).get(), partial.at(1).get(), syn_in(25)},
        {{{batch_size, output_head_count, 512},
          at::ScalarType::BFloat16,
          0},
         {{batch_size, head_count}, at::ScalarType::Float, 1}});
    syn_out(0) = std::move(combined.at(0));
    syn_out(1) = std::move(combined.at(1));
    syn_out(2) = std::move(partial.at(0));
    syn_out(3) = std::move(partial.at(1));
  }
};

bool register_compress_flashmla_c4_backend() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(14).toTensor();
    const auto& split_shape = inputs.at(16).toTensor();
    const auto& output_template = inputs.at(26).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = output_template.sizes().vec();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    habana::PartialOutputMetaData partial_output;
    partial_output.dtype = at::ScalarType::Float;
    partial_output.shape = {
        q.size(0), split_shape.size(0), q.size(1), 512};
    habana::PartialOutputMetaData partial_stats;
    partial_stats.dtype = at::ScalarType::Float;
    partial_stats.shape = {
        q.size(0), split_shape.size(0), q.size(1), 2};
    return habana::PartialOutputMetaDataVector{
        output, stats, partial_output, partial_stats};
  };

  habana::custom_op::UserCustomOpDescriptor descriptor{
      kCompressFlashMLAC4Schema,
      kSaveCompressNormC4OrderedGuid,
      output_meta,
      nullptr};
  habana::KernelRegistry().add_user_custom_op(
      [](const int device_id, const std::string& schema_name) {
        auto& registered =
            habana::KernelRegistry().get_user_custom_op_desc(schema_name);
        return std::make_shared<CompressFlashMLAC4Backend>(
            device_id, registered);
      },
      descriptor);
  return true;
}

const bool kCompressFlashMLAC4BackendRegistered =
    register_compress_flashmla_c4_backend();

class QnormCompressorC4Backend final : public habana::OpBackend {
 public:
  QnormCompressorC4Backend(
      int device_id,
      const habana::custom_op::UserCustomOpDescriptor& descriptor)
      : OpBackend(
            device_id,
            kHybridQnormRopeKvPackGuid,
            c10::ScalarType::Undefined,
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(descriptor.getOutputMetaFn());
  }

 private:
  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto token_count = stack.at(0).toTensor().size(0);
    auto qnorm_completion = BuildOp(
        graph,
        kHybridQnormRopeKvPackGuid,
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         syn_in(3),
         syn_in(4),
         syn_in(5),
         syn_in(6)},
        {{{token_count}, at::ScalarType::Byte, 0}});
    auto compressor_completion = BuildOp(
        graph,
        kSaveCompressNormC4OrderedGuid,
        {syn_in(7),
         syn_in(8),
         syn_in(9),
         syn_in(10),
         syn_in(11),
         syn_in(12),
         syn_in(5),
         syn_in(13),
         syn_in(14),
         syn_in(15),
         syn_in(16),
         syn_in(17),
         syn_in(6),
         syn_in(18),
         syn_in(19)},
        {{{token_count}, at::ScalarType::Int, 1}});
    syn_out(0) = std::move(qnorm_completion.at(0));
    syn_out(1) = std::move(compressor_completion.at(0));
  }
};

bool register_qnorm_compressor_c4_backend(const char* schema) {
  auto output_meta = [](const at::Stack& inputs) {
    const auto token_count = inputs.at(0).toTensor().size(0);
    habana::PartialOutputMetaData qnorm_completion;
    qnorm_completion.dtype = at::ScalarType::Byte;
    qnorm_completion.shape = {token_count};
    habana::PartialOutputMetaData compressor_completion;
    compressor_completion.dtype = at::ScalarType::Int;
    compressor_completion.shape = {token_count};
    return habana::PartialOutputMetaDataVector{
        qnorm_completion, compressor_completion};
  };

  habana::custom_op::UserCustomOpDescriptor descriptor{
      schema,
      kHybridQnormRopeKvPackGuid,
      output_meta,
      nullptr};
  habana::KernelRegistry().add_user_custom_op(
      [](const int device_id, const std::string& schema_name) {
        auto& registered =
            habana::KernelRegistry().get_user_custom_op_desc(schema_name);
        return std::make_shared<QnormCompressorC4Backend>(
            device_id, registered);
      },
      descriptor);
  return true;
}

const bool kQnormCompressorC4BackendRegistered =
    register_qnorm_compressor_c4_backend(kQnormCompressorC4Schema);
const bool kQnormCompressorC4NoCloneBackendRegistered =
    register_qnorm_compressor_c4_backend(kQnormCompressorC4NoCloneSchema);

bool register_qnorm_paged_sparse_attention_sequential() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    habana::PartialOutputMetaData stats;
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{stats};
  };

  habana::custom_op::registerUserCustomOp(
      kQnormPagedSparseAttnSequentialSchema,
      kQnormPagedSparseAttnSequentialGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kQnormPagedSparseAttentionSequentialRegistered =
    register_qnorm_paged_sparse_attention_sequential();

bool register_fill_short_topk() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& output = inputs.at(0).toTensor();
    habana::PartialOutputMetaData counts;
    counts.dtype = at::ScalarType::Int;
    counts.shape = {output.size(0)};
    return habana::PartialOutputMetaDataVector{counts};
  };

  habana::custom_op::registerUserCustomOp(
      kFillShortTopkSchema,
      kFillShortTopkGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kFillShortTopkRegistered = register_fill_short_topk();

bool register_mxfp4_gather() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& ids = inputs.at(0).toTensor();
    const auto& w13 = inputs.at(1).toTensor();
    const auto& w2 = inputs.at(2).toTensor();
    const auto& w13_scale = inputs.at(3).toTensor();
    const auto& w2_scale = inputs.at(4).toTensor();
    const auto topk = ids.size(1);
    habana::PartialOutputMetaData output_w13;
    output_w13.dtype = at::ScalarType::Byte;
    output_w13.shape = {topk, w13.size(1), w13.size(2)};
    habana::PartialOutputMetaData output_w2;
    output_w2.dtype = at::ScalarType::Byte;
    output_w2.shape = {topk, w2.size(1), w2.size(2)};
    habana::PartialOutputMetaData output_w13_scale;
    output_w13_scale.dtype = at::ScalarType::Byte;
    output_w13_scale.shape = {
        topk, w13_scale.size(1), w13_scale.size(2)};
    habana::PartialOutputMetaData output_w2_scale;
    output_w2_scale.dtype = at::ScalarType::Byte;
    output_w2_scale.shape = {topk, w2_scale.size(1), w2_scale.size(2)};
    return habana::PartialOutputMetaDataVector{
        output_w13, output_w2, output_w13_scale, output_w2_scale};
  };

  habana::custom_op::registerUserCustomOp(
      kMxfp4GatherSchema,
      kMxfp4GatherGuid,
      output_meta,
      nullptr);
  return true;
}

const bool kMxfp4GatherRegistered = register_mxfp4_gather();

bool register_mxfp4_dequant_fp8() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& ids = inputs.at(0).toTensor();
    const auto& w13 = inputs.at(1).toTensor();
    const auto& w2 = inputs.at(2).toTensor();
    const auto topk = ids.size(1);
    habana::PartialOutputMetaData output_w13;
    output_w13.dtype = at::ScalarType::Float8_e4m3fn;
    output_w13.shape = {topk, w13.size(1), w13.size(2) * 2};
    habana::PartialOutputMetaData output_w2;
    output_w2.dtype = at::ScalarType::Float8_e4m3fn;
    output_w2.shape = {topk, w2.size(1), w2.size(2) * 2};
    return habana::PartialOutputMetaDataVector{output_w13, output_w2};
  };

  habana::custom_op::registerUserCustomOp(
      kMxfp4DequantFp8Schema,
      kMxfp4DequantFp8Guid,
      output_meta,
      nullptr);
  return true;
}

const bool kMxfp4DequantFp8Registered = register_mxfp4_dequant_fp8();

class Mxfp4IndexedMoeBackend final : public habana::OpBackend {
 public:
  Mxfp4IndexedMoeBackend(
      int device_id,
      const habana::custom_op::UserCustomOpDescriptor& descriptor)
      : OpBackend(
            device_id,
            kMxfp4IndexedFc1Guid,
            c10::ScalarType::Undefined,
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn(descriptor.getOutputMetaFn());
  }

 private:
  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto& hidden_states = stack.at(0).toTensor();
    const auto topk = stack.at(1).toTensor().size(1);

    auto intermediate = BuildOp(
        graph,
        kMxfp4IndexedFc1Guid,
        {syn_in(0), syn_in(1), syn_in(3), syn_in(5)},
        {{{topk, 1024}, at::ScalarType::BFloat16}});
    auto output = BuildOp(
        graph,
        kMxfp4IndexedFc2Guid,
        {intermediate.at(0).get(),
         syn_in(1),
         syn_in(2),
         syn_in(4),
         syn_in(6)},
        {{{hidden_states.size(0), hidden_states.size(1)},
          at::ScalarType::BFloat16,
          0}});
    syn_out(0) = std::move(output.at(0));
  }
};

bool register_mxfp4_indexed_moe_backend() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& hidden_states = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = hidden_states.sizes().vec();
    return habana::PartialOutputMetaDataVector{output};
  };

  habana::custom_op::UserCustomOpDescriptor descriptor{
      kMxfp4IndexedMoeSchema,
      kMxfp4IndexedFc1Guid,
      output_meta,
      nullptr};
  habana::KernelRegistry().add_user_custom_op(
      [](const int device_id, const std::string& schema_name) {
        auto& registered =
            habana::KernelRegistry().get_user_custom_op_desc(schema_name);
        return std::make_shared<Mxfp4IndexedMoeBackend>(
            device_id, registered);
      },
      descriptor);
  return true;
}

const bool kMxfp4IndexedMoeBackendRegistered =
    register_mxfp4_indexed_moe_backend();

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_attention_hpu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const at::Tensor& attn_sink,
    const at::Tensor& softmax_scale) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(kv.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(softmax_scale.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512);
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512);
  TORCH_CHECK(
      indices.dim() == 2 && indices.size(0) == q.size(0));
  TORCH_CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == q.size(1));
  TORCH_CHECK(softmax_scale.numel() == 1);
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(kv.is_contiguous());
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(attn_sink.is_contiguous());
  TORCH_CHECK(softmax_scale.is_contiguous());
  TORCH_CHECK(kSparseAttentionRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kSchema);
  std::vector<c10::IValue> inputs{
      q, kv, indices, attn_sink, softmax_scale};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 3);
  return std::make_tuple(outputs.at(0), outputs.at(1), outputs.at(2));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_attention_meta(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const at::Tensor& attn_sink,
    const at::Tensor& softmax_scale) {
  (void)kv;
  (void)indices;
  (void)attn_sink;
  (void)softmax_scale;
  auto stats_sizes = q.sizes().vec();
  stats_sizes.pop_back();
  auto stats = at::empty(stats_sizes, q.options().dtype(at::ScalarType::Float));
  return std::make_tuple(at::empty_like(q), stats, at::empty_like(stats));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_attention_lengths_hpu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const at::Tensor& attn_sink,
    const at::Tensor& softmax_scale,
    const at::Tensor& topk_lengths) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(kv.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(softmax_scale.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(topk_lengths.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512);
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == 512);
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == q.size(0));
  TORCH_CHECK(
      topk_lengths.dim() == 1 && topk_lengths.size(0) == q.size(0));
  TORCH_CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == q.size(1));
  TORCH_CHECK(softmax_scale.numel() == 1);
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(kv.is_contiguous());
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(attn_sink.is_contiguous());
  TORCH_CHECK(softmax_scale.is_contiguous());
  TORCH_CHECK(topk_lengths.is_contiguous());
  TORCH_CHECK(kSparseAttentionLengthsRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kLengthsSchema);
  std::vector<c10::IValue> inputs{
      q, kv, indices, attn_sink, softmax_scale, topk_lengths};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 3);
  return std::make_tuple(outputs.at(0), outputs.at(1), outputs.at(2));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
sparse_attention_lengths_meta(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    const at::Tensor& attn_sink,
    const at::Tensor& softmax_scale,
    const at::Tensor& topk_lengths) {
  (void)topk_lengths;
  return sparse_attention_meta(q, kv, indices, attn_sink, softmax_scale);
}

at::Tensor dequant_gather_hpu(
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& indices) {
  TORCH_CHECK(cache_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(cache_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(cache_storage_u8.dim() == 1 && cache_storage_u8.numel() > 0);
  TORCH_CHECK(cache_geometry.dim() == 1 && cache_geometry.numel() == 4);
  TORCH_CHECK(indices.dim() == 1 && indices.numel() > 0);
  TORCH_CHECK(cache_storage_u8.is_contiguous());
  TORCH_CHECK(cache_geometry.is_contiguous());
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(kDequantGatherRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kDequantSchema);
  std::vector<c10::IValue> inputs{
      cache_storage_u8, cache_geometry, indices};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor dequant_gather_meta(
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& indices) {
  (void)cache_storage_u8;
  (void)cache_geometry;
  return at::empty(
      {indices.size(0), 512},
      indices.options().dtype(at::ScalarType::BFloat16));
}

at::Tensor dual_dequant_gather_hpu(
    const at::Tensor& first_storage_u8,
    const at::Tensor& first_geometry,
    const at::Tensor& first_indices,
    const at::Tensor& second_storage_u8,
    const at::Tensor& second_geometry,
    const at::Tensor& second_indices) {
  TORCH_CHECK(
      first_storage_u8.scalar_type() == at::ScalarType::Byte &&
      second_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(
      first_geometry.scalar_type() == at::ScalarType::Int &&
      second_geometry.scalar_type() == at::ScalarType::Int &&
      first_indices.scalar_type() == at::ScalarType::Int &&
      second_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      first_storage_u8.dim() == 1 && first_storage_u8.numel() > 0 &&
      second_storage_u8.dim() == 1 && second_storage_u8.numel() > 0);
  TORCH_CHECK(
      first_geometry.dim() == 1 && first_geometry.numel() == 4 &&
      second_geometry.dim() == 1 && second_geometry.numel() == 4);
  TORCH_CHECK(
      first_indices.dim() == 1 && first_indices.numel() > 0 &&
      second_indices.dim() == 1 && second_indices.numel() > 0);
  TORCH_CHECK(
      first_storage_u8.is_contiguous() &&
      second_storage_u8.is_contiguous() && first_geometry.is_contiguous() &&
      second_geometry.is_contiguous() && first_indices.is_contiguous() &&
      second_indices.is_contiguous());
  TORCH_CHECK(kDualDequantGatherRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kDualDequantSchema);
  std::vector<c10::IValue> inputs{
      first_storage_u8,
      first_geometry,
      first_indices,
      second_storage_u8,
      second_geometry,
      second_indices};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor dual_dequant_gather_meta(
    const at::Tensor& first_storage_u8,
    const at::Tensor& first_geometry,
    const at::Tensor& first_indices,
    const at::Tensor& second_storage_u8,
    const at::Tensor& second_geometry,
    const at::Tensor& second_indices) {
  (void)first_storage_u8;
  (void)first_geometry;
  (void)second_storage_u8;
  (void)second_geometry;
  return at::empty(
      {first_indices.size(0) + second_indices.size(0), 512},
      first_indices.options().dtype(at::ScalarType::BFloat16));
}

at::Tensor local_dual_dequant_gather_hpu(
    const at::Tensor& first_storage_u8,
    const at::Tensor& first_geometry,
    const at::Tensor& first_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& second_storage_u8,
    const at::Tensor& second_geometry,
    const at::Tensor& second_indices) {
  TORCH_CHECK(
      first_storage_u8.scalar_type() == at::ScalarType::Byte &&
      second_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(
      first_geometry.scalar_type() == at::ScalarType::Int &&
      first_shape_buffer.scalar_type() == at::ScalarType::Int &&
      token_to_req_indices.scalar_type() == at::ScalarType::Int &&
      block_table.scalar_type() == at::ScalarType::Int &&
      is_valid_token.scalar_type() == at::ScalarType::Int &&
      seq_lens.scalar_type() == at::ScalarType::Int &&
      second_geometry.scalar_type() == at::ScalarType::Int &&
      second_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      first_storage_u8.dim() == 1 && first_storage_u8.numel() > 0 &&
      second_storage_u8.dim() == 1 && second_storage_u8.numel() > 0);
  TORCH_CHECK(
      first_geometry.dim() == 1 && first_geometry.numel() == 4 &&
      second_geometry.dim() == 1 && second_geometry.numel() == 4);
  TORCH_CHECK(
      first_shape_buffer.dim() == 1 && first_shape_buffer.numel() > 0 &&
      token_to_req_indices.dim() == 1 &&
      token_to_req_indices.numel() > 0 && block_table.dim() == 2 &&
      block_table.numel() > 0 && is_valid_token.dim() == 1 &&
      is_valid_token.numel() > 0 && seq_lens.dim() == 1 &&
      seq_lens.numel() > 0 && second_indices.dim() == 1 &&
      second_indices.numel() > 0);
  TORCH_CHECK(
      first_storage_u8.is_contiguous() && first_geometry.is_contiguous() &&
      first_shape_buffer.is_contiguous() &&
      token_to_req_indices.is_contiguous() && block_table.is_contiguous() &&
      is_valid_token.is_contiguous() && seq_lens.is_contiguous() &&
      second_storage_u8.is_contiguous() &&
      second_geometry.is_contiguous() && second_indices.is_contiguous());
  TORCH_CHECK(kLocalDualDequantGatherRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kLocalDualDequantSchema);
  std::vector<c10::IValue> inputs{
      first_storage_u8,
      first_geometry,
      first_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      second_storage_u8,
      second_geometry,
      second_indices};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor local_dual_dequant_gather_meta(
    const at::Tensor& first_storage_u8,
    const at::Tensor& first_geometry,
    const at::Tensor& first_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& second_storage_u8,
    const at::Tensor& second_geometry,
    const at::Tensor& second_indices) {
  (void)first_storage_u8;
  (void)first_geometry;
  (void)token_to_req_indices;
  (void)block_table;
  (void)is_valid_token;
  (void)seq_lens;
  (void)second_storage_u8;
  (void)second_geometry;
  return at::empty(
      {first_shape_buffer.size(0) + second_indices.size(0), 512},
      first_shape_buffer.options().dtype(at::ScalarType::BFloat16));
}

at::Tensor save_partial_states_hpu(
    const at::Tensor& state_storage,
    const at::Tensor& state_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping) {
  TORCH_CHECK(state_storage.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(state_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(kv.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(score.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(ape.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      state_storage.dim() == 1 && state_storage.numel() > 0 &&
      state_geometry.dim() == 1 && state_geometry.numel() == 5);
  TORCH_CHECK(kv.dim() == 2 && score.sizes() == kv.sizes());
  TORCH_CHECK(
      kv.size(1) >= 64 && kv.size(1) <= 1024 && kv.size(1) % 64 == 0);
  TORCH_CHECK(
      ape.dim() == 2 && ape.size(1) == kv.size(1) &&
      (ape.size(0) == 4 || ape.size(0) == 128));
  TORCH_CHECK(
      positions.dim() == 1 && slot_mapping.dim() == 1 &&
      slot_mapping.numel() > 0 &&
      positions.numel() >= slot_mapping.numel() &&
      kv.size(0) >= slot_mapping.numel());
  TORCH_CHECK(state_storage.is_contiguous());
  TORCH_CHECK(state_geometry.is_contiguous());
  TORCH_CHECK(ape.is_contiguous());
  TORCH_CHECK(positions.is_contiguous());
  TORCH_CHECK(slot_mapping.is_contiguous());
  TORCH_CHECK(kSavePartialStatesRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kSavePartialStatesSchema);
  std::vector<c10::IValue> inputs{
      state_storage,
      state_geometry,
      kv,
      score,
      ape,
      positions,
      slot_mapping};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor save_partial_states_meta(
    const at::Tensor& state_storage,
    const at::Tensor& state_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping) {
  (void)state_storage;
  (void)state_geometry;
  (void)score;
  (void)ape;
  (void)positions;
  return at::empty(
      {slot_mapping.size(0), kv.size(1) / 64},
      kv.options().dtype(at::ScalarType::Float));
}

at::Tensor save_compress_norm_c4_hpu_impl(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& cache_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& kv_slot_mapping,
    bool bf16_inputs,
    bool bf16_norm) {
  TORCH_CHECK(storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(state_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(cache_geometry.scalar_type() == at::ScalarType::Int);
  const auto input_type = bf16_inputs
      ? at::ScalarType::BFloat16
      : at::ScalarType::Float;
  TORCH_CHECK(kv.scalar_type() == input_type);
  TORCH_CHECK(score.scalar_type() == input_type);
  TORCH_CHECK(ape.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(token_to_req_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int);
  const auto constants_type = bf16_norm
      ? at::ScalarType::BFloat16
      : at::ScalarType::Float;
  TORCH_CHECK(rms_norm_weight.scalar_type() == constants_type);
  TORCH_CHECK(rms_norm_eps.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(kv_slot_mapping.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      storage_u8.dim() == 1 && storage_u8.numel() > 0 &&
      state_geometry.dim() == 1 && state_geometry.numel() == 5 &&
      cache_geometry.dim() == 1 && cache_geometry.numel() == 4);
  TORCH_CHECK(kv.dim() == 2 && score.sizes() == kv.sizes());
  const auto head_dim = rms_norm_weight.size(0);
  const auto compress_ratio = ape.size(0);
  const auto coff = compress_ratio == 4 ? 2 : 1;
  TORCH_CHECK(
      (head_dim == 128 || head_dim == 512) &&
      kv.size(1) == coff * head_dim);
  TORCH_CHECK(
      ape.dim() == 2 && ape.size(1) == coff * head_dim &&
      (compress_ratio == 4 ||
       (compress_ratio == 128 && head_dim == 512)));
  const auto num_tokens = slot_mapping.numel();
  TORCH_CHECK(
      positions.dim() == 1 && slot_mapping.dim() == 1 &&
      token_to_req_indices.dim() == 1 && kv_slot_mapping.dim() == 1 &&
      num_tokens > 0 &&
      positions.numel() >= num_tokens &&
      kv_slot_mapping.numel() >= num_tokens &&
      token_to_req_indices.numel() >= num_tokens &&
      kv.size(0) >= num_tokens);
  TORCH_CHECK(block_table.dim() == 2 && block_table.numel() > 0);
  TORCH_CHECK(
      rms_norm_weight.dim() == 1 && rms_norm_weight.size(0) == head_dim &&
      rms_norm_eps.dim() == 1 && rms_norm_eps.numel() == 1);
  TORCH_CHECK(
      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64);
  TORCH_CHECK(storage_u8.is_contiguous());
  TORCH_CHECK(state_geometry.is_contiguous());
  TORCH_CHECK(cache_geometry.is_contiguous());
  TORCH_CHECK(ape.is_contiguous());
  TORCH_CHECK(positions.is_contiguous());
  TORCH_CHECK(slot_mapping.is_contiguous());
  TORCH_CHECK(token_to_req_indices.is_contiguous());
  TORCH_CHECK(block_table.is_contiguous());
  TORCH_CHECK(rms_norm_weight.is_contiguous());
  TORCH_CHECK(rms_norm_eps.is_contiguous());
  TORCH_CHECK(cos_sin_cache.is_contiguous());
  TORCH_CHECK(kv_slot_mapping.is_contiguous());
  TORCH_CHECK(
      bf16_inputs
          ? kSaveCompressNormC4BF16Registered
          : bf16_norm
              ? kSaveCompressNormC4MixedRegistered
              : kSaveCompressNormC4Registered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          bf16_inputs
              ? kSaveCompressNormC4BF16Schema
              : bf16_norm
                  ? kSaveCompressNormC4MixedSchema
                  : kSaveCompressNormC4Schema);
  std::vector<c10::IValue> inputs{
      storage_u8,
      state_geometry,
      cache_geometry,
      kv,
      score,
      ape,
      positions,
      slot_mapping,
      token_to_req_indices,
      block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      kv_slot_mapping};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor save_compress_norm_c4_hpu(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& cache_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& kv_slot_mapping) {
  return save_compress_norm_c4_hpu_impl(
      storage_u8,
      state_geometry,
      cache_geometry,
      kv,
      score,
      ape,
      positions,
      slot_mapping,
      token_to_req_indices,
      block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      kv_slot_mapping,
      false,
      false);
}

at::Tensor save_compress_norm_c4_bf16_hpu(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& cache_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& kv_slot_mapping) {
  return save_compress_norm_c4_hpu_impl(
      storage_u8,
      state_geometry,
      cache_geometry,
      kv,
      score,
      ape,
      positions,
      slot_mapping,
      token_to_req_indices,
      block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      kv_slot_mapping,
      true,
      true);
}

at::Tensor save_compress_norm_c4_mixed_hpu(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& cache_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& kv_slot_mapping) {
  return save_compress_norm_c4_hpu_impl(
      storage_u8,
      state_geometry,
      cache_geometry,
      kv,
      score,
      ape,
      positions,
      slot_mapping,
      token_to_req_indices,
      block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      kv_slot_mapping,
      false,
      true);
}

at::Tensor save_compress_norm_c4_meta(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& cache_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& kv_slot_mapping) {
  (void)storage_u8;
  (void)state_geometry;
  (void)cache_geometry;
  (void)score;
  (void)ape;
  (void)positions;
  (void)token_to_req_indices;
  (void)block_table;
  (void)rms_norm_weight;
  (void)rms_norm_eps;
  (void)cos_sin_cache;
  (void)kv_slot_mapping;
  return at::empty(
      {slot_mapping.size(0)},
      kv.options().dtype(at::ScalarType::Byte));
}

at::Tensor qnorm_rope_kv_pack_hpu_impl(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const char* schema,
    bool registered) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(kv.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(cache_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(cache_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(slots.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512);
  TORCH_CHECK(
      kv.dim() == 2 && kv.size(0) >= q.size(0) && kv.size(1) == 512);
  TORCH_CHECK(cache_storage_u8.dim() == 1 && cache_storage_u8.numel() >= 584);
  TORCH_CHECK(cache_geometry.dim() == 1 && cache_geometry.numel() == 4);
  TORCH_CHECK(slots.dim() == 1 && slots.size(0) >= q.size(0));
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) >= q.size(0));
  TORCH_CHECK(
      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64);
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(cache_storage_u8.is_contiguous());
  TORCH_CHECK(cache_geometry.is_contiguous());
  TORCH_CHECK(slots.is_contiguous());
  TORCH_CHECK(positions.is_contiguous());
  TORCH_CHECK(cos_sin_cache.is_contiguous());
  TORCH_CHECK(registered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          schema);
  std::vector<c10::IValue> inputs{
      q,
      kv,
      cache_storage_u8,
      cache_geometry,
      slots,
      positions,
      cos_sin_cache};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor qnorm_rope_kv_pack_hpu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache) {
  return qnorm_rope_kv_pack_hpu_impl(
      q,
      kv,
      cache_storage_u8,
      cache_geometry,
      slots,
      positions,
      cos_sin_cache,
      kQnormRopeKvPackSchema,
      kQnormRopeKvPackRegistered);
}

at::Tensor hybrid_qnorm_rope_kv_pack_hpu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache) {
  return qnorm_rope_kv_pack_hpu_impl(
      q,
      kv,
      cache_storage_u8,
      cache_geometry,
      slots,
      positions,
      cos_sin_cache,
      kHybridQnormRopeKvPackSchema,
      kHybridQnormRopeKvPackRegistered);
}

at::Tensor qnorm_rope_kv_pack_meta(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache) {
  (void)kv;
  (void)cache_storage_u8;
  (void)cache_geometry;
  (void)slots;
  (void)positions;
  (void)cos_sin_cache;
  return at::empty({q.size(0)}, q.options().dtype(at::ScalarType::Byte));
}

std::tuple<at::Tensor, at::Tensor> qnorm_compressor_c4_hpu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& state_storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& completion_input) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(kv.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(state_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(compressor_kv.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(compressor_score.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && q.size(0) == 1);
  TORCH_CHECK(kv.dim() == 2 && kv.size(0) >= q.size(0));
  TORCH_CHECK(compressor_kv.dim() == 2);
  TORCH_CHECK(compressor_score.sizes() == compressor_kv.sizes());
  TORCH_CHECK(ape.dim() == 2 && ape.size(0) == 4);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(completion_input.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(positions.is_contiguous());
  TORCH_CHECK(kQnormCompressorC4BackendRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kQnormCompressorC4Schema);
  std::vector<c10::IValue> inputs{
      q,
      kv,
      swa_storage_u8,
      swa_geometry,
      swa_slots,
      positions,
      cos_sin_cache,
      state_storage_u8,
      state_geometry,
      compressed_geometry,
      compressor_kv,
      compressor_score,
      ape,
      state_slots,
      state_token_to_req_indices,
      state_block_table,
      rms_norm_weight,
      rms_norm_eps,
      compressed_kv_slots,
      completion_input};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> qnorm_compressor_c4_meta(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& state_storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& completion_input) {
  (void)kv;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_slots;
  (void)positions;
  (void)cos_sin_cache;
  (void)state_storage_u8;
  (void)state_geometry;
  (void)compressed_geometry;
  (void)compressor_kv;
  (void)compressor_score;
  (void)ape;
  (void)state_slots;
  (void)state_token_to_req_indices;
  (void)state_block_table;
  (void)rms_norm_weight;
  (void)rms_norm_eps;
  (void)compressed_kv_slots;
  (void)completion_input;
  return std::make_tuple(
      at::empty({q.size(0)}, q.options().dtype(at::ScalarType::Byte)),
      at::empty({q.size(0)}, q.options().dtype(at::ScalarType::Int)));
}

at::Tensor insert_packed_kv_hpu(
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& packed,
    const at::Tensor& slots) {
  TORCH_CHECK(cache_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(cache_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(packed.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(slots.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(cache_storage_u8.dim() == 1 && cache_storage_u8.numel() >= 584);
  TORCH_CHECK(cache_geometry.dim() == 1 && cache_geometry.numel() == 4);
  TORCH_CHECK(packed.dim() == 2 && packed.size(1) == 584);
  TORCH_CHECK(slots.dim() == 1 && slots.numel() >= packed.size(0));
  TORCH_CHECK(cache_storage_u8.is_contiguous());
  TORCH_CHECK(cache_geometry.is_contiguous());
  TORCH_CHECK(packed.is_contiguous());
  TORCH_CHECK(slots.is_contiguous());
  TORCH_CHECK(kInsertPackedKvRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kInsertPackedKvSchema);
  std::vector<c10::IValue> inputs{
      cache_storage_u8, cache_geometry, packed, slots};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor insert_packed_kv_meta(
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& packed,
    const at::Tensor& slots) {
  (void)cache_storage_u8;
  (void)cache_geometry;
  (void)slots;
  return at::empty(
      {packed.size(0)},
      packed.options().dtype(at::ScalarType::Byte));
}

at::Tensor paged_sparse_attention_hpu_impl(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_indices,
    const at::Tensor& topk_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output,
    const char* schema,
    bool registered) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(compressed_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(compressed_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(topk_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(topk_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(swa_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && q.size(0) > 0);
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == q.size(0) &&
      output.size(1) >= q.size(1) && output.size(2) == q.size(2));
  TORCH_CHECK(
      compressed_storage_u8.dim() == 1 &&
      compressed_storage_u8.numel() >= 584 &&
      compressed_geometry.dim() == 1 &&
      compressed_geometry.numel() == 4);
  TORCH_CHECK(
      topk_indices.dim() == 2 && topk_indices.size(0) == q.size(0) &&
      topk_indices.size(1) > 0 && topk_lens.dim() == 1 &&
      topk_lens.size(0) == q.size(0));
  TORCH_CHECK(
      swa_storage_u8.dim() == 1 && swa_storage_u8.numel() >= 584 &&
      swa_geometry.dim() == 1 && swa_geometry.numel() == 4);
  TORCH_CHECK(
      swa_indices.dim() == 2 && swa_indices.size(0) == q.size(0) &&
      swa_indices.size(1) > 0 && swa_lens.dim() == 1 &&
      swa_lens.size(0) == q.size(0));
  TORCH_CHECK(
      attn_sink.dim() == 1 && attn_sink.size(0) == output.size(1));
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(compressed_storage_u8.is_contiguous());
  TORCH_CHECK(compressed_geometry.is_contiguous());
  TORCH_CHECK(topk_indices.is_contiguous());
  TORCH_CHECK(topk_lens.is_contiguous());
  TORCH_CHECK(swa_storage_u8.is_contiguous());
  TORCH_CHECK(swa_geometry.is_contiguous());
  TORCH_CHECK(swa_indices.is_contiguous());
  TORCH_CHECK(swa_lens.is_contiguous());
  TORCH_CHECK(attn_sink.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(registered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          schema);
  std::vector<c10::IValue> inputs{
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_indices,
      topk_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor paged_sparse_attention_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_indices,
    const at::Tensor& topk_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  return paged_sparse_attention_hpu_impl(
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_indices,
      topk_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output,
      kPagedSparseAttnSchema,
      kPagedSparseAttentionRegistered);
}

at::Tensor paged_sparse_attention_meta(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_indices,
    const at::Tensor& topk_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  (void)compressed_storage_u8;
  (void)compressed_geometry;
  (void)topk_indices;
  (void)topk_lens;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  (void)attn_sink;
  return at::empty(
      {q.size(0), q.size(1)},
      q.options().dtype(at::ScalarType::Float));
}

at::Tensor paged_sparse_attention_local_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& local_topk_indices,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(compressed_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(compressed_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(local_topk_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(token_to_req_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(is_valid_token.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(swa_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && q.size(0) > 0);
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == q.size(0) &&
      output.size(1) >= q.size(1) && output.size(2) == q.size(2));
  TORCH_CHECK(
      compressed_storage_u8.dim() == 1 &&
      compressed_storage_u8.numel() >= 584 &&
      compressed_geometry.dim() == 1 &&
      compressed_geometry.numel() == 4);
  TORCH_CHECK(
      local_topk_indices.dim() == 2 &&
      local_topk_indices.size(0) == q.size(0) &&
      local_topk_indices.size(1) > 0 &&
      token_to_req_indices.dim() == 1 &&
      token_to_req_indices.size(0) == q.size(0) &&
      block_table.dim() == 2 && block_table.size(0) > 0 &&
      block_table.size(1) > 0 && is_valid_token.dim() == 1 &&
      is_valid_token.size(0) == q.size(0));
  TORCH_CHECK(
      swa_storage_u8.dim() == 1 && swa_storage_u8.numel() >= 584 &&
      swa_geometry.dim() == 1 && swa_geometry.numel() == 4);
  TORCH_CHECK(
      swa_indices.dim() == 2 && swa_indices.size(0) == q.size(0) &&
      swa_indices.size(1) > 0 && swa_lens.dim() == 1 &&
      swa_lens.size(0) == q.size(0));
  TORCH_CHECK(
      attn_sink.dim() == 1 && attn_sink.size(0) == output.size(1));
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(compressed_storage_u8.is_contiguous());
  TORCH_CHECK(compressed_geometry.is_contiguous());
  TORCH_CHECK(local_topk_indices.is_contiguous());
  TORCH_CHECK(token_to_req_indices.is_contiguous());
  TORCH_CHECK(block_table.is_contiguous());
  TORCH_CHECK(is_valid_token.is_contiguous());
  TORCH_CHECK(swa_storage_u8.is_contiguous());
  TORCH_CHECK(swa_geometry.is_contiguous());
  TORCH_CHECK(swa_indices.is_contiguous());
  TORCH_CHECK(swa_lens.is_contiguous());
  TORCH_CHECK(attn_sink.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(kPagedSparseAttentionLocalRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kPagedSparseAttnLocalSchema);
  std::vector<c10::IValue> inputs{
      q,
      compressed_storage_u8,
      compressed_geometry,
      local_topk_indices,
      token_to_req_indices,
      block_table,
      is_valid_token,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor paged_sparse_attention_local_meta(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& local_topk_indices,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  (void)compressed_storage_u8;
  (void)compressed_geometry;
  (void)local_topk_indices;
  (void)token_to_req_indices;
  (void)block_table;
  (void)is_valid_token;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  (void)attn_sink;
  return at::empty(
      {q.size(0), q.size(1)},
      q.options().dtype(at::ScalarType::Float));
}

at::Tensor paged_sparse_attention_sequential_impl(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output,
    const char* schema,
    bool registered) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(compressed_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(compressed_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(topk_shape_buffer.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(token_to_req_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(is_valid_token.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(seq_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(swa_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && q.size(0) > 0);
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == q.size(0) &&
      output.size(1) >= q.size(1) && output.size(2) == q.size(2));
  TORCH_CHECK(
      compressed_storage_u8.dim() == 1 &&
      compressed_storage_u8.numel() >= 584 &&
      compressed_geometry.dim() == 1 &&
      compressed_geometry.numel() == 4);
  TORCH_CHECK(
      topk_shape_buffer.dim() == 2 &&
      topk_shape_buffer.size(0) == q.size(0) &&
      topk_shape_buffer.size(1) > 0 &&
      token_to_req_indices.dim() == 1 &&
      token_to_req_indices.size(0) == q.size(0) &&
      block_table.dim() == 2 && block_table.size(0) > 0 &&
      block_table.size(1) > 0 && is_valid_token.dim() == 1 &&
      is_valid_token.size(0) == q.size(0) && seq_lens.dim() == 1 &&
      seq_lens.size(0) > 0);
  TORCH_CHECK(
      swa_storage_u8.dim() == 1 && swa_storage_u8.numel() >= 584 &&
      swa_geometry.dim() == 1 && swa_geometry.numel() == 4);
  TORCH_CHECK(
      swa_indices.dim() == 2 && swa_indices.size(0) == q.size(0) &&
      swa_indices.size(1) > 0 && swa_lens.dim() == 1 &&
      swa_lens.size(0) == q.size(0));
  TORCH_CHECK(
      attn_sink.dim() == 1 && attn_sink.size(0) == output.size(1));
  TORCH_CHECK(q.is_contiguous());
  TORCH_CHECK(compressed_storage_u8.is_contiguous());
  TORCH_CHECK(compressed_geometry.is_contiguous());
  TORCH_CHECK(topk_shape_buffer.is_contiguous());
  TORCH_CHECK(token_to_req_indices.is_contiguous());
  TORCH_CHECK(block_table.is_contiguous());
  TORCH_CHECK(is_valid_token.is_contiguous());
  TORCH_CHECK(seq_lens.is_contiguous());
  TORCH_CHECK(swa_storage_u8.is_contiguous());
  TORCH_CHECK(swa_geometry.is_contiguous());
  TORCH_CHECK(swa_indices.is_contiguous());
  TORCH_CHECK(swa_lens.is_contiguous());
  TORCH_CHECK(attn_sink.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(registered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          schema);
  std::vector<c10::IValue> inputs{
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor paged_sparse_attention_sequential_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  return paged_sparse_attention_sequential_impl(
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output,
      kPagedSparseAttnSequentialSchema,
      kPagedSparseAttentionSequentialRegistered);
}

at::Tensor paged_sparse_attention_pair_sequential_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  return paged_sparse_attention_sequential_impl(
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output,
      kPagedSparseAttnPairSequentialSchema,
      kPagedSparseAttentionPairSequentialRegistered);
}

at::Tensor paged_swa_attention_hpu(
    const at::Tensor& q,
    const at::Tensor& dummy_storage_u8,
    const at::Tensor& dummy_geometry,
    const at::Tensor& dummy_topk_shape,
    const at::Tensor& dummy_token_to_req,
    const at::Tensor& dummy_block_table,
    const at::Tensor& dummy_valid_token,
    const at::Tensor& dummy_seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  return paged_sparse_attention_sequential_impl(
      q,
      dummy_storage_u8,
      dummy_geometry,
      dummy_topk_shape,
      dummy_token_to_req,
      dummy_block_table,
      dummy_valid_token,
      dummy_seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output,
      kPagedSwaAttnSchema,
      kPagedSwaAttentionRegistered);
}

at::Tensor paged_sparse_attention_sequential_meta(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  (void)compressed_storage_u8;
  (void)compressed_geometry;
  (void)topk_shape_buffer;
  (void)token_to_req_indices;
  (void)block_table;
  (void)is_valid_token;
  (void)seq_lens;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  (void)attn_sink;
  return at::empty(
      {q.size(0), q.size(1)},
      q.options().dtype(at::ScalarType::Float));
}

std::tuple<at::Tensor, at::Tensor> flashmla_splitkv_partial_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(compressed_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(compressed_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(topk_shape_buffer.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(split_shape_buffer.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(token_to_req_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(is_valid_token.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(seq_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(swa_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && q.size(0) > 0);
  TORCH_CHECK(
      compressed_storage_u8.dim() == 1 &&
      compressed_storage_u8.numel() >= 584 &&
      compressed_geometry.dim() == 1 &&
      compressed_geometry.numel() == 4);
  TORCH_CHECK(
      topk_shape_buffer.dim() == 2 &&
      topk_shape_buffer.size(0) == q.size(0) &&
      topk_shape_buffer.size(1) > 0);
  TORCH_CHECK(
      split_shape_buffer.dim() == 1 &&
      split_shape_buffer.size(0) > 0 &&
      split_shape_buffer.size(0) <= 16);
  TORCH_CHECK(
      token_to_req_indices.dim() == 1 &&
      token_to_req_indices.size(0) == q.size(0) &&
      block_table.dim() == 2 && block_table.numel() > 0 &&
      is_valid_token.dim() == 1 &&
      is_valid_token.size(0) == q.size(0) &&
      seq_lens.dim() == 1 && seq_lens.numel() > 0);
  TORCH_CHECK(
      swa_storage_u8.dim() == 1 && swa_storage_u8.numel() >= 584 &&
      swa_geometry.dim() == 1 && swa_geometry.numel() == 4 &&
      swa_indices.dim() == 2 && swa_indices.size(0) == q.size(0) &&
      swa_indices.size(1) > 0 && swa_lens.dim() == 1 &&
      swa_lens.size(0) == q.size(0));
  TORCH_CHECK(
      q.is_contiguous() && compressed_storage_u8.is_contiguous() &&
      compressed_geometry.is_contiguous() &&
      topk_shape_buffer.is_contiguous() &&
      split_shape_buffer.is_contiguous() &&
      token_to_req_indices.is_contiguous() && block_table.is_contiguous() &&
      is_valid_token.is_contiguous() && seq_lens.is_contiguous() &&
      swa_storage_u8.is_contiguous() && swa_geometry.is_contiguous() &&
      swa_indices.is_contiguous() && swa_lens.is_contiguous());
  TORCH_CHECK(kFlashMLASplitKVPartialRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kFlashMLASplitKVPartialSchema);
  std::vector<c10::IValue> inputs{
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      split_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> flashmla_splitkv_partial_meta(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens) {
  (void)compressed_storage_u8;
  (void)compressed_geometry;
  (void)topk_shape_buffer;
  (void)token_to_req_indices;
  (void)block_table;
  (void)is_valid_token;
  (void)seq_lens;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  const auto splits = split_shape_buffer.size(0);
  return std::make_tuple(
      at::empty(
          {q.size(0), splits, q.size(1), 512},
          q.options().dtype(at::ScalarType::Float)),
      at::empty(
          {q.size(0), splits, q.size(1), 2},
          q.options().dtype(at::ScalarType::Float)));
}

std::tuple<at::Tensor, at::Tensor> flashmla_splitkv_combine_hpu(
    const at::Tensor& partial_output,
    const at::Tensor& partial_stats,
    const at::Tensor& attn_sink) {
  TORCH_CHECK(partial_output.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(partial_stats.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(
      partial_output.dim() == 4 && partial_output.size(3) == 512 &&
      partial_output.size(0) > 0 && partial_output.size(1) > 0 &&
      partial_output.size(1) <= 16 && partial_output.size(2) > 0);
  TORCH_CHECK(
      partial_stats.dim() == 4 && partial_stats.size(0) == partial_output.size(0) &&
      partial_stats.size(1) == partial_output.size(1) &&
      partial_stats.size(2) == partial_output.size(2) &&
      partial_stats.size(3) == 2);
  TORCH_CHECK(
      attn_sink.dim() == 1 && attn_sink.size(0) >= partial_output.size(2));
  TORCH_CHECK(
      partial_output.is_contiguous() && partial_stats.is_contiguous() &&
      attn_sink.is_contiguous());
  TORCH_CHECK(kFlashMLASplitKVCombineRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kFlashMLASplitKVCombineSchema);
  std::vector<c10::IValue> inputs{
      partial_output, partial_stats, attn_sink};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> flashmla_splitkv_combine_meta(
    const at::Tensor& partial_output,
    const at::Tensor& partial_stats,
    const at::Tensor& attn_sink) {
  (void)partial_stats;
  (void)attn_sink;
  return std::make_tuple(
      at::empty(
          {partial_output.size(0), partial_output.size(2), 512},
          partial_output.options().dtype(at::ScalarType::BFloat16)),
      at::empty(
          {partial_output.size(0), partial_output.size(2)},
          partial_output.options().dtype(at::ScalarType::Float)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
flashmla_splitkv_hpu_impl(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template,
    const char* schema,
    bool backend_registered) {
  TORCH_CHECK(output_template.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      output_template.dim() == 3 && output_template.size(0) == q.size(0) &&
      output_template.size(1) >= q.size(1) &&
      output_template.size(2) == 512 && output_template.is_contiguous());
  TORCH_CHECK(backend_registered);
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          schema);
  std::vector<c10::IValue> inputs{
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      split_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output_template};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 4);
  return std::make_tuple(
      outputs.at(0), outputs.at(1), outputs.at(2), outputs.at(3));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
flashmla_splitkv_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  return flashmla_splitkv_hpu_impl(
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      split_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output_template,
      kFlashMLASplitKVSchema,
      kFlashMLASplitKVBackendRegistered);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
flashmla_splitkv_tiled_hpu(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  return flashmla_splitkv_hpu_impl(
      q,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      split_shape_buffer,
      token_to_req_indices,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output_template,
      kFlashMLASplitKVTiledSchema,
      kFlashMLASplitKVTiledBackendRegistered);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
flashmla_splitkv_meta(
    const at::Tensor& q,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  (void)compressed_storage_u8;
  (void)compressed_geometry;
  (void)topk_shape_buffer;
  (void)token_to_req_indices;
  (void)block_table;
  (void)is_valid_token;
  (void)seq_lens;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  (void)attn_sink;
  const auto split_count = split_shape_buffer.size(0);
  return std::make_tuple(
      at::empty(
          output_template.sizes(),
          q.options().dtype(at::ScalarType::BFloat16)),
      at::empty(
          {q.size(0), q.size(1)},
          q.options().dtype(at::ScalarType::Float)),
      at::empty(
          {q.size(0), split_count, q.size(1), 512},
          q.options().dtype(at::ScalarType::Float)),
      at::empty(
          {q.size(0), split_count, q.size(1), 2},
          q.options().dtype(at::ScalarType::Float)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compress_flashmla_c4_hpu(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& q,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& attention_token_to_req_indices,
    const at::Tensor& attention_block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  TORCH_CHECK(storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(state_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(compressed_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(compressor_kv.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(compressor_score.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(ape.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(state_slots.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      state_token_to_req_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(state_block_table.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(rms_norm_weight.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(rms_norm_eps.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(compressed_kv_slots.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(topk_shape_buffer.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(split_shape_buffer.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      attention_token_to_req_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attention_block_table.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(is_valid_token.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(seq_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(swa_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output_template.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      q.dim() == 3 && q.size(0) > 0 && q.size(2) == 512 &&
      output_template.dim() == 3 &&
      output_template.size(0) == q.size(0) &&
      output_template.size(1) >= q.size(1) &&
      output_template.size(2) == 512);
  TORCH_CHECK(kCompressFlashMLAC4BackendRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kCompressFlashMLAC4Schema);
  std::vector<c10::IValue> inputs{
      storage_u8,
      state_geometry,
      compressed_geometry,
      compressor_kv,
      compressor_score,
      ape,
      positions,
      state_slots,
      state_token_to_req_indices,
      state_block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      compressed_kv_slots,
      q,
      topk_shape_buffer,
      split_shape_buffer,
      attention_token_to_req_indices,
      attention_block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output_template};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 4);
  return std::make_tuple(
      outputs.at(0), outputs.at(1), outputs.at(2), outputs.at(3));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compress_flashmla_c4_meta(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& q,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& attention_token_to_req_indices,
    const at::Tensor& attention_block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  (void)storage_u8;
  (void)state_geometry;
  (void)compressed_geometry;
  (void)compressor_kv;
  (void)compressor_score;
  (void)ape;
  (void)positions;
  (void)state_slots;
  (void)state_token_to_req_indices;
  (void)state_block_table;
  (void)rms_norm_weight;
  (void)rms_norm_eps;
  (void)cos_sin_cache;
  (void)compressed_kv_slots;
  (void)topk_shape_buffer;
  (void)attention_token_to_req_indices;
  (void)attention_block_table;
  (void)is_valid_token;
  (void)seq_lens;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  (void)attn_sink;
  return std::make_tuple(
      at::empty(
          output_template.sizes(),
          q.options().dtype(at::ScalarType::BFloat16)),
      at::empty(
          {q.size(0), q.size(1)},
          q.options().dtype(at::ScalarType::Float)),
      at::empty(
          {q.size(0), split_shape_buffer.size(0), q.size(1), 512},
          q.options().dtype(at::ScalarType::Float)),
      at::empty(
          {q.size(0), split_shape_buffer.size(0), q.size(1), 2},
          q.options().dtype(at::ScalarType::Float)));
}

at::Tensor qnorm_paged_sparse_attention_sequential_hpu(
    const at::Tensor& q,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long);
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(compressed_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(compressed_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(topk_shape_buffer.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(is_valid_token.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(seq_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_storage_u8.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(swa_geometry.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_indices.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(swa_lens.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(attn_sink.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 512 && q.size(0) > 0);
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == q.size(0) &&
      output.size(1) >= q.size(1) && output.size(2) == q.size(2));
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) >= q.size(0) &&
      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64);
  TORCH_CHECK(
      compressed_storage_u8.dim() == 1 &&
      compressed_storage_u8.numel() >= 584 &&
      compressed_geometry.dim() == 1 &&
      compressed_geometry.numel() == 4);
  TORCH_CHECK(
      topk_shape_buffer.dim() == 2 &&
      topk_shape_buffer.size(0) == q.size(0) &&
      topk_shape_buffer.size(1) > 0 &&
      block_table.dim() == 2 && block_table.size(0) > 0 &&
      block_table.size(1) > 0 && is_valid_token.dim() == 1 &&
      is_valid_token.size(0) == q.size(0) && seq_lens.dim() == 1 &&
      seq_lens.size(0) > 0);
  TORCH_CHECK(
      swa_storage_u8.dim() == 1 && swa_storage_u8.numel() >= 584 &&
      swa_geometry.dim() == 1 && swa_geometry.numel() == 4);
  TORCH_CHECK(
      swa_indices.dim() == 2 && swa_indices.size(0) == q.size(0) &&
      swa_indices.size(1) > 0 && swa_lens.dim() == 1 &&
      swa_lens.size(0) == q.size(0));
  TORCH_CHECK(
      attn_sink.dim() == 1 && attn_sink.size(0) == output.size(1));
  TORCH_CHECK(
      q.is_contiguous() && positions.is_contiguous() &&
      cos_sin_cache.is_contiguous() &&
      compressed_storage_u8.is_contiguous() &&
      compressed_geometry.is_contiguous() &&
      topk_shape_buffer.is_contiguous() && block_table.is_contiguous() &&
      is_valid_token.is_contiguous() && seq_lens.is_contiguous() &&
      swa_storage_u8.is_contiguous() && swa_geometry.is_contiguous() &&
      swa_indices.is_contiguous() && swa_lens.is_contiguous() &&
      attn_sink.is_contiguous() && output.is_contiguous());
  TORCH_CHECK(kQnormPagedSparseAttentionSequentialRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kQnormPagedSparseAttnSequentialSchema);
  std::vector<c10::IValue> inputs{
      q,
      positions,
      cos_sin_cache,
      compressed_storage_u8,
      compressed_geometry,
      topk_shape_buffer,
      block_table,
      is_valid_token,
      seq_lens,
      swa_storage_u8,
      swa_geometry,
      swa_indices,
      swa_lens,
      attn_sink,
      output};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor qnorm_paged_sparse_attention_sequential_meta(
    const at::Tensor& q,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& compressed_storage_u8,
    const at::Tensor& compressed_geometry,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output) {
  (void)positions;
  (void)cos_sin_cache;
  (void)compressed_storage_u8;
  (void)compressed_geometry;
  (void)topk_shape_buffer;
  (void)block_table;
  (void)is_valid_token;
  (void)seq_lens;
  (void)swa_storage_u8;
  (void)swa_geometry;
  (void)swa_indices;
  (void)swa_lens;
  (void)attn_sink;
  return at::empty(
      {q.size(0), q.size(1)},
      q.options().dtype(at::ScalarType::Float));
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
c4_decode_chain_hpu(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& state_storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& attention_token_to_req_indices,
    const at::Tensor& attention_block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  auto qnorm_ack = hybrid_qnorm_rope_kv_pack_hpu(
      q,
      kv,
      swa_storage_u8,
      swa_geometry,
      swa_slots,
      positions,
      cos_sin_cache);
  auto compressor_ack = save_compress_norm_c4_hpu(
      state_storage_u8,
      state_geometry,
      compressed_geometry,
      compressor_kv,
      compressor_score,
      ape,
      positions,
      state_slots,
      state_token_to_req_indices,
      state_block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      compressed_kv_slots);
  auto [attention_output, stats, partial_output, partial_stats] =
      flashmla_splitkv_tiled_hpu(
          q,
          state_storage_u8,
          compressed_geometry,
          topk_shape_buffer,
          split_shape_buffer,
          attention_token_to_req_indices,
          attention_block_table,
          is_valid_token,
          seq_lens,
          swa_storage_u8,
          swa_geometry,
          swa_indices,
          swa_lens,
          attn_sink,
          output_template);
  return std::make_tuple(
      attention_output,
      stats,
      qnorm_ack,
      compressor_ack,
      partial_output,
      partial_stats);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
c4_decode_chain_meta(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& state_storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& topk_shape_buffer,
    const at::Tensor& split_shape_buffer,
    const at::Tensor& attention_token_to_req_indices,
    const at::Tensor& attention_block_table,
    const at::Tensor& is_valid_token,
    const at::Tensor& seq_lens,
    const at::Tensor& swa_indices,
    const at::Tensor& swa_lens,
    const at::Tensor& attn_sink,
    const at::Tensor& output_template) {
  auto qnorm_ack = qnorm_rope_kv_pack_meta(
      q,
      kv,
      swa_storage_u8,
      swa_geometry,
      swa_slots,
      positions,
      cos_sin_cache);
  auto compressor_ack = save_compress_norm_c4_meta(
      state_storage_u8,
      state_geometry,
      compressed_geometry,
      compressor_kv,
      compressor_score,
      ape,
      positions,
      state_slots,
      state_token_to_req_indices,
      state_block_table,
      rms_norm_weight,
      rms_norm_eps,
      cos_sin_cache,
      compressed_kv_slots);
  auto [attention_output, stats, partial_output, partial_stats] =
      flashmla_splitkv_meta(
          q,
          state_storage_u8,
          compressed_geometry,
          topk_shape_buffer,
          split_shape_buffer,
          attention_token_to_req_indices,
          attention_block_table,
          is_valid_token,
          seq_lens,
          swa_storage_u8,
          swa_geometry,
          swa_indices,
          swa_lens,
          attn_sink,
          output_template);
  return std::make_tuple(
      attention_output,
      stats,
      qnorm_ack,
      compressor_ack,
      partial_output,
      partial_stats);
}

at::Tensor fill_short_topk_hpu(
    const at::Tensor& output,
    const at::Tensor& positions) {
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(
      output.dim() == 2 && output.size(1) == 512 && output.size(0) > 0);
  TORCH_CHECK(
      positions.dim() == 1 && positions.size(0) == output.size(0));
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(positions.is_contiguous());
  TORCH_CHECK(kFillShortTopkRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kFillShortTopkSchema);
  std::vector<c10::IValue> inputs{output, positions};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor fill_short_topk_meta(
    const at::Tensor& output,
    const at::Tensor& positions) {
  (void)positions;
  return at::empty(
      {output.size(0)},
      output.options().dtype(at::ScalarType::Int));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mxfp4_gather_hpu(
    const at::Tensor& expert_ids,
    const at::Tensor& w13,
    const at::Tensor& w2,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale) {
  TORCH_CHECK(expert_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(w13.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w2.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w13_scale.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w2_scale.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(expert_ids.sizes() == at::IntArrayRef({1, 6}));
  TORCH_CHECK(w13.sizes() == at::IntArrayRef({256, 2048, 2048}));
  TORCH_CHECK(w2.sizes() == at::IntArrayRef({256, 4096, 512}));
  TORCH_CHECK(w13_scale.sizes() == at::IntArrayRef({256, 2048, 128}));
  TORCH_CHECK(w2_scale.sizes() == at::IntArrayRef({256, 4096, 32}));
  TORCH_CHECK(expert_ids.is_contiguous());
  TORCH_CHECK(w13.is_contiguous());
  TORCH_CHECK(w2.is_contiguous());
  TORCH_CHECK(w13_scale.is_contiguous());
  TORCH_CHECK(w2_scale.is_contiguous());
  TORCH_CHECK(kMxfp4GatherRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kMxfp4GatherSchema);
  std::vector<c10::IValue> inputs{
      expert_ids, w13, w2, w13_scale, w2_scale};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 4);
  return std::make_tuple(
      outputs.at(0), outputs.at(1), outputs.at(2), outputs.at(3));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mxfp4_gather_meta(
    const at::Tensor& expert_ids,
    const at::Tensor& w13,
    const at::Tensor& w2,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale) {
  const auto topk = expert_ids.size(1);
  return std::make_tuple(
      at::empty(
          {topk, w13.size(1), w13.size(2)}, w13.options()),
      at::empty(
          {topk, w2.size(1), w2.size(2)}, w2.options()),
      at::empty(
          {topk, w13_scale.size(1), w13_scale.size(2)},
          w13_scale.options()),
      at::empty(
          {topk, w2_scale.size(1), w2_scale.size(2)},
          w2_scale.options()));
}

std::tuple<at::Tensor, at::Tensor> mxfp4_dequant_fp8_hpu(
    const at::Tensor& expert_ids,
    const at::Tensor& w13,
    const at::Tensor& w2,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale) {
  TORCH_CHECK(expert_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(w13.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w2.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w13_scale.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w2_scale.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(expert_ids.sizes() == at::IntArrayRef({1, 6}));
  TORCH_CHECK(w13.sizes() == at::IntArrayRef({256, 2048, 2048}));
  TORCH_CHECK(w2.sizes() == at::IntArrayRef({256, 4096, 512}));
  TORCH_CHECK(w13_scale.sizes() == at::IntArrayRef({256, 2048, 128}));
  TORCH_CHECK(w2_scale.sizes() == at::IntArrayRef({256, 4096, 32}));
  TORCH_CHECK(expert_ids.is_contiguous());
  TORCH_CHECK(w13.is_contiguous());
  TORCH_CHECK(w2.is_contiguous());
  TORCH_CHECK(w13_scale.is_contiguous());
  TORCH_CHECK(w2_scale.is_contiguous());
  TORCH_CHECK(kMxfp4DequantFp8Registered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kMxfp4DequantFp8Schema);
  std::vector<c10::IValue> inputs{
      expert_ids, w13, w2, w13_scale, w2_scale};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 2);
  return std::make_tuple(outputs.at(0), outputs.at(1));
}

std::tuple<at::Tensor, at::Tensor> mxfp4_dequant_fp8_meta(
    const at::Tensor& expert_ids,
    const at::Tensor& w13,
    const at::Tensor& w2,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale) {
  const auto topk = expert_ids.size(1);
  const auto fp8_options = w13.options().dtype(
      at::ScalarType::Float8_e4m3fn);
  return std::make_tuple(
      at::empty(
          {topk, w13.size(1), w13.size(2) * 2}, fp8_options),
      at::empty(
          {topk, w2.size(1), w2.size(2) * 2}, fp8_options));
}

at::Tensor mxfp4_indexed_moe_hpu(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_ids,
    const at::Tensor& router_weights,
    const at::Tensor& w13,
    const at::Tensor& w2,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale) {
  TORCH_CHECK(hidden_states.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(expert_ids.scalar_type() == at::ScalarType::Int);
  TORCH_CHECK(router_weights.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(w13.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w2.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w13_scale.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(w2_scale.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK(hidden_states.sizes() == at::IntArrayRef({1, 4096}));
  TORCH_CHECK(expert_ids.sizes() == at::IntArrayRef({1, 6}));
  TORCH_CHECK(router_weights.sizes() == at::IntArrayRef({1, 6}));
  TORCH_CHECK(w13.sizes() == at::IntArrayRef({256, 2048, 2048}));
  TORCH_CHECK(w2.sizes() == at::IntArrayRef({256, 4096, 512}));
  TORCH_CHECK(w13_scale.sizes() == at::IntArrayRef({256, 2048, 128}));
  TORCH_CHECK(w2_scale.sizes() == at::IntArrayRef({256, 4096, 32}));
  TORCH_CHECK(
      hidden_states.is_contiguous() && expert_ids.is_contiguous() &&
      router_weights.is_contiguous() && w13.is_contiguous() &&
      w2.is_contiguous() && w13_scale.is_contiguous() &&
      w2_scale.is_contiguous());
  TORCH_CHECK(kMxfp4IndexedMoeBackendRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kMxfp4IndexedMoeSchema);
  std::vector<c10::IValue> inputs{
      hidden_states,
      expert_ids,
      router_weights,
      w13,
      w2,
      w13_scale,
      w2_scale};
  auto outputs = descriptor.execute(inputs);
  TORCH_CHECK(outputs.size() == 1);
  return outputs.at(0);
}

at::Tensor mxfp4_indexed_moe_meta(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_ids,
    const at::Tensor& router_weights,
    const at::Tensor& w13,
    const at::Tensor& w2,
    const at::Tensor& w13_scale,
    const at::Tensor& w2_scale) {
  (void)expert_ids;
  (void)router_weights;
  (void)w13;
  (void)w2;
  (void)w13_scale;
  (void)w2_scale;
  return at::empty_like(hidden_states);
}

at::Tensor unwrap_functional_tensor(const at::Tensor& tensor) {
  if (!at::functionalization::impl::isFunctionalTensor(tensor)) {
    return tensor;
  }
  at::functionalization::impl::sync(tensor);
  return at::functionalization::impl::from_functional_tensor(tensor);
}

at::Tensor save_compress_norm_c4_noclone_functionalize(
    const at::Tensor& storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& cache_geometry,
    const at::Tensor& kv,
    const at::Tensor& score,
    const at::Tensor& ape,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& token_to_req_indices,
    const at::Tensor& block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& kv_slot_mapping) {
  auto storage = unwrap_functional_tensor(storage_u8);
  auto state_geometry_ = unwrap_functional_tensor(state_geometry);
  auto cache_geometry_ = unwrap_functional_tensor(cache_geometry);
  auto kv_ = unwrap_functional_tensor(kv);
  auto score_ = unwrap_functional_tensor(score);
  auto ape_ = unwrap_functional_tensor(ape);
  auto positions_ = unwrap_functional_tensor(positions);
  auto slot_mapping_ = unwrap_functional_tensor(slot_mapping);
  auto token_to_req_indices_ =
      unwrap_functional_tensor(token_to_req_indices);
  auto block_table_ = unwrap_functional_tensor(block_table);
  auto rms_norm_weight_ = unwrap_functional_tensor(rms_norm_weight);
  auto rms_norm_eps_ = unwrap_functional_tensor(rms_norm_eps);
  auto cos_sin_cache_ = unwrap_functional_tensor(cos_sin_cache);
  auto kv_slot_mapping_ = unwrap_functional_tensor(kv_slot_mapping);

  static auto op_handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "custom_op::custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2",
              "")
          .typed<at::Tensor(
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&)>();

  at::Tensor mutation_token;
  {
    at::AutoDispatchSkipFunctionalize guard;
    mutation_token = op_handle.call(
        storage,
        state_geometry_,
        cache_geometry_,
        kv_,
        score_,
        ape_,
        positions_,
        slot_mapping_,
        token_to_req_indices_,
        block_table_,
        rms_norm_weight_,
        rms_norm_eps_,
        cos_sin_cache_,
        kv_slot_mapping_);
  }

  at::functionalization::impl::replace_(storage_u8, storage);
  at::functionalization::impl::commit_update(storage_u8);
  at::functionalization::impl::sync(storage_u8);
  return mutation_token;
}

std::tuple<at::Tensor, at::Tensor>
qnorm_compressor_c4_functionalize(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& swa_storage_u8,
    const at::Tensor& swa_geometry,
    const at::Tensor& swa_slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& state_storage_u8,
    const at::Tensor& state_geometry,
    const at::Tensor& compressed_geometry,
    const at::Tensor& compressor_kv,
    const at::Tensor& compressor_score,
    const at::Tensor& ape,
    const at::Tensor& state_slots,
    const at::Tensor& state_token_to_req_indices,
    const at::Tensor& state_block_table,
    const at::Tensor& rms_norm_weight,
    const at::Tensor& rms_norm_eps,
    const at::Tensor& compressed_kv_slots,
    const at::Tensor& completion_input) {
  auto q_ = unwrap_functional_tensor(q);
  auto kv_ = unwrap_functional_tensor(kv);
  auto swa_storage = unwrap_functional_tensor(swa_storage_u8);
  auto swa_geometry_ = unwrap_functional_tensor(swa_geometry);
  auto swa_slots_ = unwrap_functional_tensor(swa_slots);
  auto positions_ = unwrap_functional_tensor(positions);
  auto cos_sin_cache_ = unwrap_functional_tensor(cos_sin_cache);
  auto state_storage = unwrap_functional_tensor(state_storage_u8);
  auto state_geometry_ = unwrap_functional_tensor(state_geometry);
  auto compressed_geometry_ = unwrap_functional_tensor(compressed_geometry);
  auto compressor_kv_ = unwrap_functional_tensor(compressor_kv);
  auto compressor_score_ = unwrap_functional_tensor(compressor_score);
  auto ape_ = unwrap_functional_tensor(ape);
  auto state_slots_ = unwrap_functional_tensor(state_slots);
  auto state_token_to_req_indices_ =
      unwrap_functional_tensor(state_token_to_req_indices);
  auto state_block_table_ = unwrap_functional_tensor(state_block_table);
  auto rms_norm_weight_ = unwrap_functional_tensor(rms_norm_weight);
  auto rms_norm_eps_ = unwrap_functional_tensor(rms_norm_eps);
  auto compressed_kv_slots_ =
      unwrap_functional_tensor(compressed_kv_slots);
  auto completion_input_ = unwrap_functional_tensor(completion_input);

  static auto op_handle =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow(
              "custom_op::custom_deepseek_v4_qnorm_compressor_c4_f32_noclone_gaudi2",
              "")
          .typed<std::tuple<at::Tensor, at::Tensor>(
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&,
              const at::Tensor&)>();

  std::tuple<at::Tensor, at::Tensor> completion;
  {
    at::AutoDispatchSkipFunctionalize guard;
    completion = op_handle.call(
        q_,
        kv_,
        swa_storage,
        swa_geometry_,
        swa_slots_,
        positions_,
        cos_sin_cache_,
        state_storage,
        state_geometry_,
        compressed_geometry_,
        compressor_kv_,
        compressor_score_,
        ape_,
        state_slots_,
        state_token_to_req_indices_,
        state_block_table_,
        rms_norm_weight_,
        rms_norm_eps_,
        compressed_kv_slots_,
        completion_input_);
  }

  for (const auto& update : {
           std::pair<const at::Tensor*, const at::Tensor*>{&q, &q_},
           {&swa_storage_u8, &swa_storage},
           {&state_storage_u8, &state_storage}}) {
    at::functionalization::impl::replace_(*update.first, *update.second);
    at::functionalization::impl::commit_update(*update.first);
    at::functionalization::impl::sync(*update.first);
  }
  return completion;
}

// The mutable public contract is functionalized without cloning the KV pool.
// The private form is consumed only with the returned completion dependency.
at::Tensor native_qnorm_functionalize(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& cache_storage_u8,
    const at::Tensor& cache_geometry,
    const at::Tensor& slots,
    const at::Tensor& positions,
    const at::Tensor& cos_sin_cache) {
  auto q_ = unwrap_functional_tensor(q);
  auto kv_ = unwrap_functional_tensor(kv);
  auto cache_storage_u8_ = unwrap_functional_tensor(cache_storage_u8);
  auto cache_geometry_ = unwrap_functional_tensor(cache_geometry);
  auto slots_ = unwrap_functional_tensor(slots);
  auto positions_ = unwrap_functional_tensor(positions);
  auto cos_sin_cache_ = unwrap_functional_tensor(cos_sin_cache);
  static auto handle = c10::Dispatcher::singleton()
      .findSchemaOrThrow("custom_op::custom_deepseek_v4_native_qnorm_rope_kv_pack_ordered_bf16_gaudi2", "")
      .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>();
  at::Tensor completion;
  {
    at::AutoDispatchSkipFunctionalize guard;
    completion = handle.call(q_, kv_, cache_storage_u8_, cache_geometry_, slots_, positions_, cos_sin_cache_);
  }
  for (const auto& update : {
           std::pair<const at::Tensor*, const at::Tensor*>{&q, &q_},
           {&cache_storage_u8, &cache_storage_u8_}}) {
    at::functionalization::impl::replace_(*update.first, *update.second);
    at::functionalization::impl::commit_update(*update.first);
    at::functionalization::impl::sync(*update.first);
  }
  return completion;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2("
        "Tensor(a!) q, Tensor kv, Tensor(b!) cache_storage_u8, "
        "Tensor cache_geometry, Tensor slots, Tensor positions, Tensor cos_sin_cache) -> Tensor");
  m.def("custom_deepseek_v4_native_qnorm_rope_kv_pack_ordered_bf16_gaudi2("
        "Tensor q, Tensor kv, Tensor cache_storage_u8, "
        "Tensor cache_geometry, Tensor slots, Tensor positions, Tensor cos_sin_cache) -> Tensor");
  m.def(
      "custom_deepseek_v4_sparse_attn_bf16_gaudi2("
      "Tensor q, Tensor kv, Tensor indices, Tensor attn_sink, "
      "Tensor softmax_scale) -> (Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2("
      "Tensor q, Tensor kv, Tensor indices, Tensor attn_sink, "
      "Tensor softmax_scale, Tensor topk_lengths) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_dequant_gather_bf16_gaudi2("
      "Tensor cache_storage_u8, Tensor cache_geometry, Tensor indices) "
      "-> Tensor");
  m.def(
      "custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2("
      "Tensor first_storage_u8, Tensor first_geometry, "
      "Tensor first_indices, Tensor second_storage_u8, "
      "Tensor second_geometry, Tensor second_indices) -> Tensor");
  m.def(
      "custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2("
      "Tensor first_storage_u8, Tensor first_geometry, "
      "Tensor first_shape_buffer, Tensor token_to_req_indices, "
      "Tensor block_table, Tensor is_valid_token, Tensor seq_lens, "
      "Tensor second_storage_u8, Tensor second_geometry, "
      "Tensor second_indices) -> Tensor");
  m.def(
      "custom_deepseek_v4_save_partial_states_f32_gaudi2("
      "Tensor(a!) state_storage, Tensor state_geometry, Tensor kv, "
      "Tensor score, Tensor ape, Tensor positions, Tensor slot_mapping) "
      "-> Tensor");
  m.def(
      "custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2("
      "Tensor(a!) storage_u8, Tensor state_geometry, "
      "Tensor cache_geometry, Tensor kv, "
      "Tensor score, Tensor ape, Tensor positions, Tensor slot_mapping, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor rms_norm_weight, Tensor rms_norm_eps, Tensor cos_sin_cache, "
      "Tensor kv_slot_mapping"
      ") -> Tensor");
  // Internal alias-free form used by the custom Functionalize implementation.
  m.def(
      "custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2("
      "Tensor storage_u8, Tensor state_geometry, "
      "Tensor cache_geometry, Tensor kv, "
      "Tensor score, Tensor ape, Tensor positions, Tensor slot_mapping, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor rms_norm_weight, Tensor rms_norm_eps, Tensor cos_sin_cache, "
      "Tensor kv_slot_mapping"
      ") -> Tensor");
  m.def(
      "custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2("
      "Tensor(a!) storage_u8, Tensor state_geometry, "
      "Tensor cache_geometry, Tensor kv, "
      "Tensor score, Tensor ape, Tensor positions, Tensor slot_mapping, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor rms_norm_weight, Tensor rms_norm_eps, Tensor cos_sin_cache, "
      "Tensor kv_slot_mapping"
      ") -> Tensor");
  m.def(
      "custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2("
      "Tensor(a!) storage_u8, Tensor state_geometry, "
      "Tensor cache_geometry, Tensor kv, "
      "Tensor score, Tensor ape, Tensor positions, Tensor slot_mapping, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor rms_norm_weight, Tensor rms_norm_eps, Tensor cos_sin_cache, "
      "Tensor kv_slot_mapping"
      ") -> Tensor");
  m.def(
      "custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2("
      "Tensor(a!) storage_u8, Tensor state_geometry, "
      "Tensor cache_geometry, Tensor kv, "
      "Tensor score, Tensor ape, Tensor positions, Tensor slot_mapping, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor rms_norm_weight, Tensor rms_norm_eps, Tensor cos_sin_cache, "
      "Tensor kv_slot_mapping"
      ") -> Tensor");
  m.def(
      "custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2("
      "Tensor(a!) q, Tensor kv, Tensor(b!) cache_storage_u8, "
      "Tensor cache_geometry, Tensor slots, Tensor positions, "
      "Tensor cos_sin_cache) -> Tensor");
  m.def(
      "custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2("
      "Tensor(a!) q, Tensor kv, Tensor(b!) cache_storage_u8, "
      "Tensor cache_geometry, Tensor slots, Tensor positions, "
      "Tensor cos_sin_cache) -> Tensor");
  m.def(
      "custom_deepseek_v4_insert_packed_kv_u8_gaudi2("
      "Tensor(a!) cache_storage_u8, Tensor cache_geometry, Tensor packed, "
      "Tensor slots) -> Tensor");
  m.def(
      "custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor topk_indices, Tensor topk_lens, "
      "Tensor swa_storage_u8, Tensor swa_geometry, Tensor swa_indices, "
      "Tensor swa_lens, Tensor attn_sink, Tensor(a!) output) -> Tensor");
  m.def(
      "custom_deepseek_v4_paged_swa_attn_fp8_gaudi2("
      "Tensor q, Tensor dummy_storage_u8, Tensor dummy_geometry, "
      "Tensor dummy_topk_shape, Tensor dummy_token_to_req, "
      "Tensor dummy_block_table, Tensor dummy_valid_token, "
      "Tensor dummy_seq_lens, Tensor swa_storage_u8, Tensor swa_geometry, "
      "Tensor swa_indices, Tensor swa_lens, Tensor attn_sink, "
      "Tensor(a!) output) -> Tensor");
  m.def(
      "custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor local_topk_indices, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor is_valid_token, Tensor swa_storage_u8, Tensor swa_geometry, "
      "Tensor swa_indices, Tensor swa_lens, Tensor attn_sink, "
      "Tensor(a!) output) -> Tensor");
  m.def(
      "custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor topk_shape_buffer, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor is_valid_token, Tensor seq_lens, Tensor swa_storage_u8, "
      "Tensor swa_geometry, Tensor swa_indices, Tensor swa_lens, "
      "Tensor attn_sink, Tensor(a!) output) -> Tensor");
  m.def(
      "custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor topk_shape_buffer, "
      "Tensor token_to_req_indices, Tensor block_table, "
      "Tensor is_valid_token, Tensor seq_lens, Tensor swa_storage_u8, "
      "Tensor swa_geometry, Tensor swa_indices, Tensor swa_lens, "
      "Tensor attn_sink, Tensor(a!) output) -> Tensor");
  m.def(
      "custom_deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor topk_shape_buffer, "
      "Tensor split_shape_buffer, Tensor token_to_req_indices, "
      "Tensor block_table, Tensor is_valid_token, Tensor seq_lens, "
      "Tensor swa_storage_u8, Tensor swa_geometry, Tensor swa_indices, "
      "Tensor swa_lens) -> (Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_flashmla_splitkv_combine_gaudi2("
      "Tensor partial_output, Tensor partial_stats, Tensor attn_sink) "
      "-> (Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_flashmla_splitkv_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor topk_shape_buffer, "
      "Tensor split_shape_buffer, Tensor token_to_req_indices, "
      "Tensor block_table, Tensor is_valid_token, Tensor seq_lens, "
      "Tensor swa_storage_u8, Tensor swa_geometry, Tensor swa_indices, "
      "Tensor swa_lens, Tensor attn_sink, Tensor output_template) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_flashmla_splitkv_tiled_fp8_gaudi2("
      "Tensor q, Tensor compressed_storage_u8, "
      "Tensor compressed_geometry, Tensor topk_shape_buffer, "
      "Tensor split_shape_buffer, Tensor token_to_req_indices, "
      "Tensor block_table, Tensor is_valid_token, Tensor seq_lens, "
      "Tensor swa_storage_u8, Tensor swa_geometry, Tensor swa_indices, "
      "Tensor swa_lens, Tensor attn_sink, Tensor output_template) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_compress_flashmla_c4_f32_gaudi2("
      "Tensor storage_u8, Tensor state_geometry, "
      "Tensor compressed_geometry, Tensor compressor_kv, "
      "Tensor compressor_score, Tensor ape, Tensor positions, "
      "Tensor state_slots, Tensor state_token_to_req_indices, "
      "Tensor state_block_table, Tensor rms_norm_weight, "
      "Tensor rms_norm_eps, Tensor cos_sin_cache, "
      "Tensor compressed_kv_slots, Tensor q, "
      "Tensor topk_shape_buffer, Tensor split_shape_buffer, "
      "Tensor attention_token_to_req_indices, "
      "Tensor attention_block_table, Tensor is_valid_token, "
      "Tensor seq_lens, Tensor swa_storage_u8, Tensor swa_geometry, "
      "Tensor swa_indices, Tensor swa_lens, Tensor attn_sink, "
      "Tensor output_template) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2("
      "Tensor q, Tensor positions, Tensor cos_sin_cache, "
      "Tensor compressed_storage_u8, Tensor compressed_geometry, "
      "Tensor topk_shape_buffer, Tensor block_table, "
      "Tensor is_valid_token, Tensor seq_lens, "
      "Tensor swa_storage_u8, Tensor swa_geometry, Tensor swa_indices, "
      "Tensor swa_lens, Tensor attn_sink, Tensor(a!) output) -> Tensor");
  m.def(
      "custom_deepseek_v4_c4_decode_chain_gaudi2("
      "Tensor(a!) q, Tensor kv, Tensor(b!) swa_storage_u8, "
      "Tensor swa_geometry, Tensor swa_slots, Tensor positions, "
      "Tensor cos_sin_cache, Tensor(c!) state_storage_u8, "
      "Tensor state_geometry, Tensor compressed_geometry, "
      "Tensor compressor_kv, Tensor compressor_score, Tensor ape, "
      "Tensor state_slots, Tensor state_token_to_req_indices, "
      "Tensor state_block_table, Tensor rms_norm_weight, "
      "Tensor rms_norm_eps, Tensor compressed_kv_slots, "
      "Tensor topk_shape_buffer, "
      "Tensor split_shape_buffer, Tensor attention_token_to_req_indices, "
      "Tensor attention_block_table, Tensor is_valid_token, "
      "Tensor seq_lens, Tensor swa_indices, Tensor swa_lens, "
      "Tensor attn_sink, Tensor output_template) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_gaudi2("
      "Tensor(a!) q, Tensor kv, Tensor(b!) swa_storage_u8, "
      "Tensor swa_geometry, Tensor swa_slots, Tensor positions, "
      "Tensor cos_sin_cache, Tensor(c!) state_storage_u8, "
      "Tensor state_geometry, Tensor compressed_geometry, "
      "Tensor compressor_kv, Tensor compressor_score, Tensor ape, "
      "Tensor state_slots, Tensor state_token_to_req_indices, "
      "Tensor state_block_table, Tensor rms_norm_weight, "
      "Tensor rms_norm_eps, Tensor compressed_kv_slots, "
      "Tensor completion_input) "
      "-> (Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_noclone_gaudi2("
      "Tensor q, Tensor kv, Tensor swa_storage_u8, "
      "Tensor swa_geometry, Tensor swa_slots, Tensor positions, "
      "Tensor cos_sin_cache, Tensor state_storage_u8, "
      "Tensor state_geometry, Tensor compressed_geometry, "
      "Tensor compressor_kv, Tensor compressor_score, Tensor ape, "
      "Tensor state_slots, Tensor state_token_to_req_indices, "
      "Tensor state_block_table, Tensor rms_norm_weight, "
      "Tensor rms_norm_eps, Tensor compressed_kv_slots, "
      "Tensor completion_input) "
      "-> (Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_fill_short_topk_i32_gaudi2("
      "Tensor(a!) output, Tensor positions) -> Tensor");
  m.def(
      "custom_deepseek_v4_mxfp4_gather_u8_gaudi2("
      "Tensor expert_ids, Tensor w13, Tensor w2, Tensor w13_scale, "
      "Tensor w2_scale) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_mxfp4_dequant_fp8_gaudi2("
      "Tensor expert_ids, Tensor w13, Tensor w2, Tensor w13_scale, "
      "Tensor w2_scale) -> (Tensor, Tensor)");
  m.def(
      "custom_deepseek_v4_mxfp4_indexed_moe_gaudi2("
      "Tensor hidden_states, Tensor expert_ids, Tensor router_weights, "
      "Tensor w13, Tensor w2, Tensor w13_scale, Tensor w2_scale) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2", hybrid_qnorm_rope_kv_pack_hpu);
  m.impl("custom_deepseek_v4_native_qnorm_rope_kv_pack_ordered_bf16_gaudi2", hybrid_qnorm_rope_kv_pack_hpu);
  m.impl(
      "custom_deepseek_v4_sparse_attn_bf16_gaudi2",
      sparse_attention_hpu);
  m.impl(
      "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2",
      sparse_attention_lengths_hpu);
  m.impl(
      "custom_deepseek_v4_dequant_gather_bf16_gaudi2",
      dequant_gather_hpu);
  m.impl(
      "custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2",
      dual_dequant_gather_hpu);
  m.impl(
      "custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2",
      local_dual_dequant_gather_hpu);
  m.impl(
      "custom_deepseek_v4_save_partial_states_f32_gaudi2",
      save_partial_states_hpu);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2",
      save_compress_norm_c4_hpu);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2",
      save_compress_norm_c4_hpu);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2",
      save_compress_norm_c4_hpu);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2",
      save_compress_norm_c4_bf16_hpu);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2",
      save_compress_norm_c4_mixed_hpu);
  m.impl(
      "custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2",
      qnorm_rope_kv_pack_hpu);
  m.impl(
      "custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2",
      hybrid_qnorm_rope_kv_pack_hpu);
  m.impl(
      "custom_deepseek_v4_insert_packed_kv_u8_gaudi2",
      insert_packed_kv_hpu);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2",
      paged_sparse_attention_hpu);
  m.impl(
      "custom_deepseek_v4_paged_swa_attn_fp8_gaudi2",
      paged_swa_attention_hpu);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2",
      paged_sparse_attention_local_hpu);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2",
      paged_sparse_attention_sequential_hpu);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2",
      paged_sparse_attention_pair_sequential_hpu);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2",
      flashmla_splitkv_partial_hpu);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_combine_gaudi2",
      flashmla_splitkv_combine_hpu);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_fp8_gaudi2",
      flashmla_splitkv_hpu);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_tiled_fp8_gaudi2",
      flashmla_splitkv_tiled_hpu);
  m.impl(
      "custom_deepseek_v4_compress_flashmla_c4_f32_gaudi2",
      compress_flashmla_c4_hpu);
  m.impl(
      "custom_deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2",
      qnorm_paged_sparse_attention_sequential_hpu);
  m.impl(
      "custom_deepseek_v4_c4_decode_chain_gaudi2",
      c4_decode_chain_hpu);
  m.impl(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_gaudi2",
      qnorm_compressor_c4_hpu);
  m.impl(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_noclone_gaudi2",
      qnorm_compressor_c4_hpu);
  m.impl(
      "custom_deepseek_v4_fill_short_topk_i32_gaudi2",
      fill_short_topk_hpu);
  m.impl(
      "custom_deepseek_v4_mxfp4_gather_u8_gaudi2",
      mxfp4_gather_hpu);
  m.impl(
      "custom_deepseek_v4_mxfp4_dequant_fp8_gaudi2",
      mxfp4_dequant_fp8_hpu);
  m.impl(
      "custom_deepseek_v4_mxfp4_indexed_moe_gaudi2",
      mxfp4_indexed_moe_hpu);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2", qnorm_rope_kv_pack_meta);
  m.impl("custom_deepseek_v4_native_qnorm_rope_kv_pack_ordered_bf16_gaudi2", qnorm_rope_kv_pack_meta);
  m.impl(
      "custom_deepseek_v4_sparse_attn_bf16_gaudi2",
      sparse_attention_meta);
  m.impl(
      "custom_deepseek_v4_sparse_attn_bf16_lengths_gaudi2",
      sparse_attention_lengths_meta);
  m.impl(
      "custom_deepseek_v4_dequant_gather_bf16_gaudi2",
      dequant_gather_meta);
  m.impl(
      "custom_deepseek_v4_dual_dequant_gather_bf16_gaudi2",
      dual_dequant_gather_meta);
  m.impl(
      "custom_deepseek_v4_local_dual_dequant_gather_bf16_gaudi2",
      local_dual_dequant_gather_meta);
  m.impl(
      "custom_deepseek_v4_save_partial_states_f32_gaudi2",
      save_partial_states_meta);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_gaudi2",
      save_compress_norm_c4_meta);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_ordered_gaudi2",
      save_compress_norm_c4_meta);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2",
      save_compress_norm_c4_meta);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_bf16_gaudi2",
      save_compress_norm_c4_meta);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_mixed_gaudi2",
      save_compress_norm_c4_meta);
  m.impl(
      "custom_deepseek_v4_qnorm_rope_kv_pack_bf16_gaudi2",
      qnorm_rope_kv_pack_meta);
  m.impl(
      "custom_deepseek_v4_hybrid_qnorm_rope_kv_pack_bf16_gaudi2",
      qnorm_rope_kv_pack_meta);
  m.impl(
      "custom_deepseek_v4_insert_packed_kv_u8_gaudi2",
      insert_packed_kv_meta);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_fp8_gaudi2",
      paged_sparse_attention_meta);
  m.impl(
      "custom_deepseek_v4_paged_swa_attn_fp8_gaudi2",
      paged_sparse_attention_sequential_meta);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_local_fp8_gaudi2",
      paged_sparse_attention_local_meta);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_sequential_fp8_gaudi2",
      paged_sparse_attention_sequential_meta);
  m.impl(
      "custom_deepseek_v4_paged_sparse_attn_pair_sequential_fp8_gaudi2",
      paged_sparse_attention_sequential_meta);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_partial_fp8_gaudi2",
      flashmla_splitkv_partial_meta);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_combine_gaudi2",
      flashmla_splitkv_combine_meta);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_fp8_gaudi2",
      flashmla_splitkv_meta);
  m.impl(
      "custom_deepseek_v4_flashmla_splitkv_tiled_fp8_gaudi2",
      flashmla_splitkv_meta);
  m.impl(
      "custom_deepseek_v4_compress_flashmla_c4_f32_gaudi2",
      compress_flashmla_c4_meta);
  m.impl(
      "custom_deepseek_v4_qnorm_paged_sparse_attn_sequential_fp8_gaudi2",
      qnorm_paged_sparse_attention_sequential_meta);
  m.impl(
      "custom_deepseek_v4_c4_decode_chain_gaudi2",
      c4_decode_chain_meta);
  m.impl(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_gaudi2",
      qnorm_compressor_c4_meta);
  m.impl(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_noclone_gaudi2",
      qnorm_compressor_c4_meta);
  m.impl(
      "custom_deepseek_v4_fill_short_topk_i32_gaudi2",
      fill_short_topk_meta);
  m.impl(
      "custom_deepseek_v4_mxfp4_gather_u8_gaudi2",
      mxfp4_gather_meta);
  m.impl(
      "custom_deepseek_v4_mxfp4_dequant_fp8_gaudi2",
      mxfp4_dequant_fp8_meta);
  m.impl(
      "custom_deepseek_v4_mxfp4_indexed_moe_gaudi2",
      mxfp4_indexed_moe_meta);
}

TORCH_LIBRARY_IMPL(custom_op, Functionalize, m) {
  m.impl("custom_deepseek_v4_native_qnorm_rope_kv_pack_bf16_gaudi2", native_qnorm_functionalize);
  m.impl(
      "custom_deepseek_v4_save_compress_norm_c4_f32_noclone_gaudi2",
      save_compress_norm_c4_noclone_functionalize);
  m.impl(
      "custom_deepseek_v4_qnorm_compressor_c4_f32_gaudi2",
      qnorm_compressor_c4_functionalize);
}
