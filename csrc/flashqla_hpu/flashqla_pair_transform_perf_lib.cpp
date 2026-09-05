#include <cstring>
#include <initializer_list>

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

extern "C" unsigned char
    _binary_flashqla_pair_transform_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_flashqla_pair_transform_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_flashqla_recurrent_prefill_f32_gaudi2_o_start;
extern "C" unsigned char
    _binary_flashqla_recurrent_prefill_f32_gaudi2_o_end;
extern "C" unsigned char
    _binary_flashqla_unit_lower_inverse16_f32_gaudi2_o_start;
extern "C" unsigned char
    _binary_flashqla_unit_lower_inverse16_f32_gaudi2_o_end;
extern "C" unsigned char
    _binary_flashqla_kkt_form_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_flashqla_kkt_form_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_qwen38_compact_kkt_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_qwen38_compact_kkt_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_qwen38_post_conv_qk_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_qwen38_post_conv_qk_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_qwen38_post_conv_qk_compact_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_qwen38_post_conv_qk_compact_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_qwen38_post_conv_qk_expanded_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_qwen38_post_conv_qk_expanded_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_qwen38_conv_qkv_prep_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_qwen38_conv_qkv_prep_bf16_gaudi2_o_end;
extern "C" unsigned char
    _binary_qwen38_rmsnorm_gated_bf16_gaudi2_o_start;
extern "C" unsigned char
    _binary_qwen38_rmsnorm_gated_bf16_gaudi2_o_end;

namespace {

constexpr char kKernelName[] =
    "flashqla_pair_transform_bf16_gaudi2";
constexpr char kRecurrentKernelName[] =
    "flashqla_recurrent_prefill_f32_gaudi2";
constexpr char kInverse16KernelName[] =
    "flashqla_unit_lower_inverse16_f32_gaudi2";
constexpr char kKktFormKernelName[] =
    "flashqla_kkt_form_bf16_gaudi2";
constexpr char kCompactKktKernelName[] =
    "qwen38_compact_kkt_bf16_gaudi2";
constexpr char kPostConvKernelName[] =
    "qwen38_post_conv_qk_bf16_gaudi2";
constexpr char kCompactPostConvKernelName[] =
    "qwen38_post_conv_qk_compact_bf16_gaudi2";
constexpr char kBf16PostConvKernelName[] =
    "qwen38_post_conv_qk_expanded_bf16_gaudi2";
constexpr char kConvQkvKernelName[] =
    "qwen38_conv_qkv_prep_bf16_gaudi2";
constexpr char kRmsNormGatedKernelName[] =
    "qwen38_rmsnorm_gated_bf16_gaudi2";

void set_mapping(
    tpc_lib_api::TensorAccessPattern& pattern,
    unsigned tensor_dim,
    unsigned index_dim,
    int a,
    int start_b,
    int end_b) {
  pattern.mapping[tensor_dim].indexSpaceDim = index_dim;
  pattern.mapping[tensor_dim].a = a;
  pattern.mapping[tensor_dim].start_b = start_b;
  pattern.mapping[tensor_dim].end_b = end_b;
}

bool has_matrix_shape(const tpc_lib_api::TensorGeometry& geometry) {
  return geometry.dims == 4 && geometry.maxSizes[0] == 64 &&
      geometry.maxSizes[1] == 64;
}

bool has_vector_shape(
    const tpc_lib_api::TensorGeometry& geometry,
    uint64_t heads,
    uint64_t outer) {
  return geometry.dims == 3 && geometry.maxSizes[0] == 64 &&
      geometry.maxSizes[1] == heads && geometry.maxSizes[2] == outer;
}

bool has_half_vector_shape(
    const tpc_lib_api::TensorGeometry& geometry,
    uint64_t heads,
    uint64_t outer) {
  return geometry.dims == 3 && geometry.maxSizes[0] == 32 &&
      geometry.maxSizes[1] == heads && geometry.maxSizes[2] == outer;
}

tpc_lib_api::GlueCodeReturn copy_elf(
    tpc_lib_api::HabanaKernelInstantiation* instance,
    const unsigned char* elf_start,
    const unsigned char* elf_end) {
  const unsigned elf_size = static_cast<unsigned>(elf_end - elf_start);
  const unsigned available_size = instance->kernel.elfSize;
  instance->kernel.elfSize = elf_size;
  if (available_size < elf_size) {
    return tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER;
  }
  std::memcpy(instance->kernel.kernelElf, elf_start, elf_size);
  return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn instantiate_pair_transform(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 5) {
    params->inputTensorNr = 5;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 2) {
    params->outputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& a0 = params->inputTensors[0].geometry;
  auto& scores = params->inputTensors[1].geometry;
  auto& g_even = params->inputTensors[2].geometry;
  auto& g_odd = params->inputTensors[3].geometry;
  auto& beta = params->inputTensors[4].geometry;
  auto& ag = params->outputTensors[0].geometry;
  auto& attention = params->outputTensors[1].geometry;
  const uint64_t heads = a0.maxSizes[2];
  const uint64_t outer = a0.maxSizes[3];

  if (!has_matrix_shape(a0) || !has_matrix_shape(scores) ||
      !has_matrix_shape(ag) || !has_matrix_shape(attention) ||
      scores.maxSizes[2] != heads || scores.maxSizes[3] != outer ||
      ag.maxSizes[2] != heads || ag.maxSizes[3] != outer ||
      attention.maxSizes[2] != heads || attention.maxSizes[3] != outer ||
      !has_half_vector_shape(g_even, heads, outer) ||
      !has_half_vector_shape(g_odd, heads, outer) ||
      !has_vector_shape(beta, heads, outer)) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }

  if (a0.dataType != tpc_lib_api::DATA_BF16 ||
      scores.dataType != tpc_lib_api::DATA_BF16 ||
      g_even.dataType != tpc_lib_api::DATA_F32 ||
      g_odd.dataType != tpc_lib_api::DATA_F32 ||
      beta.dataType != tpc_lib_api::DATA_BF16 ||
      ag.dataType != tpc_lib_api::DATA_BF16 ||
      attention.dataType != tpc_lib_api::DATA_BF16) {
    a0.dataType = tpc_lib_api::DATA_BF16;
    scores.dataType = tpc_lib_api::DATA_BF16;
    g_even.dataType = tpc_lib_api::DATA_F32;
    g_odd.dataType = tpc_lib_api::DATA_F32;
    beta.dataType = tpc_lib_api::DATA_BF16;
    ag.dataType = tpc_lib_api::DATA_BF16;
    attention.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 3;
  instance->indexSpaceGeometry[0] = 64;
  instance->indexSpaceGeometry[1] = heads;
  instance->indexSpaceGeometry[2] = outer;

  for (unsigned input : {0U, 1U}) {
    set_mapping(
        instance->inputTensorAccessPattern[input], 0, 0, 0, 0, 63);
    set_mapping(
        instance->inputTensorAccessPattern[input], 1, 0, 1, 0, 0);
    set_mapping(
        instance->inputTensorAccessPattern[input], 2, 1, 1, 0, 0);
    set_mapping(
        instance->inputTensorAccessPattern[input], 3, 2, 1, 0, 0);
  }
  for (unsigned input : {2U, 3U}) {
    set_mapping(
        instance->inputTensorAccessPattern[input], 0, 0, 0, 0, 31);
    set_mapping(
        instance->inputTensorAccessPattern[input], 1, 1, 1, 0, 0);
    set_mapping(
        instance->inputTensorAccessPattern[input], 2, 2, 1, 0, 0);
  }
  set_mapping(
      instance->inputTensorAccessPattern[4], 0, 0, 0, 0, 63);
  set_mapping(
      instance->inputTensorAccessPattern[4], 1, 1, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[4], 2, 2, 1, 0, 0);
  for (unsigned output : {0U, 1U}) {
    set_mapping(
        instance->outputTensorAccessPattern[output], 0, 0, 0, 0, 63);
    set_mapping(
        instance->outputTensorAccessPattern[output], 1, 0, 1, 0, 0);
    set_mapping(
        instance->outputTensorAccessPattern[output], 2, 1, 1, 0, 0);
    set_mapping(
        instance->outputTensorAccessPattern[output], 3, 2, 1, 0, 0);
  }

  return copy_elf(
      instance,
      &_binary_flashqla_pair_transform_bf16_gaudi2_o_start,
      &_binary_flashqla_pair_transform_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_recurrent_prefill(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 6) {
    params->inputTensorNr = 6;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 2) {
    params->outputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& q = params->inputTensors[0].geometry;
  auto& k = params->inputTensors[1].geometry;
  auto& v = params->inputTensors[2].geometry;
  auto& decay = params->inputTensors[3].geometry;
  auto& beta = params->inputTensors[4].geometry;
  auto& initial_state = params->inputTensors[5].geometry;
  auto& output = params->outputTensors[0].geometry;
  auto& final_state = params->outputTensors[1].geometry;

  const uint64_t heads = q.maxSizes[1];
  const uint64_t tokens = q.maxSizes[2];
  const bool q_shape = q.dims == 3 && q.maxSizes[0] == 128;
  const bool token_shape =
      k.dims == 3 && k.maxSizes[0] == 128 &&
      k.maxSizes[1] == heads && k.maxSizes[2] == tokens &&
      v.dims == 3 && v.maxSizes[0] == 128 &&
      v.maxSizes[1] == heads && v.maxSizes[2] == tokens &&
      output.dims == 3 && output.maxSizes[0] == 128 &&
      output.maxSizes[1] == heads && output.maxSizes[2] == tokens;
  const bool scalar_shape =
      decay.dims == 2 && decay.maxSizes[0] == heads &&
      decay.maxSizes[1] == tokens &&
      beta.dims == 2 && beta.maxSizes[0] == heads &&
      beta.maxSizes[1] == tokens;
  const bool state_shape =
      initial_state.dims == 3 && initial_state.maxSizes[0] == 128 &&
      initial_state.maxSizes[1] == 128 &&
      initial_state.maxSizes[2] == heads &&
      final_state.dims == 3 && final_state.maxSizes[0] == 128 &&
      final_state.maxSizes[1] == 128 &&
      final_state.maxSizes[2] == heads;
  if (!q_shape || !token_shape || !scalar_shape || !state_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }

  for (unsigned input = 0; input < 6; ++input) {
    if (params->inputTensors[input].geometry.dataType !=
        tpc_lib_api::DATA_F32) {
      params->inputTensors[input].geometry.dataType =
          tpc_lib_api::DATA_F32;
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
  }
  for (unsigned output_index = 0; output_index < 2; ++output_index) {
    if (params->outputTensors[output_index].geometry.dataType !=
        tpc_lib_api::DATA_F32) {
      params->outputTensors[output_index].geometry.dataType =
          tpc_lib_api::DATA_F32;
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
  }

  instance->indexSpaceRank = 2;
  instance->indexSpaceGeometry[0] = 8;
  instance->indexSpaceGeometry[1] = heads;

  for (unsigned input : {0U, 1U}) {
    set_mapping(
        instance->inputTensorAccessPattern[input], 0, 0, 0, 0, 127);
    set_mapping(
        instance->inputTensorAccessPattern[input], 1, 1, 1, 0, 0);
    set_mapping(
        instance->inputTensorAccessPattern[input],
        2,
        0,
        0,
        0,
        static_cast<int>(tokens - 1));
  }

  set_mapping(instance->inputTensorAccessPattern[2], 0, 0, 16, 0, 15);
  set_mapping(instance->inputTensorAccessPattern[2], 1, 1, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[2],
      2,
      0,
      0,
      0,
      static_cast<int>(tokens - 1));

  for (unsigned input : {3U, 4U}) {
    set_mapping(
        instance->inputTensorAccessPattern[input], 0, 1, 1, 0, 0);
    set_mapping(
        instance->inputTensorAccessPattern[input],
        1,
        0,
        0,
        0,
        static_cast<int>(tokens - 1));
  }

  set_mapping(instance->inputTensorAccessPattern[5], 0, 0, 0, 0, 127);
  set_mapping(instance->inputTensorAccessPattern[5], 1, 0, 16, 0, 15);
  set_mapping(instance->inputTensorAccessPattern[5], 2, 1, 1, 0, 0);

  set_mapping(instance->outputTensorAccessPattern[0], 0, 0, 16, 0, 15);
  set_mapping(instance->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
  set_mapping(
      instance->outputTensorAccessPattern[0],
      2,
      0,
      0,
      0,
      static_cast<int>(tokens - 1));
  set_mapping(instance->outputTensorAccessPattern[1], 0, 0, 0, 0, 127);
  set_mapping(instance->outputTensorAccessPattern[1], 1, 0, 16, 0, 15);
  set_mapping(instance->outputTensorAccessPattern[1], 2, 1, 1, 0, 0);

  return copy_elf(
      instance,
      &_binary_flashqla_recurrent_prefill_f32_gaudi2_o_start,
      &_binary_flashqla_recurrent_prefill_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_inverse16(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 1) {
    params->inputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 2) {
    params->outputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& lower = params->inputTensors[0].geometry;
  auto& inverse = params->outputTensors[0].geometry;
  const bool valid_shape =
      lower.dims == 5 && lower.maxSizes[0] == 16 &&
      lower.maxSizes[1] == 16 &&
      inverse.dims == 5 && inverse.maxSizes[0] == 16 &&
      inverse.maxSizes[1] == 16 &&
      inverse.maxSizes[2] == lower.maxSizes[2] &&
      inverse.maxSizes[3] == lower.maxSizes[3] &&
      inverse.maxSizes[4] == lower.maxSizes[4];
  if (!valid_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  if (lower.dataType != tpc_lib_api::DATA_F32 ||
      inverse.dataType != tpc_lib_api::DATA_F32) {
    lower.dataType = tpc_lib_api::DATA_F32;
    inverse.dataType = tpc_lib_api::DATA_F32;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 3;
  instance->indexSpaceGeometry[0] = lower.maxSizes[2];
  instance->indexSpaceGeometry[1] = lower.maxSizes[3];
  instance->indexSpaceGeometry[2] = lower.maxSizes[4];
  for (auto* pattern : {
           &instance->inputTensorAccessPattern[0],
           &instance->outputTensorAccessPattern[0]}) {
    set_mapping(*pattern, 0, 0, 0, 0, 15);
    set_mapping(*pattern, 1, 0, 0, 0, 15);
    set_mapping(*pattern, 2, 0, 1, 0, 0);
    set_mapping(*pattern, 3, 1, 1, 0, 0);
    set_mapping(*pattern, 4, 2, 1, 0, 0);
  }

  return copy_elf(
      instance,
      &_binary_flashqla_unit_lower_inverse16_f32_gaudi2_o_start,
      &_binary_flashqla_unit_lower_inverse16_f32_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_kkt_form(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 3) {
    params->inputTensorNr = 3;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 1) {
    params->outputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& dot = params->inputTensors[0].geometry;
  auto& gate = params->inputTensors[1].geometry;
  auto& beta = params->inputTensors[2].geometry;
  auto& kkt = params->outputTensors[0].geometry;
  const uint64_t matrices = dot.maxSizes[2];
  const bool matrix_shape =
      dot.dims == 3 && dot.maxSizes[0] == 64 &&
      dot.maxSizes[1] == 64 &&
      kkt.dims == 3 && kkt.maxSizes[0] == 64 &&
      kkt.maxSizes[1] == 64 && kkt.maxSizes[2] == matrices;
  const bool vector_shape =
      gate.dims == 2 && gate.maxSizes[0] == 64 &&
      gate.maxSizes[1] == matrices &&
      beta.dims == 2 && beta.maxSizes[0] == 64 &&
      beta.maxSizes[1] == matrices;
  if (!matrix_shape || !vector_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }

  if (dot.dataType != tpc_lib_api::DATA_BF16 ||
      gate.dataType != tpc_lib_api::DATA_BF16 ||
      beta.dataType != tpc_lib_api::DATA_BF16 ||
      kkt.dataType != tpc_lib_api::DATA_BF16) {
    dot.dataType = tpc_lib_api::DATA_BF16;
    gate.dataType = tpc_lib_api::DATA_BF16;
    beta.dataType = tpc_lib_api::DATA_BF16;
    kkt.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 2;
  instance->indexSpaceGeometry[0] = 64;
  instance->indexSpaceGeometry[1] = matrices;

  for (auto* pattern : {
           &instance->inputTensorAccessPattern[0],
           &instance->outputTensorAccessPattern[0]}) {
    set_mapping(*pattern, 0, 0, 0, 0, 63);
    set_mapping(*pattern, 1, 0, 1, 0, 0);
    set_mapping(*pattern, 2, 1, 1, 0, 0);
  }
  for (unsigned input : {1U, 2U}) {
    set_mapping(
        instance->inputTensorAccessPattern[input], 0, 0, 0, 0, 63);
    set_mapping(
        instance->inputTensorAccessPattern[input], 1, 1, 1, 0, 0);
  }

  return copy_elf(
      instance,
      &_binary_flashqla_kkt_form_bf16_gaudi2_o_start,
      &_binary_flashqla_kkt_form_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_compact_kkt(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 2) {
    params->inputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 1) {
    params->outputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& dot = params->inputTensors[0].geometry;
  auto& beta = params->inputTensors[1].geometry;
  auto& lmat = params->outputTensors[0].geometry;
  const uint64_t heads = dot.maxSizes[2];
  const uint64_t outer = dot.maxSizes[3];
  const uint64_t repeats = beta.maxSizes[1];
  const bool dot_shape =
      dot.dims == 4 && dot.maxSizes[0] == 64 &&
      dot.maxSizes[1] == 64;
  const bool beta_shape =
      beta.dims == 4 && beta.maxSizes[0] == 64 &&
      beta.maxSizes[2] == heads && beta.maxSizes[3] == outer;
  const bool output_shape =
      lmat.dims == 5 && lmat.maxSizes[0] == 64 &&
      lmat.maxSizes[1] == 64 && lmat.maxSizes[2] == repeats &&
      lmat.maxSizes[3] == heads && lmat.maxSizes[4] == outer;
  if (!dot_shape || !beta_shape || !output_shape || repeats == 0) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }

  for (unsigned input = 0; input < 2; ++input) {
    if (params->inputTensors[input].geometry.dataType !=
        tpc_lib_api::DATA_BF16) {
      params->inputTensors[input].geometry.dataType =
          tpc_lib_api::DATA_BF16;
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
  }
  if (lmat.dataType != tpc_lib_api::DATA_BF16) {
    lmat.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 4;
  instance->indexSpaceGeometry[0] = 64;
  instance->indexSpaceGeometry[1] = repeats;
  instance->indexSpaceGeometry[2] = heads;
  instance->indexSpaceGeometry[3] = outer;

  set_mapping(
      instance->inputTensorAccessPattern[0], 0, 0, 0, 0, 63);
  set_mapping(
      instance->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[0], 2, 2, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[0], 3, 3, 1, 0, 0);

  set_mapping(
      instance->inputTensorAccessPattern[1], 0, 0, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[1], 1, 1, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[1], 2, 2, 1, 0, 0);
  set_mapping(
      instance->inputTensorAccessPattern[1], 3, 3, 1, 0, 0);

  set_mapping(
      instance->outputTensorAccessPattern[0], 0, 0, 0, 0, 63);
  set_mapping(
      instance->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
  set_mapping(
      instance->outputTensorAccessPattern[0], 2, 1, 1, 0, 0);
  set_mapping(
      instance->outputTensorAccessPattern[0], 3, 2, 1, 0, 0);
  set_mapping(
      instance->outputTensorAccessPattern[0], 4, 3, 1, 0, 0);

  return copy_elf(
      instance,
      &_binary_qwen38_compact_kkt_bf16_gaudi2_o_start,
      &_binary_qwen38_compact_kkt_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_post_conv_qkv(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 1) {
    params->inputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 2) {
    params->outputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& packed = params->inputTensors[0].geometry;
  auto& q = params->outputTensors[0].geometry;
  auto& k = params->outputTensors[1].geometry;
  const uint64_t tokens = packed.maxSizes[1];
  const bool packed_shape =
      packed.dims == 2 && packed.maxSizes[0] == 10240;
  const bool qk_shape =
      q.dims == 3 && q.maxSizes[0] == 128 &&
      q.maxSizes[1] == 48 && q.maxSizes[2] == tokens &&
      k.dims == 3 && k.maxSizes[0] == 128 &&
      k.maxSizes[1] == 48 && k.maxSizes[2] == tokens;
  if (!packed_shape || !qk_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  if (packed.dataType != tpc_lib_api::DATA_BF16 ||
      q.dataType != tpc_lib_api::DATA_F32 ||
      k.dataType != tpc_lib_api::DATA_F32) {
    packed.dataType = tpc_lib_api::DATA_BF16;
    q.dataType = tpc_lib_api::DATA_F32;
    k.dataType = tpc_lib_api::DATA_F32;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 1;
  instance->indexSpaceGeometry[0] = tokens;
  set_mapping(
      instance->inputTensorAccessPattern[0], 0, 0, 0, 0, 4095);
  set_mapping(
      instance->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);

  for (unsigned output : {0U, 1U}) {
    set_mapping(
        instance->outputTensorAccessPattern[output], 0, 0, 0, 0, 127);
    set_mapping(
        instance->outputTensorAccessPattern[output], 1, 0, 0, 0, 47);
    set_mapping(
        instance->outputTensorAccessPattern[output], 2, 0, 1, 0, 0);
  }
  return copy_elf(
      instance,
      &_binary_qwen38_post_conv_qk_bf16_gaudi2_o_start,
      &_binary_qwen38_post_conv_qk_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_compact_post_conv_qk(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 1) {
    params->inputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 2) {
    params->outputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& packed = params->inputTensors[0].geometry;
  auto& q = params->outputTensors[0].geometry;
  auto& k = params->outputTensors[1].geometry;
  const uint64_t tokens = packed.maxSizes[1];
  const bool packed_shape =
      packed.dims == 2 && packed.maxSizes[0] == 10240;
  const bool qk_shape =
      q.dims == 3 && q.maxSizes[0] == 128 &&
      q.maxSizes[1] == 16 && q.maxSizes[2] == tokens &&
      k.dims == 3 && k.maxSizes[0] == 128 &&
      k.maxSizes[1] == 16 && k.maxSizes[2] == tokens;
  if (!packed_shape || !qk_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  if (packed.dataType != tpc_lib_api::DATA_BF16 ||
      q.dataType != tpc_lib_api::DATA_BF16 ||
      k.dataType != tpc_lib_api::DATA_BF16) {
    packed.dataType = tpc_lib_api::DATA_BF16;
    q.dataType = tpc_lib_api::DATA_BF16;
    k.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 1;
  instance->indexSpaceGeometry[0] = tokens;
  set_mapping(
      instance->inputTensorAccessPattern[0], 0, 0, 0, 0, 4095);
  set_mapping(
      instance->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
  for (unsigned output : {0U, 1U}) {
    set_mapping(
        instance->outputTensorAccessPattern[output], 0, 0, 0, 0, 127);
    set_mapping(
        instance->outputTensorAccessPattern[output], 1, 0, 0, 0, 15);
    set_mapping(
        instance->outputTensorAccessPattern[output], 2, 0, 1, 0, 0);
  }
  return copy_elf(
      instance,
      &_binary_qwen38_post_conv_qk_compact_bf16_gaudi2_o_start,
      &_binary_qwen38_post_conv_qk_compact_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_bf16_post_conv_qk(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 1) {
    params->inputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 2) {
    params->outputTensorNr = 2;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& packed = params->inputTensors[0].geometry;
  auto& q = params->outputTensors[0].geometry;
  auto& k = params->outputTensors[1].geometry;
  const uint64_t tokens = packed.maxSizes[1];
  const bool packed_shape =
      packed.dims == 2 && packed.maxSizes[0] == 10240;
  const bool qk_shape =
      q.dims == 3 && q.maxSizes[0] == 128 &&
      q.maxSizes[1] == 48 && q.maxSizes[2] == tokens &&
      k.dims == 3 && k.maxSizes[0] == 128 &&
      k.maxSizes[1] == 48 && k.maxSizes[2] == tokens;
  if (!packed_shape || !qk_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  if (packed.dataType != tpc_lib_api::DATA_BF16 ||
      q.dataType != tpc_lib_api::DATA_BF16 ||
      k.dataType != tpc_lib_api::DATA_BF16) {
    packed.dataType = tpc_lib_api::DATA_BF16;
    q.dataType = tpc_lib_api::DATA_BF16;
    k.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 1;
  instance->indexSpaceGeometry[0] = tokens;
  set_mapping(
      instance->inputTensorAccessPattern[0], 0, 0, 0, 0, 4095);
  set_mapping(
      instance->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
  for (unsigned output : {0U, 1U}) {
    set_mapping(
        instance->outputTensorAccessPattern[output], 0, 0, 0, 0, 127);
    set_mapping(
        instance->outputTensorAccessPattern[output], 1, 0, 0, 0, 47);
    set_mapping(
        instance->outputTensorAccessPattern[output], 2, 0, 1, 0, 0);
  }
  return copy_elf(
      instance,
      &_binary_qwen38_post_conv_qk_expanded_bf16_gaudi2_o_start,
      &_binary_qwen38_post_conv_qk_expanded_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_conv_qkv_prep(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 4) {
    params->inputTensorNr = 4;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 3) {
    params->outputTensorNr = 3;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& packed = params->inputTensors[0].geometry;
  auto& state = params->inputTensors[1].geometry;
  auto& weight = params->inputTensors[2].geometry;
  auto& bias = params->inputTensors[3].geometry;
  auto& q = params->outputTensors[0].geometry;
  auto& k = params->outputTensors[1].geometry;
  auto& v = params->outputTensors[2].geometry;
  const uint64_t tokens = packed.maxSizes[1];
  const bool input_shapes =
      packed.dims == 2 && packed.maxSizes[0] == 10240 &&
      state.dims == 2 && state.maxSizes[0] == 10240 &&
      state.maxSizes[1] == 3 &&
      weight.dims == 2 && weight.maxSizes[0] == 10240 &&
      weight.maxSizes[1] == 4 &&
      bias.dims == 1 && bias.maxSizes[0] == 10240;
  const bool qk_shapes =
      q.dims == 3 && q.maxSizes[0] == 128 &&
      q.maxSizes[1] == 48 && q.maxSizes[2] == tokens &&
      k.dims == 3 && k.maxSizes[0] == 128 &&
      k.maxSizes[1] == 48 && k.maxSizes[2] == tokens;
  const bool value_shape =
      v.dims == 3 && v.maxSizes[0] == 128 &&
      v.maxSizes[1] == 48 && v.maxSizes[2] == tokens;
  if (!input_shapes || !qk_shapes || !value_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  for (unsigned input = 0; input < 4; ++input) {
    if (params->inputTensors[input].geometry.dataType !=
        tpc_lib_api::DATA_BF16) {
      params->inputTensors[input].geometry.dataType =
          tpc_lib_api::DATA_BF16;
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
  }
  if (q.dataType != tpc_lib_api::DATA_BF16 ||
      k.dataType != tpc_lib_api::DATA_BF16 ||
      v.dataType != tpc_lib_api::DATA_BF16) {
    q.dataType = tpc_lib_api::DATA_BF16;
    k.dataType = tpc_lib_api::DATA_BF16;
    v.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 2;
  instance->indexSpaceGeometry[0] = tokens;
  instance->indexSpaceGeometry[1] = 48;
  set_mapping(
      instance->inputTensorAccessPattern[0], 0, 1, 128, 0, 4223);
  set_mapping(
      instance->inputTensorAccessPattern[0], 1, 0, 1, -3, 0);
  set_mapping(
      instance->inputTensorAccessPattern[1], 0, 1, 128, 0, 4223);
  set_mapping(
      instance->inputTensorAccessPattern[1], 1, 0, 0, 0, 2);
  set_mapping(
      instance->inputTensorAccessPattern[2], 0, 1, 128, 0, 4223);
  set_mapping(
      instance->inputTensorAccessPattern[2], 1, 0, 0, 0, 3);
  set_mapping(
      instance->inputTensorAccessPattern[3], 0, 1, 128, 0, 4223);

  for (unsigned output : {0U, 1U}) {
    set_mapping(
        instance->outputTensorAccessPattern[output], 0, 0, 0, 0, 127);
    set_mapping(
        instance->outputTensorAccessPattern[output], 1, 1, 0, 0, 47);
    set_mapping(
        instance->outputTensorAccessPattern[output], 2, 0, 1, 0, 0);
  }
  set_mapping(
      instance->outputTensorAccessPattern[2], 0, 0, 0, 0, 127);
  set_mapping(
      instance->outputTensorAccessPattern[2], 1, 1, 1, 0, 0);
  set_mapping(
      instance->outputTensorAccessPattern[2], 2, 0, 1, 0, 0);
  return copy_elf(
      instance,
      &_binary_qwen38_conv_qkv_prep_bf16_gaudi2_o_start,
      &_binary_qwen38_conv_qkv_prep_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn instantiate_rmsnorm_gated(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (params->deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  if (params->inputTensorNr != 3) {
    params->inputTensorNr = 3;
    return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
  }
  if (params->outputTensorNr != 1) {
    params->outputTensorNr = 1;
    return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
  }

  auto& x = params->inputTensors[0].geometry;
  auto& z = params->inputTensors[1].geometry;
  auto& weight = params->inputTensors[2].geometry;
  auto& output = params->outputTensors[0].geometry;
  const uint64_t rows = x.maxSizes[1];
  const bool matrix_shapes =
      x.dims == 2 && x.maxSizes[0] == 128 &&
      z.dims == 2 && z.maxSizes[0] == 128 &&
      z.maxSizes[1] == rows &&
      output.dims == 2 && output.maxSizes[0] == 128 &&
      output.maxSizes[1] == rows;
  const bool weight_shape =
      weight.dims == 1 && weight.maxSizes[0] == 128;
  if (!matrix_shapes || !weight_shape) {
    return tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
  }
  for (unsigned input = 0; input < 3; ++input) {
    if (params->inputTensors[input].geometry.dataType !=
        tpc_lib_api::DATA_BF16) {
      params->inputTensors[input].geometry.dataType =
          tpc_lib_api::DATA_BF16;
      return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
    }
  }
  if (output.dataType != tpc_lib_api::DATA_BF16) {
    output.dataType = tpc_lib_api::DATA_BF16;
    return tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE;
  }

  instance->indexSpaceRank = 1;
  instance->indexSpaceGeometry[0] = rows;
  for (unsigned input : {0U, 1U}) {
    set_mapping(
        instance->inputTensorAccessPattern[input], 0, 0, 0, 0, 127);
    set_mapping(
        instance->inputTensorAccessPattern[input], 1, 0, 1, 0, 0);
  }
  set_mapping(
      instance->inputTensorAccessPattern[2], 0, 0, 0, 0, 127);
  set_mapping(
      instance->outputTensorAccessPattern[0], 0, 0, 0, 0, 127);
  set_mapping(
      instance->outputTensorAccessPattern[0], 1, 0, 1, 0, 0);
  return copy_elf(
      instance,
      &_binary_qwen38_rmsnorm_gated_bf16_gaudi2_o_start,
      &_binary_qwen38_rmsnorm_gated_bf16_gaudi2_o_end);
}

}  // namespace

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId device_id,
    uint32_t* kernel_count,
    tpc_lib_api::GuidInfo* guids) {
  const uint32_t count =
      device_id == tpc_lib_api::DEVICE_ID_GAUDI2 ? 10 : 0;
  if (kernel_count != nullptr) {
    *kernel_count = count;
  }
  if (count == 10 && guids != nullptr) {
    std::strcpy(guids[0].name, kKernelName);
    std::strcpy(guids[1].name, kRecurrentKernelName);
    std::strcpy(guids[2].name, kInverse16KernelName);
    std::strcpy(guids[3].name, kKktFormKernelName);
    std::strcpy(guids[4].name, kPostConvKernelName);
    std::strcpy(guids[5].name, kConvQkvKernelName);
    std::strcpy(guids[6].name, kCompactPostConvKernelName);
    std::strcpy(guids[7].name, kBf16PostConvKernelName);
    std::strcpy(guids[8].name, kRmsNormGatedKernelName);
    std::strcpy(guids[9].name, kCompactKktKernelName);
  }
  return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams* params,
    tpc_lib_api::HabanaKernelInstantiation* instance) {
  if (std::strcmp(params->guid.name, kKernelName) != 0) {
    if (std::strcmp(params->guid.name, kRecurrentKernelName) == 0) {
      return instantiate_recurrent_prefill(params, instance);
    }
    if (std::strcmp(params->guid.name, kInverse16KernelName) == 0) {
      return instantiate_inverse16(params, instance);
    }
    if (std::strcmp(params->guid.name, kKktFormKernelName) == 0) {
      return instantiate_kkt_form(params, instance);
    }
    if (std::strcmp(params->guid.name, kCompactKktKernelName) == 0) {
      return instantiate_compact_kkt(params, instance);
    }
    if (std::strcmp(params->guid.name, kPostConvKernelName) == 0) {
      return instantiate_post_conv_qkv(params, instance);
    }
    if (std::strcmp(params->guid.name, kCompactPostConvKernelName) == 0) {
      return instantiate_compact_post_conv_qk(params, instance);
    }
    if (std::strcmp(params->guid.name, kBf16PostConvKernelName) == 0) {
      return instantiate_bf16_post_conv_qk(params, instance);
    }
    if (std::strcmp(params->guid.name, kConvQkvKernelName) == 0) {
      return instantiate_conv_qkv_prep(params, instance);
    }
    if (std::strcmp(params->guid.name, kRmsNormGatedKernelName) == 0) {
      return instantiate_rmsnorm_gated(params, instance);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
  }
  return instantiate_pair_transform(params, instance);
}

tpc_lib_api::GlueCodeReturn GetShapeInference(
    tpc_lib_api::DeviceId,
    tpc_lib_api::ShapeInferenceParams*,
    tpc_lib_api::ShapeInferenceOutput*) {
  return tpc_lib_api::GLUE_SUCCESS;
}

}  // extern "C"
