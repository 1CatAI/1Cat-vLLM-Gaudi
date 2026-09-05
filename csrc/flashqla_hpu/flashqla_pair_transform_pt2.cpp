#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "hpu_custom_op_pt2.h"

namespace {

constexpr const char* kSchema =
    "custom_op::flashqla_pair_transform_bf16_gaudi2";
constexpr const char* kRecurrentSchema =
    "custom_op::flashqla_recurrent_prefill_f32_gaudi2";
constexpr const char* kInverse16Schema =
    "custom_op::flashqla_unit_lower_inverse16_f32_gaudi2";
constexpr const char* kKktFormSchema =
    "custom_op::flashqla_kkt_form_bf16_gaudi2";
constexpr const char* kCompactKktSchema =
    "custom_op::qwen38_compact_kkt_bf16_gaudi2";
constexpr const char* kPostConvSchema =
    "custom_op::qwen38_post_conv_qk_bf16_gaudi2";
constexpr const char* kCompactPostConvSchema =
    "custom_op::qwen38_post_conv_qk_compact_bf16_gaudi2";
constexpr const char* kBf16PostConvSchema =
    "custom_op::qwen38_post_conv_qk_expanded_bf16_gaudi2";
constexpr const char* kConvQkvSchema =
    "custom_op::qwen38_conv_qkv_prep_bf16_gaudi2";
constexpr const char* kRmsNormGatedSchema =
    "custom_op::qwen38_rmsnorm_gated_bf16_gaudi2";

int64_t compact_post_conv_heads(const at::Tensor& packed) {
  TORCH_CHECK(packed.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      packed.dim() == 2 &&
          (packed.size(1) == 10240 || packed.size(1) == 5120),
      "expected compact QKV shape [tokens, 10240] or [tokens, 5120], got ",
      packed.sizes());
  return packed.size(1) == 5120 ? 8 : 16;
}

bool register_flashqla_pair_transform() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& a0 = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = a0.sizes().vec();
    return habana::PartialOutputMetaDataVector{output, output};
  };
  habana::custom_op::registerUserCustomOp(
      kSchema,
      "flashqla_pair_transform_bf16_gaudi2",
      output_meta,
      nullptr);
  auto recurrent_output_meta = [](const at::Stack& inputs) {
    const auto& v = inputs.at(2).toTensor();
    const auto& initial_state = inputs.at(5).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::Float;
    output.shape = v.sizes().vec();
    habana::PartialOutputMetaData final_state;
    final_state.dtype = at::ScalarType::Float;
    final_state.shape = initial_state.sizes().vec();
    return habana::PartialOutputMetaDataVector{output, final_state};
  };
  habana::custom_op::registerUserCustomOp(
      kRecurrentSchema,
      "flashqla_recurrent_prefill_f32_gaudi2",
      recurrent_output_meta,
      nullptr);
  auto inverse16_output_meta = [](const at::Stack& inputs) {
    const auto& lower = inputs.at(0).toTensor();
    habana::PartialOutputMetaData inverse;
    inverse.dtype = at::ScalarType::Float;
    inverse.shape = lower.sizes().vec();
    return habana::PartialOutputMetaDataVector{inverse};
  };
  habana::custom_op::registerUserCustomOp(
      kInverse16Schema,
      "flashqla_unit_lower_inverse16_f32_gaudi2",
      inverse16_output_meta,
      nullptr);
  auto kkt_form_output_meta = [](const at::Stack& inputs) {
    const auto& dot = inputs.at(0).toTensor();
    habana::PartialOutputMetaData kkt;
    kkt.dtype = at::ScalarType::BFloat16;
    kkt.shape = dot.sizes().vec();
    return habana::PartialOutputMetaDataVector{kkt};
  };
  habana::custom_op::registerUserCustomOp(
      kKktFormSchema,
      "flashqla_kkt_form_bf16_gaudi2",
      kkt_form_output_meta,
      nullptr);
  auto compact_kkt_output_meta = [](const at::Stack& inputs) {
    const auto& dot = inputs.at(0).toTensor();
    const auto& beta = inputs.at(1).toTensor();
    habana::PartialOutputMetaData lmat;
    lmat.dtype = at::ScalarType::BFloat16;
    lmat.shape = {
        dot.size(0), dot.size(1), beta.size(2), dot.size(2), dot.size(3)};
    return habana::PartialOutputMetaDataVector{lmat};
  };
  habana::custom_op::registerUserCustomOp(
      kCompactKktSchema,
      "qwen38_compact_kkt_bf16_gaudi2",
      compact_kkt_output_meta,
      nullptr);
  auto post_conv_output_meta = [](const at::Stack& inputs) {
    const auto& packed = inputs.at(0).toTensor();
    const auto tokens = packed.size(0);
    habana::PartialOutputMetaData q;
    q.dtype = at::ScalarType::Float;
    q.shape = {tokens, 48, 128};
    return habana::PartialOutputMetaDataVector{q, q};
  };
  habana::custom_op::registerUserCustomOp(
      kPostConvSchema,
      "qwen38_post_conv_qk_bf16_gaudi2",
      post_conv_output_meta,
      nullptr);
  auto compact_post_conv_output_meta = [](const at::Stack& inputs) {
    const auto& packed = inputs.at(0).toTensor();
    const auto tokens = packed.size(0);
    habana::PartialOutputMetaData q;
    q.dtype = at::ScalarType::BFloat16;
    q.shape = {tokens, compact_post_conv_heads(packed), 128};
    return habana::PartialOutputMetaDataVector{q, q};
  };
  habana::custom_op::registerUserCustomOp(
      kCompactPostConvSchema,
      "qwen38_post_conv_qk_compact_bf16_gaudi2",
      compact_post_conv_output_meta,
      nullptr);
  auto bf16_post_conv_output_meta = [](const at::Stack& inputs) {
    const auto& packed = inputs.at(0).toTensor();
    const auto tokens = packed.size(0);
    habana::PartialOutputMetaData q;
    q.dtype = at::ScalarType::BFloat16;
    q.shape = {tokens, 48, 128};
    return habana::PartialOutputMetaDataVector{q, q};
  };
  habana::custom_op::registerUserCustomOp(
      kBf16PostConvSchema,
      "qwen38_post_conv_qk_expanded_bf16_gaudi2",
      bf16_post_conv_output_meta,
      nullptr);
  auto conv_qkv_output_meta = [](const at::Stack& inputs) {
    const auto& packed = inputs.at(0).toTensor();
    const auto tokens = packed.size(0);
    habana::PartialOutputMetaData q;
    q.dtype = at::ScalarType::BFloat16;
    q.shape = {tokens, 48, 128};
    habana::PartialOutputMetaData v;
    v.dtype = at::ScalarType::BFloat16;
    v.shape = q.shape;
    return habana::PartialOutputMetaDataVector{q, q, v};
  };
  habana::custom_op::registerUserCustomOp(
      kConvQkvSchema,
      "qwen38_conv_qkv_prep_bf16_gaudi2",
      conv_qkv_output_meta,
      nullptr);
  auto rmsnorm_gated_output_meta = [](const at::Stack& inputs) {
    const auto& x = inputs.at(0).toTensor();
    habana::PartialOutputMetaData output;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = x.sizes().vec();
    return habana::PartialOutputMetaDataVector{output};
  };
  habana::custom_op::registerUserCustomOp(
      kRmsNormGatedSchema,
      "qwen38_rmsnorm_gated_bf16_gaudi2",
      rmsnorm_gated_output_meta,
      nullptr);
  return true;
}

const bool kRegistered = register_flashqla_pair_transform();

std::tuple<at::Tensor, at::Tensor> pair_transform_hpu(
    const at::Tensor& a0,
    const at::Tensor& scores,
    const at::Tensor& g_even,
    const at::Tensor& g_odd,
    const at::Tensor& beta) {
  TORCH_CHECK(a0.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(scores.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(g_even.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(g_odd.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(beta.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(a0.dim() == 4 && a0.size(2) == 64 && a0.size(3) == 64);
  TORCH_CHECK(scores.sizes() == a0.sizes());
  TORCH_CHECK(
      g_even.dim() == 3 && g_even.size(0) == a0.size(0) &&
      g_even.size(1) == a0.size(1) && g_even.size(2) == 32);
  TORCH_CHECK(g_odd.sizes() == g_even.sizes());
  TORCH_CHECK(
      beta.dim() == 3 && beta.size(0) == a0.size(0) &&
      beta.size(1) == a0.size(1) && beta.size(2) == 64);
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kSchema);
  std::vector<c10::IValue> inputs{a0, scores, g_even, g_odd, beta};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> pair_transform_meta(
    const at::Tensor& a0,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  return {at::empty_like(a0), at::empty_like(a0)};
}

std::tuple<at::Tensor, at::Tensor> recurrent_prefill_hpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& decay,
    const at::Tensor& beta,
    const at::Tensor& initial_state) {
  for (const auto& tensor : {q, k, v, decay, beta, initial_state}) {
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float);
  }
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 128);
  TORCH_CHECK(k.sizes() == q.sizes());
  TORCH_CHECK(v.sizes() == q.sizes());
  TORCH_CHECK(
      decay.dim() == 2 && decay.size(0) == q.size(0) &&
      decay.size(1) == q.size(1));
  TORCH_CHECK(beta.sizes() == decay.sizes());
  TORCH_CHECK(
      initial_state.dim() == 3 &&
      initial_state.size(0) == q.size(1) &&
      initial_state.size(1) == 128 && initial_state.size(2) == 128);
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kRecurrentSchema);
  std::vector<c10::IValue> inputs{
      q, k, v, decay, beta, initial_state};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> recurrent_prefill_meta(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& v,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor& initial_state) {
  return {at::empty_like(v), at::empty_like(initial_state)};
}

at::Tensor inverse16_hpu(const at::Tensor& lower) {
  TORCH_CHECK(lower.scalar_type() == at::ScalarType::Float);
  TORCH_CHECK(
      lower.dim() == 5 && lower.size(3) == 16 && lower.size(4) == 16);
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kInverse16Schema);
  std::vector<c10::IValue> inputs{lower};
  return descriptor.execute(inputs).at(0);
}

at::Tensor inverse16_meta(const at::Tensor& lower) {
  return at::empty_like(lower);
}

at::Tensor kkt_form_hpu(
    const at::Tensor& dot,
    const at::Tensor& gate,
    const at::Tensor& beta) {
  TORCH_CHECK(dot.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(gate.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(beta.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(dot.dim() == 3 && dot.size(1) == 64 && dot.size(2) == 64);
  TORCH_CHECK(
      gate.dim() == 2 && gate.size(0) == dot.size(0) && gate.size(1) == 64);
  TORCH_CHECK(beta.sizes() == gate.sizes());
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kKktFormSchema);
  std::vector<c10::IValue> inputs{dot, gate, beta};
  return descriptor.execute(inputs).at(0);
}

at::Tensor kkt_form_meta(
    const at::Tensor& dot,
    const at::Tensor&,
    const at::Tensor&) {
  return at::empty_like(dot);
}

at::Tensor compact_kkt_hpu(
    const at::Tensor& compact_dot,
    const at::Tensor& grouped_beta) {
  TORCH_CHECK(compact_dot.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(grouped_beta.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      compact_dot.dim() == 4 && compact_dot.size(2) == 64 &&
          compact_dot.size(3) == 64,
      "expected compact_dot shape [outer, heads, 64, 64], got ",
      compact_dot.sizes());
  TORCH_CHECK(
      grouped_beta.dim() == 4 &&
          grouped_beta.size(0) == compact_dot.size(0) &&
          grouped_beta.size(1) == compact_dot.size(1) &&
          grouped_beta.size(3) == 64,
      "expected grouped_beta shape [outer, heads, repeats, 64], got ",
      grouped_beta.sizes());
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kCompactKktSchema);
  std::vector<c10::IValue> inputs{compact_dot, grouped_beta};
  return descriptor.execute(inputs).at(0);
}

at::Tensor compact_kkt_meta(
    const at::Tensor& compact_dot,
    const at::Tensor& grouped_beta) {
  return at::empty(
      {
          compact_dot.size(0),
          compact_dot.size(1),
          grouped_beta.size(2),
          compact_dot.size(2),
          compact_dot.size(3),
      },
      compact_dot.options());
}

std::tuple<at::Tensor, at::Tensor> post_conv_qk_hpu(
    const at::Tensor& packed) {
  TORCH_CHECK(packed.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      packed.dim() == 2 && packed.size(1) == 10240,
      "expected packed QKV shape [tokens, 10240], got ",
      packed.sizes());
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kPostConvSchema);
  std::vector<c10::IValue> inputs{packed};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> post_conv_qk_meta(
    const at::Tensor& packed) {
  const auto options = packed.options().dtype(at::ScalarType::Float);
  const auto tokens = packed.size(0);
  return {
      at::empty({tokens, 48, 128}, options),
      at::empty({tokens, 48, 128}, options),
  };
}

std::tuple<at::Tensor, at::Tensor> compact_post_conv_qk_hpu(
    const at::Tensor& packed) {
  compact_post_conv_heads(packed);
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kCompactPostConvSchema);
  std::vector<c10::IValue> inputs{packed};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> compact_post_conv_qk_meta(
    const at::Tensor& packed) {
  const auto heads = compact_post_conv_heads(packed);
  const auto options = packed.options().dtype(at::ScalarType::BFloat16);
  const auto tokens = packed.size(0);
  return {
      at::empty({tokens, heads, 128}, options),
      at::empty({tokens, heads, 128}, options),
  };
}

std::tuple<at::Tensor, at::Tensor> bf16_post_conv_qk_hpu(
    const at::Tensor& packed) {
  TORCH_CHECK(packed.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(
      packed.dim() == 2 && packed.size(1) == 10240,
      "expected packed QKV shape [tokens, 10240], got ",
      packed.sizes());
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kBf16PostConvSchema);
  std::vector<c10::IValue> inputs{packed};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> bf16_post_conv_qk_meta(
    const at::Tensor& packed) {
  const auto options = packed.options().dtype(at::ScalarType::BFloat16);
  const auto tokens = packed.size(0);
  return {
      at::empty({tokens, 48, 128}, options),
      at::empty({tokens, 48, 128}, options),
  };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_qkv_hpu(
    const at::Tensor& packed,
    const at::Tensor& initial_state,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  for (const auto& tensor : {packed, initial_state, weight, bias}) {
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16);
  }
  TORCH_CHECK(
      packed.dim() == 2 && packed.size(1) == 10240,
      "expected packed QKV shape [tokens, 10240], got ",
      packed.sizes());
  TORCH_CHECK(
      initial_state.dim() == 2 && initial_state.size(0) == 3 &&
          initial_state.size(1) == 10240,
      "expected initial state shape [3, 10240], got ",
      initial_state.sizes());
  TORCH_CHECK(
      weight.dim() == 2 && weight.size(0) == 4 &&
          weight.size(1) == 10240,
      "expected transposed convolution weight shape [4, 10240], got ",
      weight.sizes());
  TORCH_CHECK(
      bias.dim() == 1 && bias.size(0) == 10240,
      "expected convolution bias shape [10240], got ",
      bias.sizes());
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kConvQkvSchema);
  std::vector<c10::IValue> inputs{packed, initial_state, weight, bias};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1), outputs.at(2)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_qkv_meta(
    const at::Tensor& packed,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&) {
  const auto tokens = packed.size(0);
  const auto output_shape = std::vector<int64_t>{tokens, 48, 128};
  return {
      at::empty(output_shape, packed.options()),
      at::empty(output_shape, packed.options()),
      at::empty(output_shape, packed.options()),
  };
}

at::Tensor rmsnorm_gated_hpu(
    const at::Tensor& x,
    const at::Tensor& z,
    const at::Tensor& weight) {
  for (const auto& tensor : {x, z, weight}) {
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16);
  }
  TORCH_CHECK(
      x.dim() == 2 && x.size(1) == 128,
      "expected x shape [rows, 128], got ",
      x.sizes());
  TORCH_CHECK(z.sizes() == x.sizes());
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 128);
  TORCH_CHECK(kRegistered);

  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kRmsNormGatedSchema);
  std::vector<c10::IValue> inputs{x, z, weight};
  return descriptor.execute(inputs).at(0);
}

at::Tensor rmsnorm_gated_meta(
    const at::Tensor& x,
    const at::Tensor&,
    const at::Tensor&) {
  return at::empty_like(x);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def(
      "flashqla_pair_transform_bf16_gaudi2("
      "Tensor a0, Tensor scores, Tensor g_even, Tensor g_odd, Tensor beta) "
      "-> (Tensor, Tensor)");
  m.def(
      "flashqla_recurrent_prefill_f32_gaudi2("
      "Tensor q, Tensor k, Tensor v, Tensor decay, Tensor beta, "
      "Tensor initial_state) -> (Tensor, Tensor)");
  m.def(
      "flashqla_unit_lower_inverse16_f32_gaudi2("
      "Tensor lower) -> Tensor");
  m.def(
      "flashqla_kkt_form_bf16_gaudi2("
      "Tensor dot, Tensor gate, Tensor beta) -> Tensor");
  m.def(
      "qwen38_compact_kkt_bf16_gaudi2("
      "Tensor compact_dot, Tensor grouped_beta) -> Tensor");
  m.def(
      "qwen38_post_conv_qk_bf16_gaudi2("
      "Tensor packed) -> (Tensor, Tensor)");
  m.def(
      "qwen38_post_conv_qk_compact_bf16_gaudi2("
      "Tensor packed) -> (Tensor, Tensor)");
  m.def(
      "qwen38_post_conv_qk_expanded_bf16_gaudi2("
      "Tensor packed) -> (Tensor, Tensor)");
  m.def(
      "qwen38_conv_qkv_prep_bf16_gaudi2("
      "Tensor packed, Tensor initial_state, Tensor weight, Tensor bias) "
      "-> (Tensor, Tensor, Tensor)");
  m.def(
      "qwen38_rmsnorm_gated_bf16_gaudi2("
      "Tensor x, Tensor z, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl(
      "flashqla_pair_transform_bf16_gaudi2",
      pair_transform_hpu);
  m.impl(
      "flashqla_recurrent_prefill_f32_gaudi2",
      recurrent_prefill_hpu);
  m.impl(
      "flashqla_unit_lower_inverse16_f32_gaudi2",
      inverse16_hpu);
  m.impl("flashqla_kkt_form_bf16_gaudi2", kkt_form_hpu);
  m.impl("qwen38_compact_kkt_bf16_gaudi2", compact_kkt_hpu);
  m.impl("qwen38_post_conv_qk_bf16_gaudi2", post_conv_qk_hpu);
  m.impl(
      "qwen38_post_conv_qk_compact_bf16_gaudi2",
      compact_post_conv_qk_hpu);
  m.impl(
      "qwen38_post_conv_qk_expanded_bf16_gaudi2",
      bf16_post_conv_qk_hpu);
  m.impl("qwen38_conv_qkv_prep_bf16_gaudi2", conv_qkv_hpu);
  m.impl("qwen38_rmsnorm_gated_bf16_gaudi2", rmsnorm_gated_hpu);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl(
      "flashqla_pair_transform_bf16_gaudi2",
      pair_transform_meta);
  m.impl(
      "flashqla_recurrent_prefill_f32_gaudi2",
      recurrent_prefill_meta);
  m.impl(
      "flashqla_unit_lower_inverse16_f32_gaudi2",
      inverse16_meta);
  m.impl("flashqla_kkt_form_bf16_gaudi2", kkt_form_meta);
  m.impl("qwen38_compact_kkt_bf16_gaudi2", compact_kkt_meta);
  m.impl("qwen38_post_conv_qk_bf16_gaudi2", post_conv_qk_meta);
  m.impl(
      "qwen38_post_conv_qk_compact_bf16_gaudi2",
      compact_post_conv_qk_meta);
  m.impl(
      "qwen38_post_conv_qk_expanded_bf16_gaudi2",
      bf16_post_conv_qk_meta);
  m.impl("qwen38_conv_qkv_prep_bf16_gaudi2", conv_qkv_meta);
  m.impl("qwen38_rmsnorm_gated_bf16_gaudi2", rmsnorm_gated_meta);
}
