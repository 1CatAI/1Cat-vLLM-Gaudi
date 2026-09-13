// SPDX-License-Identifier: Apache-2.0
// Functional outputs for the existing SWA/global sparse TPC programs.
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr const char* kSparseSchema = "custom_op::custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2";
constexpr const char* kSparseGuid = "custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2";
constexpr const char* kSwaSchema = "custom_op::custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2";
constexpr const char* kSwaGuid = "custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2";

bool register_attention() {
  auto output_meta = [](const at::Stack& inputs) {
    const auto& q = inputs.at(0).toTensor();
    const auto& sink = inputs.back().toTensor();
    habana::PartialOutputMetaData output, stats;
    output.dtype = at::ScalarType::BFloat16;
    output.shape = {q.size(0), sink.numel(), q.size(2)};
    stats.dtype = at::ScalarType::Float;
    stats.shape = {q.size(0), q.size(1)};
    return habana::PartialOutputMetaDataVector{output, stats};
  };
  habana::custom_op::registerUserCustomOp(kSparseSchema, kSparseGuid, output_meta, nullptr);
  habana::custom_op::registerUserCustomOp(kSwaSchema, kSwaGuid, output_meta, nullptr);
  return true;
}
const bool kRegistered = register_attention();

template<bool Swa, bool Meta>
void attention(const c10::OperatorHandle&, torch::jit::Stack* stack) {
  const auto& inputs = *stack;
  TORCH_CHECK(inputs.size() == (Swa ? 13 : 10));
  const auto q = inputs.front().toTensor();
  const auto sink = inputs.back().toTensor();
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16 && q.dim() == 3 &&
              q.size(0) == 1 && q.size(1) == 32 && q.size(2) == 512,
              "Native V4 attention requires a BF16 [1,32,512] query");
  TORCH_CHECK(sink.scalar_type() == at::ScalarType::Float && sink.dim() == 1 && sink.numel() == 64);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto value = inputs[i].toTensor();
    const auto dtype = i == 0 ? at::ScalarType::BFloat16 :
        i == inputs.size() - 1 ? at::ScalarType::Float :
        i == 1 || i == (Swa ? 8 : 5) ? at::ScalarType::Byte : at::ScalarType::Int;
    TORCH_CHECK(value.scalar_type() == dtype && value.is_contiguous());
    TORCH_CHECK(value.device() == q.device());
  }
  std::vector<at::Tensor> outputs;
  if constexpr (Meta) {
    outputs = {at::empty({1, 64, 512}, q.options()),
               at::empty({1, 32}, q.options().dtype(at::ScalarType::Float))};
  } else {
    TORCH_CHECK(q.device().type() == c10::DeviceType::HPU && kRegistered);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
        Swa ? kSwaSchema : kSparseSchema);
    outputs = descriptor.execute(inputs);
  }
  TORCH_CHECK(outputs.size() == 2);
  stack->clear();
  stack->emplace_back(outputs[0]);
  stack->emplace_back(outputs[1]);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2("
        "Tensor q, Tensor compressed_storage_u8, Tensor compressed_geometry, "
        "Tensor topk_indices, Tensor topk_lens, Tensor swa_storage_u8, Tensor swa_geometry, "
        "Tensor swa_indices, Tensor swa_lens, Tensor attn_sink) -> (Tensor, Tensor)");
  m.def("custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2("
        "Tensor q, Tensor dummy_storage_u8, Tensor dummy_geometry, Tensor dummy_topk_shape, "
        "Tensor dummy_token_to_req, Tensor dummy_block_table, Tensor dummy_valid_token, Tensor dummy_seq_lens, "
        "Tensor swa_storage_u8, Tensor swa_geometry, Tensor swa_indices, Tensor swa_lens, "
        "Tensor attn_sink) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2",
         torch::CppFunction::makeFromBoxedFunction<&attention<false, false>>());
  m.impl("custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2",
         torch::CppFunction::makeFromBoxedFunction<&attention<true, false>>());
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("custom_deepseek_v4_native_paged_sparse_attn_fp8_gaudi2",
         torch::CppFunction::makeFromBoxedFunction<&attention<false, true>>());
  m.impl("custom_deepseek_v4_native_paged_swa_attn_fp8_gaudi2",
         torch::CppFunction::makeFromBoxedFunction<&attention<true, true>>());
}
