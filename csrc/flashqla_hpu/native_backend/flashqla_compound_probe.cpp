#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "hpu_ops/op_backend.h"

namespace {

constexpr const char* kSchema = "custom_op::flashqla_compound_probe";
constexpr const char* kBundledSchema =
    "custom_op::flashqla_compound_probe_bundled";
constexpr const char* kScheduledSchema =
    "custom_op::flashqla_compound_probe_scheduled";
constexpr const char* kBf16BmmF32Schema =
    "custom_op::flashqla_bf16_bmm_f32";
constexpr const char* kRecurrentScanSchema =
    "custom_op::flashqla_recurrent_scan_probe";
constexpr const char* kRecurrentScanBundled16Schema =
    "custom_op::flashqla_recurrent_scan_probe_bundled16";

enum class HintMode {
  kNone,
  kBundle,
  kSchedule,
};

habana::PartialOutputMetaDataVector probe_output_meta(
    const at::Stack& stack) {
  const auto& lhs = stack.at(0).toTensor();
  const auto& rhs = stack.at(1).toTensor();
  std::vector<int64_t> shape{lhs.size(0), lhs.size(1), rhs.size(2)};
  return {{lhs.scalar_type(), std::move(shape)}};
}

habana::PartialOutputMetaDataVector bf16_bmm_f32_output_meta(
    const at::Stack& stack) {
  const auto& lhs = stack.at(0).toTensor();
  const auto& rhs = stack.at(1).toTensor();
  std::vector<int64_t> shape{lhs.size(0), lhs.size(1), rhs.size(2)};
  return {{at::ScalarType::Float, std::move(shape)}};
}

habana::PartialOutputMetaDataVector recurrent_scan_output_meta(
    const at::Stack& stack) {
  const auto& core = stack.at(1).toTensor();
  const auto& initial_state = stack.at(3).toTensor();
  return {
      {core.scalar_type(), core.sizes().vec()},
      {initial_state.scalar_type(), initial_state.sizes().vec()},
  };
}

class FlashQlaCompoundProbe final : public habana::OpBackend {
 public:
  FlashQlaCompoundProbe(
      int device_id,
      c10::ScalarType scalar_type,
      HintMode hint_mode)
      : OpBackend(
            device_id,
            NO_TPC + std::string("flashqla_compound_probe"),
            scalar_type,
            {0},
            {},
            {},
            false),
        hint_mode_(hint_mode) {
    SetOutputMetaFn([](const at::Stack& stack) {
      const auto& lhs = stack.at(0).toTensor();
      const auto& rhs = stack.at(1).toTensor();
      return habana::OutputMetaDataVector{{
          lhs.scalar_type(),
          {lhs.size(0), lhs.size(1), rhs.size(2)},
      }};
    });
  }

  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto& lhs = stack.at(0).toTensor();
    const auto& rhs = stack.at(1).toTensor();
    const auto& bias = stack.at(2).toTensor();
    HABANA_ASSERT(lhs.dim() == 3 && rhs.dim() == 3 && bias.dim() == 3);
    HABANA_ASSERT(lhs.scalar_type() == at::ScalarType::BFloat16);
    HABANA_ASSERT(rhs.scalar_type() == lhs.scalar_type());
    HABANA_ASSERT(bias.scalar_type() == lhs.scalar_type());
    HABANA_ASSERT(lhs.size(0) == rhs.size(0));
    HABANA_ASSERT(lhs.size(2) == rhs.size(1));

    const std::vector<int64_t> output_shape{
        lhs.size(0), lhs.size(1), rhs.size(2)};
    HABANA_ASSERT(bias.sizes().vec() == output_shape);

    if (hint_mode_ == HintMode::kBundle) {
      setContextHints("user_bundle_id:739251");
    } else if (hint_mode_ == HintMode::kSchedule) {
      setContextHints("group_id:739251;exec_order:0");
    }
    auto gemm = BuildNode(
        this,
        graph,
        {"batch_gemm",
         {syn_in(0), syn_in(1)},
         {{output_shape, at::ScalarType::BFloat16}}});
    if (hint_mode_ == HintMode::kSchedule) {
      setContextHints("group_id:739251;exec_order:1");
    }
    auto output = BuildNode(
        this,
        graph,
        {"add_fwd_bf16",
         {gemm.at(0).get(), syn_in(2)},
         {{output_shape, at::ScalarType::BFloat16, 0}}});
    syn_out(0) = std::move(output.at(0));
  }

 private:
  HintMode hint_mode_;
};

class FlashQlaBf16BmmF32 final : public habana::OpBackend {
 public:
  FlashQlaBf16BmmF32(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + std::string("flashqla_bf16_bmm_f32"),
            scalar_type,
            {0},
            {},
            {},
            false) {
    SetOutputMetaFn([](const at::Stack& stack) {
      const auto& lhs = stack.at(0).toTensor();
      const auto& rhs = stack.at(1).toTensor();
      return habana::OutputMetaDataVector{{
          at::ScalarType::Float,
          {lhs.size(0), lhs.size(1), rhs.size(2)},
      }};
    });
  }

  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto& lhs = stack.at(0).toTensor();
    const auto& rhs = stack.at(1).toTensor();
    HABANA_ASSERT(lhs.dim() == 3 && rhs.dim() == 3);
    HABANA_ASSERT(lhs.scalar_type() == at::ScalarType::BFloat16);
    HABANA_ASSERT(rhs.scalar_type() == lhs.scalar_type());
    HABANA_ASSERT(lhs.size(0) == rhs.size(0));
    HABANA_ASSERT(lhs.size(2) == rhs.size(1));

    const std::vector<int64_t> output_shape{
        lhs.size(0), lhs.size(1), rhs.size(2)};
    auto output = BuildNode(
        this,
        graph,
        {"batch_gemm",
         {syn_in(0), syn_in(1)},
         {{output_shape, at::ScalarType::Float, 0}}});
    syn_out(0) = std::move(output.at(0));
  }
};

class FlashQlaRecurrentScan final : public habana::OpBackend {
 public:
  FlashQlaRecurrentScan(
      int device_id,
      c10::ScalarType scalar_type,
      int bundle_block_size)
      : OpBackend(
            device_id,
            NO_TPC + std::string("flashqla_recurrent_scan_probe"),
            scalar_type,
            {0, 1},
            {},
            {},
            false),
        bundle_block_size_(bundle_block_size) {
    SetOutputMetaFn([](const at::Stack& stack) {
      const auto& core = stack.at(1).toTensor();
      const auto& initial_state = stack.at(3).toTensor();
      return habana::OutputMetaDataVector{
          {core.scalar_type(), core.sizes().vec()},
          {initial_state.scalar_type(), initial_state.sizes().vec()},
      };
    });
  }

  void AddNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack) override {
    const auto& projections = stack.at(0).toTensor();
    const auto& core = stack.at(1).toTensor();
    const auto& state_bias = stack.at(2).toTensor();
    const auto& initial_state = stack.at(3).toTensor();
    for (const auto* tensor : {
             &projections,
             &core,
             &state_bias,
             &initial_state,
         }) {
      HABANA_ASSERT(tensor->dim() == 3);
      HABANA_ASSERT(tensor->scalar_type() == at::ScalarType::BFloat16);
    }

    const int64_t heads = initial_state.size(0);
    const int64_t key_dim = initial_state.size(1);
    const int64_t value_dim = initial_state.size(2);
    const int64_t chunk_size = core.size(1);
    HABANA_ASSERT(heads > 0 && projections.size(0) % heads == 0);
    const int64_t chunks = projections.size(0) / heads;
    HABANA_ASSERT(chunks > 0);
    HABANA_ASSERT(projections.size(1) == chunk_size + key_dim);
    HABANA_ASSERT(projections.size(2) == key_dim);
    HABANA_ASSERT(core.size(0) == chunks * heads);
    HABANA_ASSERT(core.size(2) == value_dim);
    HABANA_ASSERT(state_bias.size(0) == chunks * heads);
    HABANA_ASSERT(state_bias.size(1) == key_dim);
    HABANA_ASSERT(state_bias.size(2) == value_dim);

    const auto dtype = at::ScalarType::BFloat16;
    const std::vector<int64_t> projection_chunk_shape{
        heads, chunk_size + key_dim, key_dim};
    const std::vector<int64_t> core_chunk_shape{
        heads, chunk_size, value_dim};
    const std::vector<int64_t> bias_chunk_shape{
        heads, key_dim, value_dim};
    const std::vector<int64_t> state_shape{heads, key_dim, value_dim};

    auto split_chunks = [&](synTensor input,
                            const std::vector<int64_t>& output_shape) {
      std::vector<habana::NodeAttr::NodeOutputAttr> outputs;
      outputs.reserve(chunks);
      for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        outputs.push_back({output_shape, dtype});
      }
      synAxisParams params{};
      params.axis = 2;
      return BuildNode(
          this,
          graph,
          {"split", {input}, std::move(outputs), &params, sizeof(params)});
    };

    auto projection_chunks = split_chunks(syn_in(0), projection_chunk_shape);
    auto bias_chunks = split_chunks(syn_in(2), bias_chunk_shape);

    auto set_bundle_hint = [&](int64_t chunk) {
      if (bundle_block_size_ <= 0) {
        return;
      }
      const int64_t bundle = 739300 + chunk / bundle_block_size_;
      setContextHints("user_bundle_id:" + std::to_string(bundle));
    };

    synTensor state_handle = syn_in(3);
    std::optional<synapse_helpers::tensor> state_owner;
    std::vector<synapse_helpers::tensor> block_outputs;
    std::vector<synapse_helpers::tensor> pending_outputs;
    const int64_t block_size =
        bundle_block_size_ > 0 ? bundle_block_size_ : chunks;
    pending_outputs.reserve(block_size);
    block_outputs.reserve((chunks + block_size - 1) / block_size);

    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
      set_bundle_hint(chunk);
      auto projected = BuildNode(
          this,
          graph,
           {"batch_gemm",
           {projection_chunks.at(chunk).get(), state_handle},
           {{{heads, chunk_size + key_dim, value_dim}, dtype}}});

      synAxisParams split_params{};
      split_params.axis = 1;
      set_bundle_hint(chunk);
      auto projected_parts = BuildNode(
          this,
          graph,
          {"split",
           {projected.at(0).get()},
           {{core_chunk_shape, dtype}, {state_shape, dtype}},
           &split_params,
           sizeof(split_params)});

      pending_outputs.push_back(std::move(projected_parts.at(0)));

      const bool is_last = chunk + 1 == chunks;
      std::optional<int> final_state_index =
          is_last ? std::optional<int>(1) : std::nullopt;
      set_bundle_hint(chunk);
      auto next_state = BuildNode(
          this,
          graph,
          {"add_fwd_bf16",
           {projected_parts.at(1).get(), bias_chunks.at(chunk).get()},
           {{state_shape, dtype, final_state_index}}});
      state_owner = std::move(next_state.at(0));
      state_handle = state_owner->get();

      const bool closes_block =
          (chunk + 1) % block_size == 0 || is_last;
      if (closes_block) {
        std::vector<synTensor> concat_inputs;
        concat_inputs.reserve(pending_outputs.size());
        for (auto& tensor : pending_outputs) {
          concat_inputs.push_back(tensor.get());
        }
        synConcatenateParams concat_params{};
        concat_params.axis = 2;
        const std::vector<int64_t> block_shape{
            static_cast<int64_t>(pending_outputs.size()) * heads,
            chunk_size,
            value_dim};
        set_bundle_hint(chunk);
        auto block_output = BuildNode(
            this,
            graph,
            {"concat",
             std::move(concat_inputs),
             {{block_shape, dtype}},
             &concat_params,
             sizeof(concat_params)});
        block_outputs.push_back(std::move(block_output.at(0)));
        pending_outputs.clear();
      }
    }

    std::vector<synTensor> concat_inputs;
    concat_inputs.reserve(block_outputs.size());
    for (auto& tensor : block_outputs) {
      concat_inputs.push_back(tensor.get());
    }
    synConcatenateParams concat_params{};
    concat_params.axis = 2;
    auto output = BuildNode(
        this,
        graph,
        {"concat",
         std::move(concat_inputs),
         {{core.sizes().vec(), dtype, 0}},
         &concat_params,
         sizeof(concat_params)});
    syn_out(0) = std::move(output.at(0));
    HABANA_ASSERT(state_owner.has_value());
    syn_out(1) = std::move(*state_owner);
  }

 private:
  int bundle_block_size_;
};

void register_probe_variant(const char* schema, HintMode hint_mode) {
  // The public descriptor supplies frontend allocation and Meta information.
  // The regular registry entry takes precedence during lowering and emits a
  // mixed MME/TPC Synapse subgraph instead of the descriptor's dummy GUID.
  habana::custom_op::registerUserCustomOp(
      schema,
      "flashqla_compound_probe_unused",
      probe_output_meta,
      nullptr);
  habana::KernelRegistry().add(
      schema,
      [hint_mode](const synDeviceId device_id, c10::ScalarType scalar_type) {
        return std::make_shared<FlashQlaCompoundProbe>(
            device_id,
            scalar_type,
            hint_mode);
      });
}

bool register_probe() {
  register_probe_variant(kSchema, HintMode::kNone);
  register_probe_variant(kBundledSchema, HintMode::kBundle);
  register_probe_variant(kScheduledSchema, HintMode::kSchedule);
  habana::custom_op::registerUserCustomOp(
      kBf16BmmF32Schema,
      "flashqla_bf16_bmm_f32_unused",
      bf16_bmm_f32_output_meta,
      nullptr);
  habana::KernelRegistry().add(
      kBf16BmmF32Schema,
      [](const synDeviceId device_id, c10::ScalarType scalar_type) {
        return std::make_shared<FlashQlaBf16BmmF32>(
            device_id,
            scalar_type);
      });
  for (const auto& variant : {
           std::pair<const char*, int>{kRecurrentScanSchema, 0},
           std::pair<const char*, int>{kRecurrentScanBundled16Schema, 16},
       }) {
    habana::custom_op::registerUserCustomOp(
        variant.first,
        "flashqla_recurrent_scan_probe_unused",
        recurrent_scan_output_meta,
        nullptr);
    habana::KernelRegistry().add(
        variant.first,
        [block_size = variant.second](
            const synDeviceId device_id,
            c10::ScalarType scalar_type) {
          return std::make_shared<FlashQlaRecurrentScan>(
              device_id,
              scalar_type,
              block_size);
        });
  }
  return true;
}

const bool kRegistered = register_probe();

at::Tensor compound_probe_hpu_impl(
    const char* schema,
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& bias) {
  TORCH_CHECK(kRegistered);
  TORCH_CHECK(lhs.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(rhs.scalar_type() == lhs.scalar_type());
  TORCH_CHECK(bias.scalar_type() == lhs.scalar_type());
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          schema);
  return descriptor.execute({lhs, rhs, bias}).at(0);
}

at::Tensor compound_probe_hpu(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& bias) {
  return compound_probe_hpu_impl(kSchema, lhs, rhs, bias);
}

at::Tensor compound_probe_bundled_hpu(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& bias) {
  return compound_probe_hpu_impl(kBundledSchema, lhs, rhs, bias);
}

at::Tensor compound_probe_scheduled_hpu(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor& bias) {
  return compound_probe_hpu_impl(kScheduledSchema, lhs, rhs, bias);
}

at::Tensor bf16_bmm_f32_hpu(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  TORCH_CHECK(kRegistered);
  TORCH_CHECK(lhs.scalar_type() == at::ScalarType::BFloat16);
  TORCH_CHECK(rhs.scalar_type() == lhs.scalar_type());
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kBf16BmmF32Schema);
  return descriptor.execute({lhs, rhs}).at(0);
}

std::tuple<at::Tensor, at::Tensor> recurrent_scan_hpu_impl(
    const char* schema,
    const at::Tensor& projections,
    const at::Tensor& core,
    const at::Tensor& state_bias,
    const at::Tensor& initial_state) {
  TORCH_CHECK(kRegistered);
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          schema);
  auto outputs = descriptor.execute(
      {projections, core, state_bias, initial_state});
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> recurrent_scan_hpu(
    const at::Tensor& projections,
    const at::Tensor& core,
    const at::Tensor& state_bias,
    const at::Tensor& initial_state) {
  return recurrent_scan_hpu_impl(
      kRecurrentScanSchema,
      projections,
      core,
      state_bias,
      initial_state);
}

std::tuple<at::Tensor, at::Tensor> recurrent_scan_bundled16_hpu(
    const at::Tensor& projections,
    const at::Tensor& core,
    const at::Tensor& state_bias,
    const at::Tensor& initial_state) {
  return recurrent_scan_hpu_impl(
      kRecurrentScanBundled16Schema,
      projections,
      core,
      state_bias,
      initial_state);
}

at::Tensor compound_probe_meta(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Tensor&) {
  return lhs.new_empty({lhs.size(0), lhs.size(1), rhs.size(2)});
}

at::Tensor bf16_bmm_f32_meta(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  return lhs.new_empty(
      {lhs.size(0), lhs.size(1), rhs.size(2)},
      at::TensorOptions().dtype(at::ScalarType::Float));
}

std::tuple<at::Tensor, at::Tensor> recurrent_scan_meta(
    const at::Tensor&,
    const at::Tensor& core,
    const at::Tensor&,
    const at::Tensor& initial_state) {
  return {at::empty_like(core), at::empty_like(initial_state)};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def(
      "flashqla_compound_probe(Tensor lhs, Tensor rhs, Tensor bias) -> Tensor");
  m.def(
      "flashqla_compound_probe_bundled(Tensor lhs, Tensor rhs, Tensor bias) "
      "-> Tensor");
  m.def(
      "flashqla_compound_probe_scheduled(Tensor lhs, Tensor rhs, Tensor bias) "
      "-> Tensor");
  m.def("flashqla_bf16_bmm_f32(Tensor lhs, Tensor rhs) -> Tensor");
  m.def(
      "flashqla_recurrent_scan_probe("
      "Tensor projections, Tensor core, Tensor state_bias, "
      "Tensor initial_state) -> (Tensor, Tensor)");
  m.def(
      "flashqla_recurrent_scan_probe_bundled16("
      "Tensor projections, Tensor core, Tensor state_bias, "
      "Tensor initial_state) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("flashqla_compound_probe", compound_probe_hpu);
  m.impl("flashqla_compound_probe_bundled", compound_probe_bundled_hpu);
  m.impl("flashqla_compound_probe_scheduled", compound_probe_scheduled_hpu);
  m.impl("flashqla_bf16_bmm_f32", bf16_bmm_f32_hpu);
  m.impl("flashqla_recurrent_scan_probe", recurrent_scan_hpu);
  m.impl(
      "flashqla_recurrent_scan_probe_bundled16",
      recurrent_scan_bundled16_hpu);
}

TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("flashqla_compound_probe", compound_probe_meta);
  m.impl("flashqla_compound_probe_bundled", compound_probe_meta);
  m.impl("flashqla_compound_probe_scheduled", compound_probe_meta);
  m.impl("flashqla_bf16_bmm_f32", bf16_bmm_f32_meta);
  m.impl("flashqla_recurrent_scan_probe", recurrent_scan_meta);
  m.impl(
      "flashqla_recurrent_scan_probe_bundled16",
      recurrent_scan_meta);
}
