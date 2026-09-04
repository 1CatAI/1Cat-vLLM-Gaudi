// SPDX-License-Identifier: Apache-2.0

#include <habanalabs/perf_lib_layer_params.h>
#include <habanalabs/synapse_api.h>
#include <hccl.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/helpers/generic_resource_holder.h"
#include "backend/helpers/tensor_info.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_kernels/hccl_kernels.h"
#include "habana_serialization/deserializers.h"
#include "habana_serialization/serializers.h"
#include "python_packages/habana_frameworks/torch/distributed/hccl/process_group_eager_hccl.hpp"

namespace {

constexpr char kAddGuid[] = "add_fwd_bf16";
constexpr char kRmsNormGuid[] = "rms_norm_ex_fwd_bf16";
std::atomic<uint64_t> g_collective_launch_count{0};
std::atomic<uint64_t> g_collective_allocate_count{0};
std::atomic<uint64_t> g_collective_allocate_dry_count{0};
std::atomic<uint64_t> g_collective_allocate_real_count{0};
std::atomic<uint64_t> g_collective_frontend_count{0};
std::mutex g_collective_launch_mutex;

void checkSynapse(synStatus status, const char *operation) {
  TORCH_CHECK(status == synSuccess, operation,
              " failed with synStatus=", static_cast<int>(status));
}

struct FusedRecipe {
  synGraphHandle graph = nullptr;
  synRecipeHandle handle = nullptr;
  std::vector<synTensor> tensors;
  std::vector<synSectionHandle> sections;
  uint64_t workspace_address = 0;
  uint64_t workspace_bytes = 0;
};

synTensor createPersistentTensor(FusedRecipe &recipe, const char *name,
                                 synDataType type,
                                 const std::vector<uint64_t> &sizes) {
  TORCH_CHECK(!sizes.empty(), "A persistent tensor must have a rank");
  TORCH_CHECK(sizes.size() <= HABANA_DIM_MAX,
              "Tensor rank exceeds Synapse's maximum rank");

  synTensorDescriptor descriptor = {};
  descriptor.m_dataType = type;
  descriptor.m_dims = static_cast<unsigned>(sizes.size());
  descriptor.m_name = name;
  for (size_t index = 0; index < sizes.size(); ++index) {
    descriptor.m_sizes[index] = sizes[index];
    descriptor.m_minSizes[index] = sizes[index];
  }

  synSectionHandle section = nullptr;
  checkSynapse(synSectionCreate(&section, 0, recipe.graph), "synSectionCreate");
  checkSynapse(synSectionSetPersistent(section, true),
               "synSectionSetPersistent");
  recipe.sections.push_back(section);

  synTensor tensor = nullptr;
  checkSynapse(synTensorCreate(&tensor, &descriptor, section, 0),
               "synTensorCreate");
  recipe.tensors.push_back(tensor);
  return tensor;
}

struct RecipeKey {
  uint64_t elements;
  uint64_t hidden_size;
  uint32_t epsilon_bits;

  bool operator==(const RecipeKey &other) const {
    return elements == other.elements && hidden_size == other.hidden_size &&
           epsilon_bits == other.epsilon_bits;
  }
};

struct RecipeKeyHash {
  size_t operator()(const RecipeKey &key) const {
    size_t value = std::hash<uint64_t>{}(key.elements);
    value ^= std::hash<uint64_t>{}(key.hidden_size) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    value ^= std::hash<uint32_t>{}(key.epsilon_bits) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    return value;
  }
};

uint32_t floatBits(float value) {
  uint32_t result = 0;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

class FusedRecipeCache {
public:
  FusedRecipe *get(uint64_t elements, uint64_t hidden_size, float epsilon) {
    RecipeKey key{elements, hidden_size, floatBits(epsilon)};
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = recipes_.find(key);
    if (found != recipes_.end())
      return found->second.get();

    TORCH_CHECK(hidden_size > 0 && elements % hidden_size == 0,
                "Element count must be divisible by hidden size");
    const uint64_t rows = elements / hidden_size;
    auto recipe = std::make_unique<FusedRecipe>();
    checkSynapse(synGraphCreate(&recipe->graph, synDeviceGaudi2),
                 "synGraphCreate");

    const std::vector<uint64_t> activation_sizes{hidden_size, rows, 1};
    const std::vector<uint64_t> weight_sizes{hidden_size};
    const std::vector<uint64_t> inverse_sizes{1, rows, 1};
    synTensor reduced = createPersistentTensor(*recipe, "reduced",
                                               syn_type_bf16, activation_sizes);
    synTensor residual = createPersistentTensor(
        *recipe, "residual", syn_type_bf16, activation_sizes);
    synTensor weight =
        createPersistentTensor(*recipe, "weight", syn_type_bf16, weight_sizes);
    synTensor residual_out = createPersistentTensor(
        *recipe, "residual_out", syn_type_bf16, activation_sizes);
    synTensor normalized = createPersistentTensor(
        *recipe, "normalized", syn_type_bf16, activation_sizes);
    synTensor inverse_rms = createPersistentTensor(
        *recipe, "inverse_rms", syn_type_float, inverse_sizes);

    synTensor add_inputs[] = {reduced, residual};
    synTensor add_outputs[] = {residual_out};
    checkSynapse(synNodeCreate(recipe->graph, add_inputs, add_outputs, 2, 1,
                               nullptr, 0, kAddGuid, "tp2_residual_add",
                               nullptr, nullptr),
                 "synNodeCreate(tp2_residual_add)");

    ns_LayerNormKernel::ParamsRmsNormV3 params = {};
    params.epsValid = true;
    params.eps = epsilon;
    params.fastMath = false;
    params.hasResidual = false;
    params.addOneToWeight = false;
    synTensor norm_inputs[] = {residual_out, weight};
    synTensor norm_outputs[] = {normalized, inverse_rms};
    checkSynapse(synNodeCreate(recipe->graph, norm_inputs, norm_outputs, 2, 2,
                               &params, sizeof(params), kRmsNormGuid,
                               "tp2_residual_rms_norm", nullptr, nullptr),
                 "synNodeCreate(tp2_residual_rms_norm)");

    const std::string recipe_name =
        "tp2_ar_residual_rms_norm_n" + std::to_string(elements) + "_h" +
        std::to_string(hidden_size) + "_e" + std::to_string(key.epsilon_bits);
    checkSynapse(synGraphCompile(&recipe->handle, recipe->graph,
                                 recipe_name.c_str(), nullptr),
                 "synGraphCompile(tp2_ar_residual_rms_norm)");
    checkSynapse(synWorkspaceGetSize(&recipe->workspace_bytes, recipe->handle),
                 "synWorkspaceGetSize(tp2_ar_residual_rms_norm)");
    if (recipe->workspace_bytes > 0) {
      auto &device = habana::HPUDeviceContext::get_device();
      recipe->workspace_address =
          device.get_workspace_buffer(recipe->workspace_bytes);
      TORCH_CHECK(recipe->workspace_address != 0,
                  "HPU backend returned a null fused recipe workspace");
    }

    FusedRecipe *result = recipe.get();
    recipes_.emplace(key, std::move(recipe));
    return result;
  }

private:
  std::mutex mutex_;
  std::unordered_map<RecipeKey, std::unique_ptr<FusedRecipe>, RecipeKeyHash>
      recipes_;
};

FusedRecipeCache &fusedRecipeCache() {
  static auto *cache = new FusedRecipeCache();
  return *cache;
}

void launchFusedRecipe(synStreamHandle stream,
                       const std::array<uint64_t, 6> &addresses,
                       uint64_t elements, uint64_t hidden_size, float epsilon) {
  FusedRecipe *recipe = fusedRecipeCache().get(elements, hidden_size, epsilon);
  std::array<synLaunchTensorInfo, 6> launch_info = {};
  const char *names[] = {
      "reduced",      "residual",   "weight",
      "residual_out", "normalized", "inverse_rms",
  };
  for (size_t index = 0; index < launch_info.size(); ++index) {
    launch_info[index].tensorName = names[index];
    launch_info[index].pTensorAddress = addresses[index];
    launch_info[index].tensorType = DATA_TENSOR;
  }
  checkSynapse(synLaunch(stream, launch_info.data(),
                         static_cast<uint32_t>(launch_info.size()),
                         recipe->workspace_address, recipe->handle,
                         SYN_FLAGS_TENSOR_NAME),
               "synLaunch(tp2_ar_residual_rms_norm)");
}

void validateTensor(const at::Tensor &tensor, const at::Tensor &reference,
                    const char *name, at::ScalarType dtype) {
  TORCH_CHECK(tensor.device().type() == at::kHPU, name, " must be on HPU");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " shape mismatch");
}

void runFusedAllReduceNorm(
    const std::shared_ptr<habana::HcclCommunicator> &communicator,
    at::Tensor partial, at::Tensor residual, at::Tensor weight,
    at::Tensor reduced, at::Tensor normalized, at::Tensor residual_out,
    at::Tensor inverse_rms, float epsilon,
    synapse_helpers::hpuStream_t hpu_stream) {
  auto device_context = communicator->getDeviceCtxt();
  auto &device = habana::HPUDeviceContext::get_device();
  auto &compute_stream = device.get_stream(hpu_stream);
  const synStreamHandle stream = compute_stream;

  auto resources = std::make_shared<GenericResourceHolder>();
  std::array<at::Tensor, 7> tensors = {
      partial, residual, weight, reduced, normalized, residual_out, inverse_rms,
  };
  std::vector<void *> tensor_addresses;
  tensor_addresses.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    resources->add_tensor(tensor);
    tensor_addresses.push_back(tensor.data_ptr());
  }
  device_context->lock_address(tensor_addresses, resources->get_address_lock());
  const auto &locked = *resources->get_address_lock();

  const hcclResult_t result =
      hcclAllReduce(reinterpret_cast<const void *>(locked.at(0)),
                    reinterpret_cast<void *>(locked.at(3)),
                    static_cast<size_t>(partial.numel()), hcclBfloat16, hcclSum,
                    *(communicator->GetHcclHandle()), stream);
  TORCH_CHECK(result == hcclSuccess, "HCCL all-reduce returned error ", result);

  launchFusedRecipe(stream,
                    {
                        locked.at(3),
                        locked.at(1),
                        locked.at(2),
                        locked.at(5),
                        locked.at(4),
                        locked.at(6),
                    },
                    static_cast<uint64_t>(partial.numel()),
                    static_cast<uint64_t>(weight.numel()), epsilon);

  auto normalized_storage = reinterpret_cast<synapse_helpers::device_ptr>(
      normalized.storage().data_ptr().get());
  auto residual_storage = reinterpret_cast<synapse_helpers::device_ptr>(
      residual_out.storage().data_ptr().get());
  auto &recipe_counter = device_context->get_active_recipe_counter();
  recipe_counter.increase();
  device.register_producer_on_stream({normalized_storage, residual_storage},
                                     compute_stream,
                                     [resources, &recipe_counter]() mutable {
                                       resources.reset();
                                       recipe_counter.decrease_and_notify();
                                     });
}

std::pair<at::Tensor, at::Tensor>
allReduceResidualRmsNorm(c10d::ProcessGroupEagerHCCL *backend,
                         const at::Tensor &partial, const at::Tensor &residual,
                         const at::Tensor &weight, const at::Tensor &reduced,
                         const at::Tensor &normalized,
                         const at::Tensor &residual_out,
                         const at::Tensor &inverse_rms, float epsilon) {
  TORCH_CHECK(partial.dim() >= 2, "partial must have at least two dimensions");
  TORCH_CHECK(partial.scalar_type() == at::kBFloat16,
              "The fused TP2 path supports BF16 activations only");
  TORCH_CHECK(partial.is_contiguous(), "partial must be contiguous");
  TORCH_CHECK(partial.numel() > 0, "partial must not be empty");
  validateTensor(residual, partial, "residual", at::kBFloat16);
  validateTensor(reduced, partial, "reduced", at::kBFloat16);
  validateTensor(normalized, partial, "normalized", at::kBFloat16);
  validateTensor(residual_out, partial, "residual_out", at::kBFloat16);
  TORCH_CHECK(weight.device().type() == at::kHPU, "weight must be on HPU");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16,
              "weight must have BF16 dtype");
  TORCH_CHECK(weight.dim() == 1 && weight.is_contiguous(),
              "weight must be a contiguous vector");
  TORCH_CHECK(partial.size(-1) == weight.numel(),
              "The last activation dimension must match weight");
  TORCH_CHECK(inverse_rms.device().type() == at::kHPU,
              "inverse_rms must be on HPU");
  TORCH_CHECK(inverse_rms.scalar_type() == at::kFloat,
              "inverse_rms must have FP32 dtype");
  TORCH_CHECK(inverse_rms.is_contiguous(), "inverse_rms must be contiguous");
  TORCH_CHECK(inverse_rms.numel() == partial.numel() / weight.numel(),
              "inverse_rms element count mismatch");
  TORCH_CHECK(partial.data_ptr() != residual.data_ptr() &&
                  partial.data_ptr() != reduced.data_ptr() &&
                  partial.data_ptr() != normalized.data_ptr() &&
                  partial.data_ptr() != residual_out.data_ptr() &&
                  residual.data_ptr() != reduced.data_ptr() &&
                  residual.data_ptr() != normalized.data_ptr() &&
                  residual.data_ptr() != residual_out.data_ptr() &&
                  reduced.data_ptr() != normalized.data_ptr() &&
                  reduced.data_ptr() != residual_out.data_ptr() &&
                  normalized.data_ptr() != residual_out.data_ptr(),
              "Activation tensors must use distinct storage");

  auto communicator = backend->lowLatencyCommunicator();
  TORCH_CHECK(communicator != nullptr, "HCCL communicator is not initialized");
  const auto hpu_stream = c10::hpu::getCurrentHPUStream().stream();
  const bool pipeline_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);
  if (pipeline_enabled) {
    std::array<at::Tensor, 7> backend_tensors = {
        habana::eager::HbEagerTensorPool::get_backend_tensor(partial),
        habana::eager::HbEagerTensorPool::get_backend_tensor(residual),
        habana::eager::HbEagerTensorPool::get_backend_tensor(weight),
        habana::eager::HbEagerTensorPool::get_backend_tensor(reduced),
        habana::eager::HbEagerTensorPool::get_backend_tensor(normalized),
        habana::eager::HbEagerTensorPool::get_backend_tensor(residual_out),
        habana::eager::HbEagerTensorPool::get_backend_tensor(inverse_rms),
    };
    for (const auto &tensor : backend_tensors) {
      habana::get_tensor_extra_meta(tensor)->set_tensor_pipelined();
    }
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        [communicator, backend_tensors = std::move(backend_tensors), epsilon,
         hpu_stream]() mutable {
          runFusedAllReduceNorm(
              communicator, std::move(backend_tensors[0]),
              std::move(backend_tensors[1]), std::move(backend_tensors[2]),
              std::move(backend_tensors[3]), std::move(backend_tensors[4]),
              std::move(backend_tensors[5]), std::move(backend_tensors[6]),
              epsilon, hpu_stream);
        });
  } else {
    habana::eager::JoinPendingPipelineThreads();
    runFusedAllReduceNorm(communicator, partial, residual, weight, reduced,
                          normalized, residual_out, inverse_rms, epsilon,
                          hpu_stream);
  }
  return {normalized, residual_out};
}

class Tp2FusedAllReduceRmsNormOperator : public habana::CollectiveOperator {
public:
  Tp2FusedAllReduceRmsNormOperator(int device_id, c10::ScalarType scalar_type)
      : CollectiveOperator("hccl::tp2_allreduce_residual_rms_norm", device_id,
                           scalar_type) {
    CreateSynContext(device_id);
  }

  habana::InferOutputMetaRetType
  InferOutputMeta(torch::jit::Stack &inputs) override {
    TORCH_CHECK(inputs.size() == 5 && inputs.at(0).isTensor(),
                "Unexpected fused collective metadata inputs");
    const auto &reference = inputs.at(0).toTensor();
    const auto activation_shape = reference.sizes().vec();
    const auto memory_format = reference.suggest_memory_format();

    auto packed_shape = activation_shape;
    packed_shape.insert(packed_shape.begin(), 3);
    habana::InferOutputMetaRetType output_meta;
    output_meta.AddOutputTensor(habana::TensorMetaData(
        packed_shape,
        habana::HabanaOperator::CalculateStrides(
            packed_shape, at::MemoryFormat::Contiguous),
        reference.scalar_type(), at::MemoryFormat::Contiguous));
    auto inverse_shape = activation_shape;
    inverse_shape.back() = 1;
    output_meta.AddOutputTensor(habana::TensorMetaData(
        inverse_shape,
        habana::HabanaOperator::CalculateStrides(inverse_shape, memory_format),
        at::kFloat, memory_format));
    return output_meta;
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph &graph, torch::jit::Stack &inputs,
      const habana::OutputMetaDataVector &output_metadata) override {
    g_collective_allocate_count.fetch_add(1, std::memory_order_relaxed);
    (graph.is_dry_run() ? g_collective_allocate_dry_count
                        : g_collective_allocate_real_count)
        .fetch_add(1, std::memory_order_relaxed);
    TORCH_CHECK(inputs.size() == 5, "Unexpected fused collective input count");
    for (size_t index = 0; index < 3; ++index) {
      TORCH_CHECK(inputs.at(index).isTensor(),
                  "The first three fused collective inputs must be tensors");
    }
    TORCH_CHECK(
        inputs.at(3).isScalar() && inputs.at(4).isScalar(),
        "The fused collective epsilon and communicator must be scalars");
    TORCH_CHECK(output_metadata.size() == 2,
                "Unexpected fused collective output count");
    epsilon_ = inputs.at(3).toDouble();
    comm_id_ = inputs.at(4).toInt();

    if (p_context_->pt_inputs_.empty()) {
      for (size_t index = 0; index < 3; ++index) {
        p_context_->pt_inputs_.emplace_back(inputs.at(index).toTensor());
      }
    }
    const auto &reference = inputs.at(0).toTensor();
    auto packed_shape = reference.sizes().vec();
    packed_shape.insert(packed_shape.begin(), 3);
    auto packed = !graph.is_dry_run() &&
                          output_metadata.at(0).allocated_tensor.has_value()
                      ? output_metadata.at(0).allocated_tensor.value()
                      : habana::createPTTensor(
                            reference, packed_shape, reference.options(),
                            at::MemoryFormat::Contiguous,
                            output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, packed, output_metadata.at(0));
    std::vector<int64_t> inverse_shape = reference.sizes().vec();
    inverse_shape.back() = 1;
    auto inverse_rms = !graph.is_dry_run() &&
                               output_metadata.at(1).allocated_tensor.has_value()
                           ? output_metadata.at(1).allocated_tensor.value()
                           : habana::createPTTensor(
                                 reference, inverse_shape,
                                 reference.options().dtype(at::kFloat),
                                 reference.suggest_memory_format(),
                                 output_metadata.at(1).persistent);
    AllocateSynapseOutput(graph, inverse_rms, output_metadata.at(1));
  }

  void Serialize(std::ostream &stream) const override {
    serialization::serialize(stream, epsilon_);
    serialization::serialize(stream, comm_id_);
  }

  void Deserialize(std::istream &stream) override {
    serialization::deserialize(stream, epsilon_);
    serialization::deserialize(stream, comm_id_);
  }

  void RunCollective(
      const std::vector<PtTensorInfoShared> &inputs,
      std::vector<at::Tensor> &pt_inputs, std::vector<at::Tensor> &pt_outputs,
      bool async,
      synapse_helpers::event_done_callback done_callback) const override {
    (void)inputs;
    (void)pt_inputs;
    (void)pt_outputs;
    (void)async;
    (void)done_callback;
    TORCH_CHECK(false,
                "TP2 fused collective requires per-node output metadata");
  }

  void RunCollective(
      const std::vector<PtTensorInfoShared> &inputs,
      const std::vector<PtTensorInfoShared> &outputs,
      std::vector<at::Tensor> &pt_inputs, std::vector<at::Tensor> &pt_outputs,
      bool async,
      synapse_helpers::event_done_callback done_callback) const override {
    std::lock_guard<std::mutex> launch_guard(g_collective_launch_mutex);
    g_collective_launch_count.fetch_add(1, std::memory_order_relaxed);
    TORCH_CHECK(inputs.size() == 5, "Unexpected fused collective tensor info");
    TORCH_CHECK(outputs.size() == 2 && outputs.at(0) != nullptr &&
                    outputs.at(1) != nullptr,
                "Unexpected fused collective output metadata");
    auto communicator = habana::HcclCommunicator::Get(comm_id_);
    TORCH_CHECK(communicator != nullptr, "HCCL communicator is unavailable");
    auto device_context = communicator->getDeviceCtxt();
    const synStreamHandle stream = communicator->getCommStream();

    auto resources = std::make_shared<GenericResourceHolder>();
    for (const auto &tensor : pt_inputs) {
      resources->add_tensor(tensor);
    }
    for (const auto &tensor : pt_outputs) {
      resources->add_tensor(tensor);
    }

    std::vector<void *> buffers;
    buffers.reserve(5);
    for (size_t index = 0; index < 3; ++index) {
      const auto &info = inputs.at(index);
      TORCH_CHECK(info != nullptr, "Missing fused collective tensor metadata");
      device_context->prepare_stream(stream, info->get_buffer_start_syn());
      buffers.push_back(info->get_buffer());
    }
    for (const auto &output : outputs) {
      device_context->prepare_stream(stream, output->get_buffer_start_syn());
      buffers.push_back(output->get_buffer());
    }
    device_context->lock_address(buffers, resources->get_address_lock());
    const auto &locked = *resources->get_address_lock();

    const hcclResult_t result = hcclAllReduce(
        reinterpret_cast<const void *>(locked.at(0)),
        reinterpret_cast<void *>(locked.at(3) +
                                 2 * inputs.at(0)->get_size()),
        static_cast<size_t>(inputs.at(0)->get_numel()), hcclBfloat16, hcclSum,
        *(communicator->GetHcclHandle()), stream);
    TORCH_CHECK(result == hcclSuccess, "HCCL all-reduce returned error ",
                result);
    launchFusedRecipe(stream,
                      {
                          locked.at(3) + 2 * inputs.at(0)->get_size(),
                          locked.at(1),
                          locked.at(2),
                          locked.at(3) + inputs.at(0)->get_size(),
                          locked.at(3),
                          locked.at(4),
                      },
                      static_cast<uint64_t>(inputs.at(0)->get_numel()),
                      static_cast<uint64_t>(inputs.at(2)->get_numel()),
                      static_cast<float>(epsilon_));

    auto &recipe_counter = device_context->get_active_recipe_counter();
    recipe_counter.increase();
    device_context->submit_events(
        stream,
        reinterpret_cast<synapse_helpers::device_ptr>(
            outputs.at(0)->get_buffer_start()),
        [resources, &recipe_counter,
         done_callback = std::move(done_callback)]() mutable {
          resources.reset();
          recipe_counter.decrease_and_notify();
          done_callback();
        });
    if (!async) {
      checkSynapse(synStreamSynchronize(stream),
                   "synStreamSynchronize(tp2_ar_residual_rms_norm)");
    }
  }

private:
  double epsilon_ = 0;
  int64_t comm_id_ = 0;
};

std::tuple<at::Tensor, at::Tensor>
graphNativeAllReduceResidualRmsNorm(const at::Tensor &partial,
                                    const at::Tensor &residual,
                                    const at::Tensor &weight, double epsilon,
                                    int64_t comm_id) {
  g_collective_frontend_count.fetch_add(1, std::memory_order_relaxed);
  TORCH_CHECK(partial.scalar_type() == at::kBFloat16 &&
                  residual.scalar_type() == at::kBFloat16 &&
                  weight.scalar_type() == at::kBFloat16,
              "Graph-native TP2 fusion requires BF16 tensors");
  std::vector<int64_t> packed_shape = partial.sizes().vec();
  packed_shape.insert(packed_shape.begin(), 3);
  std::vector<int64_t> inverse_shape = partial.sizes().vec();
  inverse_shape.back() = 1;
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>>
      hpu_op("hccl::tp2_allreduce_residual_rms_norm",
             {partial, residual, weight, epsilon, comm_id},
             {std::move(packed_shape), std::move(inverse_shape)});
  hpu_op.set_scalar_types({at::kBFloat16, at::kFloat});
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor>
graphNativeAllReduceResidualRmsNormMeta(const at::Tensor &partial,
                                        const at::Tensor &residual,
                                        const at::Tensor &weight,
                                        double epsilon, int64_t comm_id) {
  (void)residual;
  (void)weight;
  (void)epsilon;
  (void)comm_id;
  std::vector<int64_t> packed_shape = partial.sizes().vec();
  packed_shape.insert(packed_shape.begin(), 3);
  std::vector<int64_t> inverse_shape = partial.sizes().vec();
  inverse_shape.back() = 1;
  return {torch::empty(packed_shape, partial.options()),
          torch::empty(inverse_shape, partial.options().dtype(at::kFloat))};
}

auto &kTp2FusedCollectiveRegistry = habana::KernelRegistry().add(
    "hccl::tp2_allreduce_residual_rms_norm",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<Tp2FusedAllReduceRmsNormOperator>(device_id,
                                                                node_type);
    });

} // namespace

TORCH_LIBRARY_FRAGMENT(hccl, library) {
  library.def(
      "tp2_allreduce_residual_rms_norm(Tensor partial, Tensor residual, "
      "Tensor weight, float epsilon, int comm_id) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(hccl, HPU, library) {
  library.impl("tp2_allreduce_residual_rms_norm",
               graphNativeAllReduceResidualRmsNorm);
}

TORCH_LIBRARY_IMPL(hccl, Meta, library) {
  library.impl("tp2_allreduce_residual_rms_norm",
               graphNativeAllReduceResidualRmsNormMeta);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("collective_launch_count", []() {
    return g_collective_launch_count.load(std::memory_order_relaxed);
  });
  module.def("collective_debug_counts", []() {
    return std::make_tuple(
        g_collective_frontend_count.load(std::memory_order_relaxed),
        g_collective_allocate_count.load(std::memory_order_relaxed),
        g_collective_allocate_dry_count.load(std::memory_order_relaxed),
        g_collective_allocate_real_count.load(std::memory_order_relaxed),
        g_collective_launch_count.load(std::memory_order_relaxed));
  });
  module.def(
      "communicator_id",
      [](const c10::intrusive_ptr<c10d::Backend> &backend) {
        auto *hccl_backend =
            dynamic_cast<c10d::ProcessGroupEagerHCCL *>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr,
                    "Fused TP2 all-reduce requires ProcessGroupEagerHCCL");
        auto communicator = hccl_backend->lowLatencyCommunicator();
        TORCH_CHECK(communicator != nullptr,
                    "HCCL communicator is not initialized");
        return communicator->GetId();
      },
      py::arg("backend"));
  module.def(
      "allreduce_residual_rms_norm_current_stream",
      [](const c10::intrusive_ptr<c10d::Backend> &backend,
         const at::Tensor &partial, const at::Tensor &residual,
         const at::Tensor &weight, const at::Tensor &reduced,
         const at::Tensor &normalized, const at::Tensor &residual_out,
         const at::Tensor &inverse_rms, double epsilon) {
        auto *hccl_backend =
            dynamic_cast<c10d::ProcessGroupEagerHCCL *>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr,
                    "Fused TP2 all-reduce requires ProcessGroupEagerHCCL");
        return allReduceResidualRmsNorm(
            hccl_backend, partial, residual, weight, reduced, normalized,
            residual_out, inverse_rms, static_cast<float>(epsilon));
      },
      py::arg("backend"), py::arg("partial"), py::arg("residual"),
      py::arg("weight"), py::arg("reduced"), py::arg("normalized"),
      py::arg("residual_out"), py::arg("inverse_rms"), py::arg("epsilon"));
}
