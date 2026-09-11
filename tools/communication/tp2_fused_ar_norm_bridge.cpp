// SPDX-License-Identifier: Apache-2.0

#include <habanalabs/perf_lib_layer_params.h>
#include <habanalabs/synapse_api.h>
#include <hccl.h>
#include <dlfcn.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <time.h>
#include <cstdint>
#include <cstdlib>
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
#include "backend/helpers/get_n_bytes.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_info.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "habana_eager/eager_tensor.h"
#include "habana_eager/graph_storage.h"
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
std::atomic<bool> g_use_tensor_ids{false};
std::atomic<uint64_t> g_state_dma_batches{0};
std::atomic<uint64_t> g_state_dma_tensors{0};
std::atomic<uint64_t> g_state_dma_bytes{0};
std::atomic<uint64_t> g_state_precise_records{0};
std::atomic<uint64_t> g_state_precise_waits{0};
std::atomic<bool> g_prepared_comm_enabled{false};
std::mutex g_collective_launch_mutex;

struct HostStageCounters {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> lock_ns{0};
  std::atomic<uint64_t> hccl_ns{0};
  std::atomic<uint64_t> recipe_ns{0};
  std::atomic<uint64_t> producer_ns{0};
  std::atomic<uint64_t> total_ns{0};

  void reset() {
    calls.store(0, std::memory_order_relaxed);
    lock_ns.store(0, std::memory_order_relaxed);
    hccl_ns.store(0, std::memory_order_relaxed);
    recipe_ns.store(0, std::memory_order_relaxed);
    producer_ns.store(0, std::memory_order_relaxed);
    total_ns.store(0, std::memory_order_relaxed);
  }
};

HostStageCounters g_host_stages;

using Tp2DirectExchangeFn = hcclResult_t (*)(
    const void *, void *, size_t, hcclDataType_t, hcclRedOp_t, hcclComm_t,
    synStreamHandle);

Tp2DirectExchangeFn resolveTp2DirectExchange() {
  static Tp2DirectExchangeFn function = [] {
    void *handle = dlopen("libhcl.so", RTLD_LAZY | RTLD_NOLOAD);
    TORCH_CHECK(handle != nullptr, "The loaded HCL library is unavailable");
    dlerror();
    void *symbol = dlsym(handle, "hcclTp2DirectExchange");
    const char *error = dlerror();
    TORCH_CHECK(symbol != nullptr && error == nullptr,
                "The loaded HCL library does not export "
                "hcclTp2DirectExchange: ",
                error == nullptr ? "unknown error" : error);
    return reinterpret_cast<Tp2DirectExchangeFn>(symbol);
  }();
  return function;
}

bool tp2ExchangeEnabled() {
  const char *algorithm =
      std::getenv("VLLM_HPU_TP2_FUSED_AR_NORM_DIRECT_ALGORITHM");
  return algorithm != nullptr && std::strcmp(algorithm, "tp2-exchange") == 0;
}

bool profileHostStagesEnabled() {
  const char *enabled =
      std::getenv("VLLM_HPU_TP2_FUSED_AR_NORM_PROFILE_HOST_STAGES");
  return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

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
  std::array<uint64_t, 7> tensor_ids{};
  std::array<synLaunchTensorInfo, 7> launch_template{};
  uint32_t tensor_count = 0;
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

synTensor createTransientTensor(FusedRecipe &recipe, const char *name,
                                synDataType type,
                                const std::vector<uint64_t> &sizes) {
  TORCH_CHECK(!sizes.empty(), "A transient tensor must have a rank");
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
  synTensor tensor = nullptr;
  checkSynapse(synTensorCreate(&tensor, &descriptor, nullptr, 0),
               "synTensorCreate(transient)");
  recipe.tensors.push_back(tensor);
  return tensor;
}

struct RecipeKey {
  uint64_t elements;
  uint64_t hidden_size;
  uint32_t epsilon_bits;
  bool exchange;

  bool operator==(const RecipeKey &other) const {
    return elements == other.elements && hidden_size == other.hidden_size &&
           epsilon_bits == other.epsilon_bits && exchange == other.exchange;
  }
};

struct RecipeKeyHash {
  size_t operator()(const RecipeKey &key) const {
    size_t value = std::hash<uint64_t>{}(key.elements);
    value ^= std::hash<uint64_t>{}(key.hidden_size) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    value ^= std::hash<uint32_t>{}(key.epsilon_bits) + 0x9e3779b9 +
             (value << 6) + (value >> 2);
    value ^= std::hash<bool>{}(key.exchange) + 0x9e3779b9 + (value << 6) +
             (value >> 2);
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
  FusedRecipe *get(uint64_t elements, uint64_t hidden_size, float epsilon,
                   bool exchange) {
    RecipeKey key{elements, hidden_size, floatBits(epsilon), exchange};
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = recipes_.find(key);
    if (found != recipes_.end())
      return found->second.get();

    TORCH_CHECK(hidden_size > 0 && elements % hidden_size == 0,
                "Element count must be divisible by hidden size");
    const uint64_t rows = elements / hidden_size;
    const std::string node_suffix = "_n" + std::to_string(elements) + "_h" +
                                    std::to_string(hidden_size);
    const std::string exchange_add_name =
        "tp2_local_peer_add" + node_suffix;
    const std::string residual_add_name =
        "tp2_residual_add" + node_suffix;
    const std::string rms_norm_name =
        "tp2_residual_rms_norm" + node_suffix;
    auto recipe = std::make_unique<FusedRecipe>();
    checkSynapse(synGraphCreate(&recipe->graph, synDeviceGaudi2),
                 "synGraphCreate");

    const std::vector<uint64_t> activation_sizes{hidden_size, rows, 1};
    const std::vector<uint64_t> weight_sizes{hidden_size};
    const std::vector<uint64_t> inverse_sizes{1, rows, 1};
    synTensor partial = exchange
        ? createPersistentTensor(*recipe, "partial", syn_type_bf16,
                                 activation_sizes)
        : nullptr;
    synTensor reduced = createPersistentTensor(
        *recipe, exchange ? "peer" : "reduced", syn_type_bf16,
        activation_sizes);
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

    synTensor norm_input = reduced;
    if (exchange) {
      synTensor local_peer_sum = createTransientTensor(
          *recipe, "local_peer_sum", syn_type_bf16, activation_sizes);
      synTensor exchange_add_inputs[] = {partial, reduced};
      synTensor exchange_add_outputs[] = {local_peer_sum};
      checkSynapse(synNodeCreate(recipe->graph, exchange_add_inputs,
                                 exchange_add_outputs, 2, 1, nullptr, 0,
                                 kAddGuid, exchange_add_name.c_str(), nullptr,
                                 nullptr),
                   "synNodeCreate(tp2_local_peer_add)");
      norm_input = local_peer_sum;
    }
    synTensor add_inputs[] = {norm_input, residual};
    synTensor add_outputs[] = {residual_out};
    checkSynapse(synNodeCreate(recipe->graph, add_inputs, add_outputs, 2, 1,
                               nullptr, 0, kAddGuid, residual_add_name.c_str(),
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
                               rms_norm_name.c_str(), nullptr, nullptr),
                 "synNodeCreate(tp2_residual_rms_norm)");

    const std::string recipe_name =
        "tp2_ar_residual_rms_norm_n" + std::to_string(elements) + "_h" +
        std::to_string(hidden_size) + "_e" + std::to_string(key.epsilon_bits) +
        (exchange ? "_exchange" : "_hccl");
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

    std::array<const char *, 7> tensor_names = {
        exchange ? "partial" : "reduced", exchange ? "peer" : "residual",
        exchange ? "residual" : "weight",
        exchange ? "weight" : "residual_out",
        exchange ? "residual_out" : "normalized",
        exchange ? "normalized" : "inverse_rms",
        exchange ? "inverse_rms" : nullptr,
    };
    recipe->tensor_count = exchange ? 7 : 6;
    checkSynapse(synTensorRetrieveIds(recipe->handle, tensor_names.data(),
                                      recipe->tensor_ids.data(),
                                      recipe->tensor_count),
                 "synTensorRetrieveIds(tp2_ar_residual_rms_norm)");
    for (size_t i = 0; i < recipe->tensor_count; ++i) {
      recipe->launch_template[i].tensorName = tensor_names[i];
      recipe->launch_template[i].tensorId = recipe->tensor_ids[i];
      recipe->launch_template[i].tensorType = DATA_TENSOR;
    }

    // The executable recipe is independent of the source graph, so release
    // the compile-only graph metadata once all persistent launch metadata has
    // been captured.
    checkSynapse(synGraphDestroy(recipe->graph),
                 "synGraphDestroy(tp2_ar_residual_rms_norm)");
    recipe->graph = nullptr;
    recipe->tensors.clear();
    recipe->sections.clear();

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

void launchFusedRecipe(synStreamHandle stream, uint64_t partial_address,
                       const std::array<uint64_t, 6> &addresses,
                       uint64_t elements, uint64_t hidden_size, float epsilon,
                       bool exchange, FusedRecipe* prepared_recipe = nullptr) {
  FusedRecipe *recipe = prepared_recipe ? prepared_recipe :
      fusedRecipeCache().get(elements, hidden_size, epsilon, exchange);
  if (prepared_recipe || g_prepared_comm_enabled.load(std::memory_order_relaxed)) {
    auto descriptors = recipe->launch_template;
    const size_t offset = exchange ? 1 : 0;
    if (exchange)
      descriptors[0].pTensorAddress = partial_address;
    for (size_t i = 0; i < addresses.size(); ++i)
      descriptors[i + offset].pTensorAddress = addresses[i];
    checkSynapse(synLaunch(stream, descriptors.data(), recipe->tensor_count,
                           recipe->workspace_address, recipe->handle,
                           g_use_tensor_ids.load(std::memory_order_relaxed) ? 0 : SYN_FLAGS_TENSOR_NAME),
                 "synLaunch(prepared_tp2_ar_norm)");
    return;
  }
  std::vector<synLaunchTensorInfo> launch_info;
  launch_info.reserve(exchange ? 7 : 6);
  if (exchange) {
    synLaunchTensorInfo partial_info = {};
    partial_info.tensorName = "partial";
    partial_info.pTensorAddress = partial_address;
    partial_info.tensorType = DATA_TENSOR;
    partial_info.tensorId = recipe->tensor_ids[0];
    launch_info.push_back(partial_info);
  }
  const char *names[] = {exchange ? "peer" : "reduced", "residual",
                         "weight", "residual_out", "normalized",
                         "inverse_rms"};
  for (size_t index = 0; index < addresses.size(); ++index) {
    synLaunchTensorInfo info = {};
    info.tensorName = names[index];
    info.pTensorAddress = addresses[index];
    info.tensorType = DATA_TENSOR;
    info.tensorId = recipe->tensor_ids[index + (exchange ? 1 : 0)];
    launch_info.push_back(info);
  }
  checkSynapse(synLaunch(stream, launch_info.data(),
                         static_cast<uint32_t>(launch_info.size()),
                         recipe->workspace_address, recipe->handle,
                         g_use_tensor_ids.load(std::memory_order_relaxed)
                             ? 0
                             : SYN_FLAGS_TENSOR_NAME),
               "synLaunch(tp2_ar_residual_rms_norm)");
}

void validateTensor(const at::Tensor &tensor, const at::Tensor &reference,
                    const char *name, at::ScalarType dtype) {
  TORCH_CHECK(tensor.device().type() == at::kHPU, name, " must be on HPU");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.sizes() == reference.sizes(), name, " shape mismatch");
}

void runTp2ExchangePeer(
    const std::shared_ptr<habana::HcclCommunicator> &communicator,
    at::Tensor partial, at::Tensor peer,
    synapse_helpers::hpuStream_t hpu_stream, bool reduction_only = false) {
  g_collective_launch_count.fetch_add(1, std::memory_order_relaxed);
  auto device_context = communicator->getDeviceCtxt();
  auto &device = habana::HPUDeviceContext::get_device();
  auto &compute_stream = device.get_stream(hpu_stream);
  const synStreamHandle stream = compute_stream;

  auto resources = std::make_shared<GenericResourceHolder>();
  std::array<at::Tensor, 2> tensors = {partial, peer};
  std::vector<void *> tensor_addresses;
  tensor_addresses.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    resources->add_tensor(tensor);
    tensor_addresses.push_back(tensor.data_ptr());
  }
  device_context->lock_address(tensor_addresses,
                               resources->get_address_lock());
  const auto &locked = *resources->get_address_lock();
  const hcclResult_t result = reduction_only
      ? hcclAllReduce(reinterpret_cast<const void *>(locked.at(0)),
                      reinterpret_cast<void *>(locked.at(1)),
                      static_cast<size_t>(partial.numel()), hcclBfloat16, hcclSum,
                      *(communicator->GetHcclHandle()), stream)
      : resolveTp2DirectExchange()(reinterpret_cast<const void *>(locked.at(0)),
                      reinterpret_cast<void *>(locked.at(1)),
                      static_cast<size_t>(partial.numel()), hcclBfloat16, hcclSum,
                      *(communicator->GetHcclHandle()), stream);
  TORCH_CHECK(result == hcclSuccess, "HCCL TP2 exchange returned error ",
              result);

  auto peer_storage = reinterpret_cast<synapse_helpers::device_ptr>(
      peer.storage().data_ptr().get());
  device.register_producer_on_stream(
      {peer_storage}, compute_stream, [resources]() mutable {
        resources.reset();
      });
}

struct GdnStateDMATicket {
  synapse_helpers::hpuEvent_t ready{}, done{};
  synapse_helpers::hpuStream_t copy_stream{};
  bool created = false;
  bool precise = false;
  std::unique_ptr<synapse_helpers::CachedEventHandle> physical_ready, physical_done;

  ~GdnStateDMATicket() {
    if (created && !precise && habana::HPUDeviceContext::is_device_acquired()) {
      auto &device = habana::HPUDeviceContext::get_device();
      device.delete_event(ready, false);
      device.delete_event(done, false);
    }
  }
};

void runGdnStateCopies(
    const std::shared_ptr<habana::HcclCommunicator> &communicator,
    const std::vector<at::Tensor> &sources,
    const std::vector<at::Tensor> &destinations,
    synapse_helpers::hpuStream_t hpu_stream,
    std::shared_ptr<GdnStateDMATicket> ticket = nullptr) {
  auto device_context = communicator->getDeviceCtxt();
  auto &device = habana::HPUDeviceContext::get_device();
  auto &stream = device.get_stream(hpu_stream);
  auto resources = std::make_shared<GenericResourceHolder>();
  std::vector<void *> addresses;
  std::vector<uint64_t> sizes;
  std::vector<synapse_helpers::device_ptr> inputs, outputs;
  uint64_t total_bytes = 0;
  for (size_t index = 0; index < sources.size(); ++index) {
    const auto &source = sources[index];
    const auto &destination = destinations[index];
    resources->add_tensor(source);
    resources->add_tensor(destination);
    addresses.push_back(source.data_ptr());
    const auto bytes = source.numel() * source.element_size();
    sizes.push_back(bytes);
    total_bytes += bytes;
    inputs.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
        source.storage().data_ptr().get()));
    outputs.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
        destination.storage().data_ptr().get()));
  }
  for (const auto &destination : destinations)
    addresses.push_back(destination.data_ptr());
  // Validate every range before submitting any DMA. data_ptr includes the
  // contiguous view offset; the copy length never includes unused pool rows.
  const auto count = sources.size();
  auto overlaps = [](void *a, uint64_t a_size, void *b, uint64_t b_size) {
    const auto x = reinterpret_cast<uint64_t>(a);
    const auto y = reinterpret_cast<uint64_t>(b);
    return x < y + b_size && y < x + a_size;
  };
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < count; ++j) {
      TORCH_CHECK(!overlaps(addresses[count + i], sizes[i], addresses[j], sizes[j]),
                  "GDN DMA destinations must not alias any source");
      if (i < j)
        TORCH_CHECK(!overlaps(addresses[count + i], sizes[i], addresses[count + j], sizes[j]),
                    "GDN DMA destinations must not overlap");
    }
  }
  // Generic callers retain storage producer dependencies. Queued group calls
  // own disjoint state slices and order every consumer/reset with tickets.
  // Registering a whole-pool write for them would serialize unrelated groups.
  // Precise tickets joined these exact source producers on their producer
  // stream before recording readiness. Its event transitively covers them.
  if (!ticket || !ticket->precise)
    device.add_wait_events_on_stream(inputs, stream);
  if (!ticket)
    device.add_wait_events_on_stream(outputs, stream);
  device_context->lock_address(addresses, resources->get_address_lock());
  const auto &locked = *resources->get_address_lock();
  std::vector<uint64_t> locked_sources, locked_destinations;
  for (size_t i = 0; i < count; ++i) {
    locked_sources.push_back(locked.at(i));
    locked_destinations.push_back(locked.at(count + i));
  }
  checkSynapse(synMemCopyAsyncMultiple(
                   stream, locked_sources.data(), sizes.data(),
                   locked_destinations.data(), synDmaDir::DRAM_TO_DRAM, count),
               "synMemCopyAsyncMultiple(gdn_state)");
  g_state_dma_batches.fetch_add(1, std::memory_order_relaxed);
  g_state_dma_tensors.fetch_add(count, std::memory_order_relaxed);
  g_state_dma_bytes.fetch_add(total_bytes, std::memory_order_relaxed);
  std::sort(outputs.begin(), outputs.end());
  outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
  if (ticket) {
    if (ticket->precise) {
      checkSynapse(synEventRecord(ticket->physical_done->get(), stream), "GDN physical done record");
      g_state_precise_records.fetch_add(1, std::memory_order_relaxed);
    } else {
      device.record_event(ticket->done, hpu_stream);
    }
    outputs.clear();
  }
  auto &recipe_counter = device_context->get_active_recipe_counter();
  recipe_counter.increase();
  device.register_producer_on_stream(
      std::move(outputs), stream, [resources, ticket, &recipe_counter]() mutable {
        resources.reset();
        ticket.reset();
        recipe_counter.decrease_and_notify();
      });
}

void validateGdnStates(const std::vector<at::Tensor> &sources,
                        const std::vector<at::Tensor> &destinations) {
  TORCH_CHECK(!sources.empty() && sources.size() == destinations.size(),
              "GDN DMA requires nonempty matching source/destination lists");
  for (size_t i = 0; i < sources.size(); ++i) {
    validateTensor(sources[i], destinations[i], "source", at::kFloat);
    validateTensor(destinations[i], sources[i], "destination", at::kFloat);
    TORCH_CHECK(sources[i].device() == destinations[i].device() &&
                    sources[i].device() == sources[0].device(),
                "GDN DMA tensors must be on the same device");
    TORCH_CHECK(sources[i].numel() > 0, "GDN DMA states must not be empty");
  }
}

void prepareGdnBackendTensors(std::vector<at::Tensor> &sources,
                              std::vector<at::Tensor> &destinations) {
  for (auto *tensors : {&sources, &destinations}) {
    for (auto &tensor : *tensors) {
      tensor = habana::eager::HbEagerTensorPool::get_backend_tensor(tensor);
      habana::get_tensor_extra_meta(tensor)->set_tensor_pipelined();
    }
  }
}

void copyGdnStates(c10d::ProcessGroupEagerHCCL *backend,
                   std::vector<at::Tensor> sources,
                   std::vector<at::Tensor> destinations) {
  validateGdnStates(sources, destinations);
  auto communicator = backend->lowLatencyCommunicator();
  TORCH_CHECK(communicator != nullptr, "HCCL communicator is not initialized");
  const auto hpu_stream = c10::hpu::getCurrentHPUStream().stream();
  if (GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE)) {
    prepareGdnBackendTensors(sources, destinations);
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        [communicator, sources = std::move(sources),
         destinations = std::move(destinations), hpu_stream]() {
          runGdnStateCopies(communicator, sources, destinations, hpu_stream);
        });
  } else {
    habana::eager::JoinPendingPipelineThreads();
    runGdnStateCopies(communicator, sources, destinations, hpu_stream);
  }
}

void copyC1PipelineTensors(c10d::ProcessGroupEagerHCCL* backend,
                           std::vector<at::Tensor> sources,
                           std::vector<at::Tensor> destinations) {
  TORCH_CHECK(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0 &&
                  GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
                  c10::hpu::getCurrentHPUStream().stream() == 0,
              "C1 PP DMA requires the eager default producer stream");
  TORCH_CHECK(sources.size() == 2 && destinations.size() == 2,
              "C1 PP DMA requires hidden and pre-mix");
  for (size_t i = 0; i < 2; ++i) {
    const auto dtype = i == 0 ? at::kBFloat16 : at::kFloat;
    const int64_t elements = i == 0 ? 20480 : 4;
    TORCH_CHECK(sources[i].device().type() == at::kHPU &&
                    sources[i].device() == destinations[i].device() &&
                    sources[i].device() == sources[0].device() &&
                    sources[i].scalar_type() == dtype && destinations[i].scalar_type() == dtype &&
                    sources[i].numel() == elements && destinations[i].numel() == elements &&
                    sources[i].is_contiguous() && destinations[i].is_contiguous(),
                "C1 PP DMA changed the contiguous hidden/pre-mix contract");
  }
  auto communicator = backend->lowLatencyCommunicator();
  TORCH_CHECK(communicator != nullptr, "C1 PP DMA requires an initialized communicator");
  prepareGdnBackendTensors(sources, destinations);
  habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
      [communicator, sources = std::move(sources), destinations = std::move(destinations)]() {
        for (const auto* values : {&sources, &destinations}) {
          for (const auto& tensor : *values) {
            const auto permutation = std::get<0>(habana_helpers::get_tensor_memory_permutation(tensor));
            TORCH_CHECK(std::is_sorted(permutation.begin(), permutation.end()),
                        "C1 PP DMA cannot consume a permuted physical tensor");
          }
        }
        runGdnStateCopies(communicator, sources, destinations, 0);
      });
}

std::shared_ptr<GdnStateDMATicket> queueGdnStateCopies(
    c10d::ProcessGroupEagerHCCL *backend, std::vector<at::Tensor> sources,
    std::vector<at::Tensor> destinations, synapse_helpers::hpuStream_t copy_stream,
    bool precise = false) {
  TORCH_CHECK(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0 &&
                  GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE),
              "Queued GDN DMA requires the eager pipeline");
  validateGdnStates(sources, destinations);
  auto communicator = backend->lowLatencyCommunicator();
  TORCH_CHECK(communicator != nullptr, "HCCL communicator is not initialized");
  const auto compute_stream = c10::hpu::getCurrentHPUStream().stream();
  prepareGdnBackendTensors(sources, destinations);
  auto ticket = std::make_shared<GdnStateDMATicket>();
  habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
      [communicator, sources = std::move(sources), destinations = std::move(destinations),
       compute_stream, copy_stream, ticket, precise]() {
        auto &device = habana::HPUDeviceContext::get_device();
        ticket->copy_stream = copy_stream;
        ticket->precise = precise;
        if (precise) {
          auto& cache = device.get_event_handle_cache();
          ticket->physical_ready = std::make_unique<synapse_helpers::CachedEventHandle>(cache);
          ticket->physical_done = std::make_unique<synapse_helpers::CachedEventHandle>(cache);
          std::vector<synapse_helpers::device_ptr> producers;
          for (const auto& source : sources)
            producers.push_back(reinterpret_cast<synapse_helpers::device_ptr>(source.storage().data_ptr().get()));
          auto& producer_stream = device.get_stream(compute_stream);
          device.add_wait_events_on_stream(producers, producer_stream);
          checkSynapse(synEventRecord(ticket->physical_ready->get(), producer_stream), "GDN physical ready record");
          g_state_precise_records.fetch_add(1, std::memory_order_relaxed);
          if (copy_stream != compute_stream) {
            checkSynapse(synStreamWaitEvent(device.get_stream(copy_stream), ticket->physical_ready->get(), 0),
                         "GDN physical copy wait");
            g_state_precise_waits.fetch_add(1, std::memory_order_relaxed);
          }
        } else {
          ticket->ready = device.create_event(false);
          ticket->done = device.create_event(false);
          device.record_event(ticket->ready, compute_stream);
          if (copy_stream != compute_stream)
            device.wait_event(ticket->ready, copy_stream);
        }
        ticket->created = true;
        // This task follows group execution in the same bridge queue. Do not
        // call HPUEvent::record or setCurrentHPUStream: both join host queues.
        runGdnStateCopies(communicator, sources, destinations, copy_stream, ticket);
      });
  return ticket;
}

void queueGdnStateWaits(std::vector<std::shared_ptr<GdnStateDMATicket>> tickets,
                       bool all_consumers = false) {
  if (tickets.empty())
    return;
  const auto compute_stream = c10::hpu::getCurrentHPUStream().stream();
  habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
      [tickets = std::move(tickets), compute_stream, all_consumers]() mutable {
        auto &device = habana::HPUDeviceContext::get_device();
        for (const auto &ticket : tickets) {
          TORCH_CHECK(ticket->created, "GDN state consumer preceded its DMA submission");
          if (ticket->precise) {
            // Normal GDN reads run on COMPUTE. Reset/prefill may copy caches
            // on a DMA substream, so those boundaries explicitly join all
            // generic substreams, once per actual completion event.
            const size_t count = all_consumers && compute_stream == 0 &&
                    GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM) ? synapse_helpers::END_TYPE_ : 1;
            for (size_t index = 0; index < count; ++index) {
              auto& consumer = device.get_stream(compute_stream,
                  static_cast<synapse_helpers::default_stream_type>(index));
              if (static_cast<synStreamHandle>(consumer) !=
                  static_cast<synStreamHandle>(device.get_stream(ticket->copy_stream))) {
                checkSynapse(synStreamWaitEvent(consumer, ticket->physical_done->get(), 0),
                             "GDN physical consumer wait");
                g_state_precise_waits.fetch_add(1, std::memory_order_relaxed);
              }
            }
          } else if (compute_stream != ticket->copy_stream) {
            device.wait_event(ticket->done, compute_stream);
          }
        }
        // Keep event handles alive until the consumer stream has reached the
        // waits. Releasing the Python ticket after enqueue is insufficient.
        const bool has_precise = std::any_of(tickets.begin(), tickets.end(),
            [](const auto& ticket) { return ticket->precise; });
        const size_t completion_streams = has_precise && all_consumers && compute_stream == 0 &&
                GET_ENV_FLAG_NEW(PT_HPU_ENABLE_GENERIC_STREAM) ? synapse_helpers::END_TYPE_ : 1;
        auto retained = std::make_shared<std::vector<std::shared_ptr<GdnStateDMATicket>>>(std::move(tickets));
        for (size_t index = 0; index < completion_streams; ++index)
          device.register_producer_on_stream({}, device.get_stream(compute_stream,
              static_cast<synapse_helpers::default_stream_type>(index)), [retained]() {});
      });
}

at::Tensor tp2ExchangePeer(c10d::ProcessGroupEagerHCCL *backend,
                           const at::Tensor &partial,
                           const at::Tensor &peer, bool reduction_only = false) {
  TORCH_CHECK(tp2ExchangeEnabled(),
              "TP2 peer exchange requires the tp2-exchange algorithm");
  TORCH_CHECK(partial.dim() >= 2, "partial must have at least two dimensions");
  TORCH_CHECK(partial.scalar_type() == at::kBFloat16,
              "TP2 peer exchange supports BF16 activations only");
  TORCH_CHECK(partial.is_contiguous(), "partial must be contiguous");
  TORCH_CHECK(partial.numel() > 0, "partial must not be empty");
  validateTensor(peer, partial, "peer", at::kBFloat16);
  TORCH_CHECK(partial.data_ptr() != peer.data_ptr(),
              "partial and peer must use distinct storage");

  auto communicator = backend->lowLatencyCommunicator();
  TORCH_CHECK(communicator != nullptr, "HCCL communicator is not initialized");
  const auto hpu_stream = c10::hpu::getCurrentHPUStream().stream();
  const bool pipeline_enabled =
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE);
  if (pipeline_enabled) {
    std::array<at::Tensor, 2> backend_tensors = {
        habana::eager::HbEagerTensorPool::get_backend_tensor(partial),
        habana::eager::HbEagerTensorPool::get_backend_tensor(peer),
    };
    for (const auto &tensor : backend_tensors) {
      habana::get_tensor_extra_meta(tensor)->set_tensor_pipelined();
    }
    habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
        [communicator, backend_tensors = std::move(backend_tensors),
         hpu_stream, reduction_only]() mutable {
          runTp2ExchangePeer(communicator, std::move(backend_tensors[0]),
                             std::move(backend_tensors[1]), hpu_stream, reduction_only);
        });
  } else {
    habana::eager::JoinPendingPipelineThreads();
    runTp2ExchangePeer(communicator, partial, peer, hpu_stream, reduction_only);
  }
  return peer;
}

void runFusedAllReduceNorm(
    const std::shared_ptr<habana::HcclCommunicator> &communicator,
    at::Tensor partial, at::Tensor residual, at::Tensor weight,
    at::Tensor reduced, at::Tensor normalized, at::Tensor residual_out,
    at::Tensor inverse_rms, float epsilon,
    synapse_helpers::hpuStream_t hpu_stream, FusedRecipe* prepared_recipe = nullptr) {
  using Clock = std::chrono::steady_clock;
  const bool profile_host = profileHostStagesEnabled();
  const auto stage_start = profile_host ? Clock::now() : Clock::time_point{};
  g_collective_launch_count.fetch_add(1, std::memory_order_relaxed);
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
  const auto lock_done = profile_host ? Clock::now() : Clock::time_point{};

  const bool exchange = tp2ExchangeEnabled();

  const hcclResult_t result = exchange
      ? resolveTp2DirectExchange()(
            reinterpret_cast<const void *>(locked.at(0)),
            reinterpret_cast<void *>(locked.at(3)),
            static_cast<size_t>(partial.numel()), hcclBfloat16, hcclSum,
            *(communicator->GetHcclHandle()), stream)
      : hcclAllReduce(reinterpret_cast<const void *>(locked.at(0)),
                      reinterpret_cast<void *>(locked.at(3)),
                      static_cast<size_t>(partial.numel()), hcclBfloat16,
                      hcclSum, *(communicator->GetHcclHandle()), stream);
  TORCH_CHECK(result == hcclSuccess, "HCCL all-reduce returned error ", result);
  const auto hccl_done = profile_host ? Clock::now() : Clock::time_point{};

  launchFusedRecipe(stream, locked.at(0),
                    {
                        locked.at(3),
                        locked.at(1),
                        locked.at(2),
                        locked.at(5),
                        locked.at(4),
                        locked.at(6),
                    },
                    static_cast<uint64_t>(partial.numel()),
                    static_cast<uint64_t>(weight.numel()), epsilon, exchange, prepared_recipe);
  const auto recipe_done = profile_host ? Clock::now() : Clock::time_point{};

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
  if (profile_host) {
    const auto producer_done = Clock::now();
    auto elapsed = [](Clock::time_point begin, Clock::time_point end) {
      return static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
              .count());
    };
    g_host_stages.calls.fetch_add(1, std::memory_order_relaxed);
    g_host_stages.lock_ns.fetch_add(elapsed(stage_start, lock_done),
                                    std::memory_order_relaxed);
    g_host_stages.hccl_ns.fetch_add(elapsed(lock_done, hccl_done),
                                    std::memory_order_relaxed);
    g_host_stages.recipe_ns.fetch_add(elapsed(hccl_done, recipe_done),
                                      std::memory_order_relaxed);
    g_host_stages.producer_ns.fetch_add(elapsed(recipe_done, producer_done),
                                        std::memory_order_relaxed);
    g_host_stages.total_ns.fetch_add(elapsed(stage_start, producer_done),
                                     std::memory_order_relaxed);
  }
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

#include "tp2_prepared_plan.h"
#include "tp2_native_decode_graph.h"
#include "tp2_native_graph_probe.h"

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
    launchFusedRecipe(stream, locked.at(0),
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
                      static_cast<float>(epsilon_), false);

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
  module.def("set_prepared_communication", [](bool enabled) { g_prepared_comm_enabled.store(enabled); });
  py::class_<PreparedGroupPlan, std::shared_ptr<PreparedGroupPlan>>(module, "PreparedGroupPlan")
      .def(py::init<>())
      .def("add_slot", [](PreparedGroupPlan& self, py::object value, bool input) {
        return self.add_slot(preparedIValue(value), input);
      })
      .def("add_compute", &PreparedGroupPlan::add_compute)
      .def("add_norm_view", &PreparedGroupPlan::add_norm_view)
      .def("add_reshape_view", &PreparedGroupPlan::add_reshape_view)
      .def("add_exchange", &PreparedGroupPlan::add_exchange)
      .def("add_peer_exchange", &PreparedGroupPlan::add_peer_exchange)
      .def("add_all_reduce", &PreparedGroupPlan::add_all_reduce)
      .def("prepare", [](PreparedGroupPlan& self, const c10::intrusive_ptr<c10d::Backend>& backend,
                          std::vector<int64_t> outputs) {
        auto* group = dynamic_cast<c10d::ProcessGroupEagerHCCL*>(backend.get());
        TORCH_CHECK(group, "Prepared plan requires ProcessGroupEagerHCCL");
        self.prepare(group, std::move(outputs));
      })
      .def("matches", [](const PreparedGroupPlan& self, py::list inputs) {
        torch::jit::Stack stack;
        for (const auto& value : inputs) stack.push_back(preparedIValue(value));
        return self.matches(stack);
      })
      .def("outputs", &PreparedGroupPlan::outputs)
      .def("invalidate", [](PreparedGroupPlan& self) { self.valid.store(false); });
  module.def("record_native_completion", &tp2_native::recordNativeCompletion);
  module.def("copy_sampled_tokens_to_host", &tp2_native::copySampledTokensToHost);
  module.def("copy_integer_record_to_host", &tp2_native::copyIntegerRecordToHost);
  module.def("copy_c1_pipeline_tensors",
      [](const c10::intrusive_ptr<c10d::Backend>& backend, std::vector<at::Tensor> sources,
         std::vector<at::Tensor> destinations) {
        auto* hccl_backend = dynamic_cast<c10d::ProcessGroupEagerHCCL*>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr, "C1 PP DMA requires ProcessGroupEagerHCCL");
        copyC1PipelineTensors(hccl_backend, std::move(sources), std::move(destinations));
      });
  py::class_<tp2_native::NativeCompletion, std::shared_ptr<tp2_native::NativeCompletion>>(
      module, "NativeCompletion")
      .def("query", &tp2_native::NativeCompletion::query, py::call_guard<py::gil_scoped_release>())
      .def("synchronize", &tp2_native::NativeCompletion::synchronize, py::call_guard<py::gil_scoped_release>());

  py::class_<tp2_native::NativeDecodeGraph, std::shared_ptr<tp2_native::NativeDecodeGraph>>(
      module, "NativeDecodeGraph")
      .def(py::init<>())
      .def("configure_topology", &tp2_native::NativeDecodeGraph::configureTopology)
      .def("capture", [](const std::shared_ptr<tp2_native::NativeDecodeGraph>& self,
                          std::vector<std::shared_ptr<PreparedGroupPlan>> plans, py::list inputs) {
        std::vector<torch::jit::Stack> stacks;
        for (const auto& row : inputs) {
          torch::jit::Stack stack;
          for (const auto& value : row.cast<py::list>()) stack.push_back(preparedIValue(value));
          stacks.push_back(std::move(stack));
        }
        self->capture(std::move(plans), std::move(stacks));
      })
      .def("instantiate", &tp2_native::NativeDecodeGraph::instantiate)
      .def("bind_dynamic_inputs", &tp2_native::NativeDecodeGraph::bindDynamicInputs)
      .def("bind_state_tensors", &tp2_native::NativeDecodeGraph::bindStateTensors)
      .def("workspace_bytes", &tp2_native::NativeDecodeGraph::workspaceBytes)
      .def("stage_fixed_inputs", &tp2_native::NativeDecodeGraph::stageFixedInputs)
      .def("replay_fixed", &tp2_native::NativeDecodeGraph::replayFixed)
      .def("replay_fixed_with_completion", &tp2_native::NativeDecodeGraph::replayFixedWithCompletion)
      .def("update_inputs", [](const std::shared_ptr<tp2_native::NativeDecodeGraph>& self,
                                std::vector<std::shared_ptr<PreparedGroupPlan>> plans, py::list inputs) {
        std::vector<torch::jit::Stack> stacks;
        for (const auto& row : inputs) {
          torch::jit::Stack stack;
          for (const auto& value : row.cast<py::list>()) stack.push_back(preparedIValue(value));
          stacks.push_back(std::move(stack));
        }
        self->updateInputs(plans, stacks);
      })
      .def("replay", &tp2_native::NativeDecodeGraph::replay)
      .def("reset_slots", &tp2_native::NativeDecodeGraph::resetSlots)
      .def("close", &tp2_native::NativeDecodeGraph::close)
      .def("state", &tp2_native::NativeDecodeGraph::state)
      .def("replay_count", &tp2_native::NativeDecodeGraph::replayCount)
      .def("segment_count", &tp2_native::NativeDecodeGraph::segmentCount)
      .def("collective_count", &tp2_native::NativeDecodeGraph::collectiveCount)
      .def("external_collective_count", &tp2_native::NativeDecodeGraph::externalCollectiveCount)
      .def("captured_command_count", &tp2_native::NativeDecodeGraph::capturedCommandCount)
      .def("captured_relocation_count", &tp2_native::NativeDecodeGraph::capturedRelocationCount)
      .def("global_program_bytes", &tp2_native::NativeDecodeGraph::globalProgramBytes)
      .def("arc_program_bytes", &tp2_native::NativeDecodeGraph::arcProgramBytes)
      .def("hcl_command_bytes_per_replay", &tp2_native::NativeDecodeGraph::hclCommandBytesPerReplay)
      .def("hcl_stream_ccb_bytes", &tp2_native::NativeDecodeGraph::hclStreamCcbBytes)
      .def("hcl_replay_bytes", &tp2_native::NativeDecodeGraph::hclReplayBytes)
      .def("hcl_ccb_wrap_count", &tp2_native::NativeDecodeGraph::hclCcbWrapCount)
      .def("hcl_submission_count", &tp2_native::NativeDecodeGraph::hclSubmissionCount)
      .def("input_update_copies", &tp2_native::NativeDecodeGraph::inputUpdateCopies)
      .def("input_update_bytes", &tp2_native::NativeDecodeGraph::inputUpdateBytes)
      .def("joint_info", &tp2_native::NativeDecodeGraph::jointInfo)
      .def("hcl_shared_stream_info", &tp2_native::NativeDecodeGraph::hclSharedStreamInfo)
      .def("retirement_info", &tp2_native::NativeDecodeGraph::retirementInfo);
  module.def("native_decode_graph_available", [] { return tp2_native::RuntimeApis::get().available(); });
  py::class_<tp2_native::NativeTp2GraphProbe, std::shared_ptr<tp2_native::NativeTp2GraphProbe>>(
      module, "NativeTp2GraphProbe")
      .def(py::init<const c10::intrusive_ptr<c10d::Backend>&, at::Tensor, at::Tensor, at::Tensor, double>())
      .def("reference", &tp2_native::NativeTp2GraphProbe::reference)
      .def("capture", &tp2_native::NativeTp2GraphProbe::capture)
      .def("replay", &tp2_native::NativeTp2GraphProbe::replay)
      .def("outputs", &tp2_native::NativeTp2GraphProbe::outputs)
      .def("info", &tp2_native::NativeTp2GraphProbe::info)
      .def("last_execution_timing", &tp2_native::NativeTp2GraphProbe::lastExecutionTiming)
      .def("joint_info", &tp2_native::NativeTp2GraphProbe::jointInfo)
      .def("synchronize", &tp2_native::NativeTp2GraphProbe::synchronize)
      .def("close", &tp2_native::NativeTp2GraphProbe::close);
  module.def("replay_prepared_groups", [](std::vector<std::shared_ptr<PreparedGroupPlan>> plans, py::list inputs) {
    std::vector<torch::jit::Stack> stacks;
    for (const auto& row : inputs) {
      torch::jit::Stack stack;
      for (const auto& value : row.cast<py::list>()) stack.push_back(preparedIValue(value));
      stacks.push_back(std::move(stack));
    }
    replayPreparedGroups(std::move(plans), std::move(stacks));
  });
  module.def("prepared_plan_counts", [] {
    return std::make_tuple(g_prepared_batches.load(), g_prepared_groups.load(),
                          g_prepared_compute_nodes.load(), g_prepared_exchange_nodes.load());
  });
  py::class_<GdnStateDMATicket, std::shared_ptr<GdnStateDMATicket>>(module, "GdnStateDMATicket");
  module.def("queue_gdn_state_waits", &queueGdnStateWaits, py::arg("tickets"), py::arg("all_consumers") = false);
  module.def("precise_gdn_event_counts", [] {
    return std::make_pair(g_state_precise_records.load(std::memory_order_relaxed),
                          g_state_precise_waits.load(std::memory_order_relaxed));
  });
  module.def("queue_gdn_state_copies_precise",
      [](const c10::intrusive_ptr<c10d::Backend>& backend, std::vector<at::Tensor> sources,
         std::vector<at::Tensor> destinations, synapse_helpers::hpuStream_t copy_stream) {
        auto* hccl_backend = dynamic_cast<c10d::ProcessGroupEagerHCCL*>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr, "GDN DMA requires ProcessGroupEagerHCCL");
        return queueGdnStateCopies(hccl_backend, std::move(sources), std::move(destinations), copy_stream, true);
      });
  module.def(
      "queue_gdn_state_copies",
      [](const c10::intrusive_ptr<c10d::Backend> &backend,
         std::vector<at::Tensor> sources, std::vector<at::Tensor> destinations,
         synapse_helpers::hpuStream_t copy_stream) {
        auto *hccl_backend = dynamic_cast<c10d::ProcessGroupEagerHCCL *>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr, "GDN DMA requires ProcessGroupEagerHCCL");
        return queueGdnStateCopies(hccl_backend, std::move(sources), std::move(destinations), copy_stream);
      },
      py::arg("backend"), py::arg("sources"), py::arg("destinations"), py::arg("copy_stream"));
  module.def("gdn_state_dma_counts", []() {
    return std::make_tuple(
        g_state_dma_batches.load(std::memory_order_relaxed),
        g_state_dma_tensors.load(std::memory_order_relaxed),
        g_state_dma_bytes.load(std::memory_order_relaxed));
  });
  module.def(
      "copy_gdn_states_current_stream",
      [](const c10::intrusive_ptr<c10d::Backend> &backend,
         std::vector<at::Tensor> sources, std::vector<at::Tensor> destinations) {
        auto *hccl_backend = dynamic_cast<c10d::ProcessGroupEagerHCCL *>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr, "GDN DMA requires ProcessGroupEagerHCCL");
        copyGdnStates(hccl_backend, std::move(sources), std::move(destinations));
      },
      py::arg("backend"), py::arg("sources"), py::arg("destinations"));
  module.def("set_use_tensor_ids", [](bool enabled) {
    g_use_tensor_ids.store(enabled, std::memory_order_relaxed);
  });
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
  module.def("reset_host_stage_debug", []() { g_host_stages.reset(); });
  module.def("host_stage_debug", []() {
    return std::make_tuple(
        g_host_stages.calls.load(std::memory_order_relaxed),
        g_host_stages.lock_ns.load(std::memory_order_relaxed),
        g_host_stages.hccl_ns.load(std::memory_order_relaxed),
        g_host_stages.recipe_ns.load(std::memory_order_relaxed),
        g_host_stages.producer_ns.load(std::memory_order_relaxed),
        g_host_stages.total_ns.load(std::memory_order_relaxed));
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
  module.def(
      "tp2_exchange_peer_current_stream",
      [](const c10::intrusive_ptr<c10d::Backend> &backend,
         const at::Tensor &partial, const at::Tensor &peer) {
        auto *hccl_backend =
            dynamic_cast<c10d::ProcessGroupEagerHCCL *>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr,
                    "TP2 peer exchange requires ProcessGroupEagerHCCL");
        return tp2ExchangePeer(hccl_backend, partial, peer);
      },
      py::arg("backend"), py::arg("partial"), py::arg("peer"));
  module.def(
      "tp2_allreduce_plain_current_stream",
      [](const c10::intrusive_ptr<c10d::Backend> &backend,
         const at::Tensor &partial, const at::Tensor &reduced) {
        auto *hccl_backend = dynamic_cast<c10d::ProcessGroupEagerHCCL *>(backend.get());
        TORCH_CHECK(hccl_backend != nullptr, "Plain TP2 reduction requires ProcessGroupEagerHCCL");
        return tp2ExchangePeer(hccl_backend, partial, reduced, true);
      }, py::arg("backend"), py::arg("partial"), py::arg("reduced"));
}
