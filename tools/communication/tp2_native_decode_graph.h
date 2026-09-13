// SPDX-License-Identifier: Apache-2.0
// Native Gaudi2 TP2 decoder replay. Included after tp2_prepared_plan.h.

#include "tp2_native_graph_topology.h"
#include "tp2_native_dependencies.h"
#include <strings.h>

namespace tp2_native {

struct SyncInfo {
  uint32_t longSoIndex = 0;
  uint64_t targetValue = 0;
};

struct ComputeGraphInfo {
  int state = 0;
  uint64_t segmentCount = 0;
  uint64_t replayCount = 0;
  uint32_t completionLongSoIndex = 0;
  uint64_t completionTarget = 0;
  uint64_t globalProgramBytes = 0;
  uint64_t arcProgramBytes = 0;
};

struct HclGraphInfo {
  int state = 0;
  uint64_t preparedReplayCount = 0;
  uint64_t nativeReplayCount = 0;
  uint64_t capturedCommandCount = 0;
  uint64_t capturedStreamCount = 0;
  uint64_t capturedRelocationCount = 0;
  uint64_t dependencyRelocationCount = 0;
  uint64_t wqeRelocationCount = 0;
  uint64_t externalCompletionAddressRelocationCount = 0;
  uint64_t internalCompletionAddressRelocationCount = 0;
  uint64_t capturedByteCount = 0;
  uint64_t maxStreamBytes = 0;
  uint64_t streamCcbBytes = 0;
  uint64_t nativeReplayBytes = 0;
  uint64_t nativeCcbWrapCount = 0;
  uint64_t nativeSubmissionCount = 0;
  uint64_t wrapCapturedCommandCount = 0;
  uint64_t completionRingSize = 0;
  uint64_t nativeWrapReplayCount = 0;
  SyncInfo completion;
};

class RuntimeApis {
 public:
  using SynGraph = void*;
  using HclGraph = void*;
  using SynCreate = synStatus (*)(SynGraph*, synStreamHandle);
  using SynBeginCapture = synStatus (*)(SynGraph);
  using SynEndCapture = synStatus (*)(SynGraph);
  using SynAbortCapture = synStatus (*)(SynGraph);
  using SynBeginReplay = synStatus (*)(SynGraph);
  using SynReplaySegment = synStatus (*)(SynGraph, uint64_t, const SyncInfo*, uint8_t, SyncInfo*);
  using SynGetInfo = synStatus (*)(SynGraph, ComputeGraphInfo*);
  using SynGetWorkspaceBytes = synStatus (*)(SynGraph, uint64_t*);
  using SynDestroy = synStatus (*)(SynGraph);
  using HclCreate = hcclResult_t (*)(const void*, void*, size_t, hcclDataType_t, hcclRedOp_t,
                                     hcclComm_t, void*, int, HclGraph*);
  using HclCapture = hcclResult_t (*)(HclGraph);
  using HclReplayPrepared = hcclResult_t (*)(HclGraph, const SyncInfo*, SyncInfo*);
  using HclStage = hcclResult_t (*)(HclGraph, const SyncInfo*, SyncInfo*);
  using HclSubmit = hcclResult_t (*)(HclGraph);
  using HclGetInfo = hcclResult_t (*)(HclGraph, HclGraphInfo*);
  using HclDestroy = hcclResult_t (*)(HclGraph);
  using HclBatch = void*;
  using HclBatchCreate = hcclResult_t (*)(const HclGraph*, size_t, HclBatch*);
  using HclBatchReplay = hcclResult_t (*)(HclBatch, const SyncInfo*, size_t, SyncInfo*);
  using HclBatchDestroy = hcclResult_t (*)(HclBatch);
  using SynExchangeCallback = int (*)(void*, const SyncInfo*, uint64_t, SyncInfo*);
  using SynPreparePlan = synStatus (*)(SynGraph, const uint32_t*, uint64_t, uint32_t, SynExchangeCallback, void*);
  using SynPreparePlanV2 = synStatus (*)(SynGraph, const uint32_t*, const uint32_t*, uint64_t, uint32_t,
                                        SynExchangeCallback, void*);
  using SynReplayPlan = synStatus (*)(SynGraph, SyncInfo*, uint64_t*, uint64_t);

  static RuntimeApis& get() {
    static RuntimeApis value;
    return value;
  }

  void require() {
    std::call_once(once_, [this]() {
      syn_create = resolve<SynCreate>("synNativeComputeGraphCreate");
      syn_begin_capture = resolve<SynBeginCapture>("synNativeComputeGraphBeginCapture");
      syn_end_capture = resolve<SynEndCapture>("synNativeComputeGraphEndCapture");
      syn_abort_capture = resolve<SynAbortCapture>("synNativeComputeGraphAbortCapture");
      syn_begin_replay = resolve<SynBeginReplay>("synNativeComputeGraphBeginReplay");
      syn_replay_segment = resolve<SynReplaySegment>("synNativeComputeGraphReplaySegment");
      syn_get_info = resolve<SynGetInfo>("synNativeComputeGraphGetInfo");
      syn_get_workspace_bytes = resolve<SynGetWorkspaceBytes>("synNativeComputeGraphGetWorkspaceBytes");
      syn_destroy = resolve<SynDestroy>("synNativeComputeGraphDestroy");
      hcl_create = resolve<HclCreate>("hcclTp2NativeGraphCreate");
      hcl_capture = resolve<HclCapture>("hcclTp2NativeGraphCapture");
      hcl_replay_prepared = resolve<HclReplayPrepared>("hcclTp2NativeGraphReplayPrepared");
      hcl_stage = resolve<HclStage>("hcclTp2NativeGraphStage");
      hcl_submit = resolve<HclSubmit>("hcclTp2NativeGraphSubmit");
      hcl_get_info = resolve<HclGetInfo>("hcclTp2NativeGraphGetInfo");
      hcl_destroy = resolve<HclDestroy>("hcclTp2NativeGraphDestroy");
      hcl_batch_create = resolve<HclBatchCreate>("hcclTp2NativeBatchCreate");
      hcl_batch_replay = resolve<HclBatchReplay>("hcclTp2NativeBatchReplay");
      hcl_batch_destroy = resolve<HclBatchDestroy>("hcclTp2NativeBatchDestroy");
      complete_ = true;
    });
    TORCH_CHECK(complete_, "Native TP2 runtime API set is incomplete; no fallback was executed");
  }

  bool available() noexcept {
    try {
      require();
      return true;
    } catch (...) {
      return false;
    }
  }

  void requirePlan() {
    require();
    std::call_once(plan_once_, [this]() {
      syn_prepare_plan = resolve<SynPreparePlan>("synNativeComputeGraphPreparePlan");
      syn_replay_plan = resolve<SynReplayPlan>("synNativeComputeGraphReplayPlan");
    });
  }

  void requirePlanV2() {
    requirePlan();
    std::call_once(plan_v2_once_, [this]() {
      syn_prepare_plan_v2 = resolve<SynPreparePlanV2>("synNativeComputeGraphPreparePlanV2");
    });
  }

  SynPreparePlanV2 syn_prepare_plan_v2 = nullptr;

  SynCreate syn_create = nullptr;
  SynBeginCapture syn_begin_capture = nullptr;
  SynEndCapture syn_end_capture = nullptr;
  SynAbortCapture syn_abort_capture = nullptr;
  SynBeginReplay syn_begin_replay = nullptr;
  SynReplaySegment syn_replay_segment = nullptr;
  SynGetInfo syn_get_info = nullptr;
  SynGetWorkspaceBytes syn_get_workspace_bytes = nullptr;
  SynDestroy syn_destroy = nullptr;
  HclCreate hcl_create = nullptr;
  HclCapture hcl_capture = nullptr;
  HclReplayPrepared hcl_replay_prepared = nullptr;
  HclStage hcl_stage = nullptr;
  HclSubmit hcl_submit = nullptr;
  HclGetInfo hcl_get_info = nullptr;
  HclDestroy hcl_destroy = nullptr;
  HclBatchCreate hcl_batch_create = nullptr;
  HclBatchReplay hcl_batch_replay = nullptr;
  HclBatchDestroy hcl_batch_destroy = nullptr;
  SynPreparePlan syn_prepare_plan = nullptr;
  SynReplayPlan syn_replay_plan = nullptr;

 private:
  template <typename Function>
  Function resolve(const char* name) {
    dlerror();
    void* symbol = dlsym(RTLD_DEFAULT, name);
    const char* error = dlerror();
    TORCH_CHECK(symbol != nullptr && error == nullptr, "Missing native TP2 runtime symbol ", name,
                error == nullptr ? "" : error);
    return reinterpret_cast<Function>(symbol);
  }

  std::once_flag once_;
  std::once_flag plan_once_;
  std::once_flag plan_v2_once_;
  bool complete_ = false;
};

inline bool jointPlanEnabled() {
  const char* value = std::getenv("VLLM_HPU_TP2_NATIVE_JOINT_PLAN");
  return value && std::strcmp(value, "1") == 0;
}

inline bool mhcOverlapEnabled() {
  const char* value = std::getenv("VLLM_HPU_DSV41_TP_MHC_OVERLAP");
  return value && (std::strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0);
}

inline int replayNicBatch(void* context, const SyncInfo* producers, uint64_t count, SyncInfo* completions) {
  return static_cast<int>(RuntimeApis::get().hcl_batch_replay(context, producers, count, completions));
}

struct FixedInputSignature {
  PreparedInputSignature logical;
  uint64_t address = 0;

  explicit FixedInputSignature(const c10::IValue& value) : logical(value) {
    if (value.isTensor()) address = reinterpret_cast<uint64_t>(value.toTensor().data_ptr());
  }

  bool matches(const c10::IValue& value) const {
    // Volatile inputs are staged into the capture allocation. The logical
    // signature still rejects a changed FP32 state destination or layout.
    return logical.matches(value);
  }
};

struct PendingInputCopy {
  at::Tensor source;
  at::Tensor destination;
  uint64_t bytes = 0;
};

class NativeCompletion {
 public:
  ~NativeCompletion() {
    if (event_ && habana::HPUDeviceContext::is_device_acquired()) synEventDestroy(event_);
  }

  // Like the PR's precise state-copy events, this observes one physical stream
  // completion, without taking Bridge's user-event mutex or joining pipelines.
  void record(synStreamHandle stream) {
    auto& device = habana::HPUDeviceContext::get_device();
    checkSynapse(synEventCreate(&event_, device.id(), 0), "native completion create");
    checkSynapse(synEventRecord(event_, stream), "native completion record");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      published_ = true;
    }
    condition_.notify_all();
  }

  void fail(std::exception_ptr error) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = error;
      published_ = true;
    }
    condition_.notify_all();
  }

  void completeHostCopy() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      host_copy_complete_ = true;
      published_ = true;
    }
    condition_.notify_all();
  }

  bool query() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) std::rethrow_exception(error_);
    if (!published_) return false;
    if (host_copy_complete_) return true;
    const auto status = synEventQuery(event_);
    if (status == synBusy) return false;
    checkSynapse(status, "native completion query");
    return true;
  }

  void synchronize() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [&] { return published_; });
      if (error_) std::rethrow_exception(error_);
      if (host_copy_complete_) return;
    }
    checkSynapse(synEventSynchronize(event_), "native completion wait");
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  synEventHandle event_ = nullptr;
  bool published_ = false;
  bool host_copy_complete_ = false;
  std::exception_ptr error_;
};

inline std::shared_ptr<NativeCompletion> recordNativeCompletion() {
  auto ticket = std::make_shared<NativeCompletion>();
  habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([ticket]() {
    habana::HPUDeviceContext::execute_thread().enqueue([ticket]() {
      try {
        ticket->record(habana::HPUDeviceContext::get_device().get_stream(0));
      } catch (...) {
        ticket->fail(std::current_exception());
        throw;
      }
    });
  });
  return ticket;
}

inline std::pair<at::Tensor, std::shared_ptr<NativeCompletion>> copyIntegerRowToHost(const at::Tensor& source, int64_t columns) {
  TORCH_CHECK(source.device().type() == at::kHPU && source.sizes() == at::IntArrayRef({1, columns}) &&
                  (source.scalar_type() == at::kInt || source.scalar_type() == at::kLong) &&
                  source.is_contiguous() && source.storage_offset() == 0,
              "Native sampled-token copy requires a contiguous C1 integer tensor");
  TORCH_CHECK(c10::hpu::getCurrentHPUStream().stream() == 0,
              "Native sampled-token copy requires the default producer stream");
  // Some Bridge configurations store logical int64 as int32 on the device.
  // The private host buffer uses that exact wire type; Python consumes integer
  // values with tolist() only after the actual copy-completion callback.
  const auto bytes = habana_helpers::GetNBytes(source);
  TORCH_CHECK((columns == 1 || columns == 4) && (bytes == 4 * columns || bytes == 8 * columns),
              "Unsupported bounded integer row wire width");
  auto host = at::empty({1, columns}, at::TensorOptions().device(at::kCPU)
                                  .dtype(bytes == 4 * columns ? at::kInt : at::kLong).pinned_memory(true));
  auto ticket = std::make_shared<NativeCompletion>();
  habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([source, host, ticket, bytes]() {
    try {
      auto backend = habana::eager::HbEagerTensorPool::get_backend_tensor(source);
      habana::HPUDeviceContext::execute_thread().enqueue([backend, host, ticket, bytes]() {
        try {
          TORCH_CHECK(backend.is_contiguous() && backend.storage_offset() == 0 &&
                          habana_helpers::GetNBytes(backend) == bytes,
                      "Sampled-token storage changed before its asynchronous copy");
          // The normal device copy helper retains producer/substream waits and
          // its address lock. FIFO lowering places this after the sampler's
          // launch without a frontend stream switch or pipeline join.
          habana::HPUDeviceContext::copy_data_to_host(
              reinterpret_cast<synapse_helpers::device_ptr>(backend.data_ptr()), host.data_ptr(),
              reinterpret_cast<synapse_helpers::device_ptr>(backend.storage().data_ptr().get()), bytes,
              [backend, host, ticket]() { ticket->completeHostCopy(); }, true, 0);
        } catch (...) {
          ticket->fail(std::current_exception());
          throw;
        }
      });
    } catch (...) {
      ticket->fail(std::current_exception());
      throw;
    }
  });
  return {host, ticket};
}

inline std::pair<at::Tensor, std::shared_ptr<NativeCompletion>> copySampledTokensToHost(const at::Tensor& source) {
  return copyIntegerRowToHost(source, 1);
}

inline std::pair<at::Tensor, std::shared_ptr<NativeCompletion>> copyIntegerRecordToHost(const at::Tensor& source) {
  return copyIntegerRowToHost(source, 4);
}

class NativeDecodeGraph : public std::enable_shared_from_this<NativeDecodeGraph> {
 public:
  enum class State { Created, Capturing, Instantiated, Invalid, Closed };

  ~NativeDecodeGraph() {
    if (state_.load() != State::Closed && state_.load() != State::Created) {
      // Normal shutdown explicitly closes after joining the pipeline. Avoid
      // issuing device work from a late static/Python destructor.
      state_.store(State::Invalid);
    }
  }

  bool matches(const std::vector<std::shared_ptr<PreparedGroupPlan>>& plans,
               const std::vector<torch::jit::Stack>& inputs) const {
    if (plans.size() != plans_.size() || inputs.size() != signatures_.size()) return false;
    for (size_t group = 0; group < plans.size(); ++group) {
      if (plans[group].get() != plans_[group].get() || inputs[group].size() != signatures_[group].size()) return false;
      for (size_t index = 0; index < inputs[group].size(); ++index)
        if (!signatures_[group][index].matches(inputs[group][index])) return false;
    }
    return state_.load() == State::Capturing || state_.load() == State::Instantiated;
  }

  void configureTopology(size_t groups, size_t collectives, bool externalPrefix) {
    TORCH_CHECK(state_.load() == State::Created && groups > 0 && collectives > 0,
                "Topology must be configured before capture");
    expected_groups_ = groups;
    expected_collectives_ = collectives;
    external_prefix_ = externalPrefix;
  }

  void capture(std::vector<std::shared_ptr<PreparedGroupPlan>> plans,
               std::vector<torch::jit::Stack> inputs) {
    RuntimeApis::get().require();
    if (mhcOverlapEnabled()) {
      TORCH_CHECK(jointPlanEnabled() && !external_prefix_ &&
                      ((expected_groups_ == 5 && (expected_collectives_ == 40 || expected_collectives_ == 42)) ||
                       (expected_groups_ == 1 && expected_collectives_ == 8)),
                  "Explicit TP dependencies require the V4.1 C1 stage topology");
      RuntimeApis::get().requirePlanV2();
    }
    TORCH_CHECK(state_.exchange(State::Capturing) == State::Created,
                "Native decoder graph capture has already started");
    validateAndRememberInputs(plans, inputs);
    plans_ = plans;
    auto self = shared_from_this();
    habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>(
        [self, plans = std::move(plans), inputs = std::move(inputs)]() mutable {
          self->scheduleCapture(std::move(plans), std::move(inputs));
        });
  }

  void instantiate() {
    habana::eager::JoinPendingPipelineThreads();
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated,
                "Native decoder graph instantiation failed; no fallback was executed");
  }

  void updateInputs(const std::vector<std::shared_ptr<PreparedGroupPlan>>& plans,
                    const std::vector<torch::jit::Stack>& inputs) {
    TORCH_CHECK(matches(plans, inputs),
                "Native decoder input address, layout, bucket, or cache generation changed; no fallback was executed");
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated,
                "Native decoder inputs can only be updated after instantiation");
    TORCH_CHECK(!fixed_inputs_ready_, "Fixed decoder inputs must use replay_fixed");
    TORCH_CHECK(input_epoch_.load() == scheduled_epoch_.load(),
                "Native decoder input update would overwrite an unreplayed token");
    pending_input_copies_.clear();
    pending_input_dependencies_.clear();
    for (size_t group = 0; group < inputs.size(); ++group) {
      for (size_t index = 0; index < inputs[group].size(); ++index) {
        if (!inputs[group][index].isTensor()) continue;
        const at::Tensor source = inputs[group][index].toTensor();
        const at::Tensor destination = signatures_[group][index].logical.value.toTensor();
        if (destination.device().type() == at::kHPU)
          pending_input_dependencies_.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
              destination.storage().data_ptr().get()));
        if (source.data_ptr() == destination.data_ptr()) continue;
        TORCH_CHECK(source.device().type() == at::kHPU && destination.device().type() == at::kHPU,
                    "Native decoder changed-address inputs must reside on HPU");
        TORCH_CHECK(source.is_contiguous() && destination.is_contiguous(),
                    "Native input staging requires contiguous tensors");
        const uint64_t bytes = static_cast<uint64_t>(source.numel()) * source.element_size();
        const uint64_t source_address = reinterpret_cast<uint64_t>(source.data_ptr());
        const uint64_t destination_address = reinterpret_cast<uint64_t>(destination.data_ptr());
        TORCH_CHECK(source_address + bytes <= destination_address || destination_address + bytes <= source_address,
                    "Native decoder input staging ranges overlap");
        pending_input_copies_.push_back(PendingInputCopy{source, destination, bytes});
      }
    }
    std::sort(pending_input_dependencies_.begin(), pending_input_dependencies_.end());
    pending_input_dependencies_.erase(
        std::unique(pending_input_dependencies_.begin(), pending_input_dependencies_.end()),
        pending_input_dependencies_.end());
    input_epoch_.fetch_add(1);
  }

  void bindDynamicInputs(std::vector<at::Tensor> tensors) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated && replay_count_.load() == 0,
                "Dynamic input bindings must be installed once after native instantiation");
    dynamic_inputs_ = std::move(tensors);
    input_dependencies_.clear();
    for (const auto& tensor : dynamic_inputs_) {
      TORCH_CHECK(tensor.device().type() == at::kHPU, "Native graph inputs must reside on HPU");
      input_dependencies_.push_back(
          reinterpret_cast<synapse_helpers::device_ptr>(tensor.storage().data_ptr().get()));
    }
    std::sort(input_dependencies_.begin(), input_dependencies_.end());
    input_dependencies_.erase(std::unique(input_dependencies_.begin(), input_dependencies_.end()),
                              input_dependencies_.end());
    prepareCompletionAddresses();
    fixed_inputs_ready_ = true;
  }

  void stageFixedInputs(const std::vector<at::Tensor>& sources,
                        const std::vector<at::Tensor>& destinations) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated && fixed_inputs_ready_,
                "Fixed input staging requires an instantiated native plan");
    TORCH_CHECK(sources.size() == destinations.size() && pending_input_copies_.empty(),
                "Fixed input staging count mismatch or an unreplayed update");
    std::vector<PendingInputCopy> copies;
    for (size_t index = 0; index < sources.size(); ++index) {
      const auto& source = sources[index];
      const auto& destination = destinations[index];
      const bool bound = std::any_of(dynamic_inputs_.begin(), dynamic_inputs_.end(), [&](const at::Tensor& tensor) {
        return tensor.unsafeGetTensorImpl() == destination.unsafeGetTensorImpl();
      });
      TORCH_CHECK(bound && source.device().type() == at::kHPU && source.device() == destination.device() &&
                      source.scalar_type() == destination.scalar_type() && source.sizes() == destination.sizes() &&
                      source.is_contiguous() && destination.is_contiguous(),
                  "Fixed input staging changed a bound tensor contract");
      copies.push_back({source, destination, source.numel() * source.element_size()});
    }
    for (auto& copy : copies) {
      copy.source = habana::eager::HbEagerTensorPool::get_backend_tensor(copy.source);
      copy.destination = habana::eager::HbEagerTensorPool::get_backend_tensor(copy.destination);
      habana::get_tensor_extra_meta(copy.source)->set_tensor_pipelined();
      habana::get_tensor_extra_meta(copy.destination)->set_tensor_pipelined();
    }
    pending_input_copies_ = std::move(copies);
  }

  void bindStateTensors(std::vector<at::Tensor> tensors) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated && fixed_inputs_ready_ &&
                    replay_count_.load() == 0 && state_tensors_.empty(),
                "State allocations must be bound once before the first fixed replay");
    TORCH_CHECK(!tensors.empty(), "Explicit native state bindings cannot be empty");
    for (const auto& tensor : tensors) {
      TORCH_CHECK(tensor.device().type() == at::kHPU,
                  "Native state allocations must reside on HPU");
      const auto address = reinterpret_cast<synapse_helpers::device_ptr>(tensor.storage().data_ptr().get());
      bool captured = false;
      for (const auto& frame : frames_)
        for (const auto& value : frame->values)
          if (value.isTensor() && value.toTensor().device().type() == at::kHPU &&
              value.toTensor().storage().data_ptr().get() == tensor.storage().data_ptr().get())
            captured = true;
      TORCH_CHECK(captured, "Native state allocation is absent from the captured tensor program");
      input_dependencies_.push_back(address);
    }
    state_tensors_ = std::move(tensors);
    std::sort(input_dependencies_.begin(), input_dependencies_.end());
    input_dependencies_.erase(std::unique(input_dependencies_.begin(), input_dependencies_.end()),
                              input_dependencies_.end());
    // Mutated state is both a consumer of preceding prefill/reset writes and
    // a producer for the next request. Keep the allocation, not a copy of it.
    prepareCompletionAddresses();
  }

  void replayFixed() {
    scheduleFixed(nullptr);
  }

  std::shared_ptr<NativeCompletion> replayFixedWithCompletion() {
    auto ticket = std::make_shared<NativeCompletion>();
    scheduleFixed(ticket);
    return ticket;
  }

  void scheduleFixed(const std::shared_ptr<NativeCompletion>& ticket) {
    RECORD_FUNCTION("vllm_gaudi::native_decoder_enqueue", std::vector<c10::IValue>());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      TORCH_CHECK(state_.load() == State::Instantiated && fixed_inputs_ready_,
                  "Native fixed replay requires prepared input ownership");
      const uint64_t epoch = input_epoch_.fetch_add(1) + 1;
      TORCH_CHECK(epoch == scheduled_epoch_.load() + 1, "Native fixed replay input epoch changed");
    }
    replayImpl(ticket);
  }

  void replay() {
    replayImpl(nullptr);
  }

  void replayImpl(const std::shared_ptr<NativeCompletion>& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated,
                "Native decoder graph is not instantiated; no fallback was executed");
    const uint64_t epoch = input_epoch_.load();
    TORCH_CHECK(!ticket || prefix_node_count_ == 0,
                "Native completion tickets require a decoder without an external prefix");
    TORCH_CHECK(epoch == scheduled_epoch_.load() + 1,
                "Native decoder replay requires exactly one preceding input update");
    scheduled_epoch_.store(epoch);
    auto copies = std::move(pending_input_copies_);
    auto dependencies = std::move(pending_input_dependencies_);
    auto self = shared_from_this();
    if (ticket) {
      habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>(
          [self, epoch, ticket, copies = std::move(copies), dependencies = std::move(dependencies)]() mutable {
        habana::HPUDeviceContext::execute_thread().enqueue(
            [self, epoch, ticket, copies = std::move(copies), dependencies = std::move(dependencies)]() {
          try {
            if (!copies.empty()) self->stageInputCopiesOnExecute(copies);
            self->prepareInputDependenciesOnExecute(dependencies);
            self->replayOnExecute(epoch);
            ticket->record(habana::HPUDeviceContext::get_device().get_stream(0));
          } catch (...) {
            self->state_.store(State::Invalid);
            ticket->fail(std::current_exception());
            throw;
          }
        });
      });
      return;
    }
    habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>(
        [self, epoch, copies = std::move(copies), dependencies = std::move(dependencies)]() mutable {
      if (!copies.empty())
        habana::HPUDeviceContext::execute_thread().enqueue(
            [self, copies = std::move(copies)]() { self->stageInputCopiesOnExecute(copies); });
      habana::HPUDeviceContext::execute_thread().enqueue(
          [self, dependencies = std::move(dependencies)]() { self->prepareInputDependenciesOnExecute(dependencies); });
      self->scheduleExternalPrefix();
      habana::HPUDeviceContext::execute_thread().enqueue(
          [self, epoch]() { self->replayOnExecute(epoch); });
    });
  }

  void resetSlots() {
    habana::eager::JoinPendingPipelineThreads();
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated,
                "Native decoder slots can only be reset for an instantiated graph");
    auto& device = habana::HPUDeviceContext::get_device();
    checkSynapse(synStreamSynchronize(device.get_stream(0)), "synStreamSynchronize(native decoder reset)");
  }

  void close() {
    habana::eager::JoinPendingPipelineThreads();
    std::lock_guard<std::mutex> lock(mutex_);
    State state = state_.load();
    if (state == State::Closed || state == State::Created) {
      state_.store(State::Closed);
      return;
    }
    auto& api = RuntimeApis::get();
    synStatus first_syn = synSuccess;
    hcclResult_t first_hcl = hcclSuccess;
    if (hcl_batch_ != nullptr) {
      checkSynapse(synStreamSynchronize(habana::HPUDeviceContext::get_device().get_stream(0)),
                   "synStreamSynchronize(native plan close)");
      if (!hcl_graphs_.empty()) {
        HclGraphInfo info;
        TORCH_CHECK(api.hcl_get_info(hcl_graphs_.back(), &info) == hcclSuccess,
                    "Native graph retirement counters are unavailable");
        retirement_statistics_ = {info.nativeReplayCount, info.nativeReplayBytes, info.nativeCcbWrapCount,
            info.nativeSubmissionCount, info.completionRingSize, info.nativeWrapReplayCount,
            info.completion.longSoIndex, info.completion.targetValue};
      }
      first_hcl = api.hcl_batch_destroy(hcl_batch_);
      hcl_batch_ = nullptr;
    }
    for (auto graph : hcl_graphs_) {
      if (graph != nullptr) {
        const hcclResult_t status = api.hcl_destroy(graph);
        if (first_hcl == hcclSuccess && status != hcclSuccess) first_hcl = status;
      }
    }
    hcl_graphs_.clear();
    if (syn_graph_ != nullptr) {
      ComputeGraphInfo info;
      if (api.syn_get_info(syn_graph_, &info) == synSuccess && info.state == 1)
        api.syn_abort_capture(syn_graph_);
      first_syn = api.syn_destroy(syn_graph_);
      syn_graph_ = nullptr;
    }
    resources_.clear();
    frames_.clear();
    // Producer events may retain this graph until their storage is retired.
    // A closed graph must release the opposite side of that ownership chain
    // after device completion, including its communicator and input aliases.
    plans_.clear();
    signatures_.clear();
    dynamic_inputs_.clear();
    state_tensors_.clear();
    pending_input_copies_.clear();
    pending_input_dependencies_.clear();
    input_dependencies_.clear();
    completion_addresses_.clear();
    fixed_inputs_ready_ = false;
    communicator_.reset();
    state_.store(first_syn == synSuccess && first_hcl == hcclSuccess ? State::Closed : State::Invalid);
    TORCH_CHECK(first_hcl == hcclSuccess && first_syn == synSuccess,
                "Native decoder graph close failed: HCL=", first_hcl, " Synapse=", first_syn);
  }

  uint64_t replayCount() const { return replay_count_.load(); }
  uint64_t segmentCount() const { return segment_count_.load(); }
  uint64_t collectiveCount() const { return collective_count_.load(); }
  uint64_t externalCollectiveCount() const { return external_collective_count_.load(); }
  uint64_t capturedCommandCount() const { return captured_command_count_.load(); }
  uint64_t capturedRelocationCount() const { return captured_relocation_count_.load(); }
  uint64_t globalProgramBytes() const { return global_program_bytes_.load(); }
  uint64_t arcProgramBytes() const { return arc_program_bytes_.load(); }
  uint64_t workspaceBytes() const { return workspace_bytes_.load(); }
  uint64_t hclCommandBytesPerReplay() const { return hcl_command_bytes_per_replay_.load(); }
  uint64_t hclStreamCcbBytes() const { return hcl_stream_ccb_bytes_.load(); }
  uint64_t hclReplayBytes() const { return hcl_replay_bytes_.load(); }
  uint64_t hclCcbWrapCount() const { return hcl_ccb_wrap_count_.load(); }
  uint64_t hclSubmissionCount() const { return hcl_submission_count_.load(); }
  int state() const { return static_cast<int>(state_.load()); }
  uint64_t inputUpdateCopies() const { return input_update_copies_.load(); }
  uint64_t inputUpdateBytes() const { return input_update_bytes_.load(); }
  std::array<uint64_t, 12> jointInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return joint_statistics_;
  }

  std::array<uint64_t, 3> hclSharedStreamInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (hcl_graphs_.empty()) return {};
    HclGraphInfo info;
    TORCH_CHECK(RuntimeApis::get().hcl_get_info(hcl_graphs_.back(), &info) == hcclSuccess,
                "hcclTp2NativeGraphGetInfo failed during stream statistics snapshot");
    // These live stream counters include any other users of the same streams.
    // Do not sum snapshots from different graphs that share those streams.
    return {info.nativeReplayBytes, info.nativeCcbWrapCount, info.nativeSubmissionCount};
  }

  std::array<uint64_t, 8> retirementInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Closed, "Retirement counters require completed graph close");
    return retirement_statistics_;
  }

 private:
  size_t expected_groups_ = 0;
  size_t expected_collectives_ = 0;
  bool external_prefix_ = false;

  void validateAndRememberInputs(const std::vector<std::shared_ptr<PreparedGroupPlan>>& plans,
                                 const std::vector<torch::jit::Stack>& inputs) {
    TORCH_CHECK((expected_groups_ ? plans.size() == expected_groups_ : (plans.size() == 1 || plans.size() == 8))
                    && plans.size() == inputs.size(),
                "Native decoder requires one qualification group or eight full decoder groups");
    signatures_.clear();
    signatures_.reserve(inputs.size());
    std::shared_ptr<habana::HcclCommunicator> communicator;
    for (size_t group = 0; group < plans.size(); ++group) {
      TORCH_CHECK(plans[group] && plans[group]->matches(inputs[group]),
                  "Native decoder capture requires a sealed fixed-shape prepared group");
      TORCH_CHECK(plans[group]->communicator, "Native decoder capture requires an initialized communicator");
      if (!communicator) communicator = plans[group]->communicator;
      TORCH_CHECK(plans[group]->communicator == communicator,
                  "Native decoder groups must share one communicator and HCL stream");
      std::vector<FixedInputSignature> row;
      row.reserve(inputs[group].size());
      for (const auto& value : inputs[group]) row.emplace_back(value);
      signatures_.push_back(std::move(row));
    }
    communicator_ = std::move(communicator);
  }

  static std::vector<std::shared_ptr<PreparedFrame>> buildFrames(
      const std::vector<std::shared_ptr<PreparedGroupPlan>>& plans,
      std::vector<torch::jit::Stack> inputs) {
    std::vector<std::shared_ptr<PreparedFrame>> frames;
    frames.reserve(plans.size());
    for (size_t group = 0; group < plans.size(); ++group) {
      auto held = std::make_shared<PreparedFrame>(PreparedFrame{plans[group], plans[group]->slots});
      auto incoming = std::move(inputs[group]);
      for (size_t index = 0; index < incoming.size(); ++index)
        held->values[plans[group]->input_slots[index]] = std::move(incoming[index]);
      plans[group]->refresh_reshape_views(held->values);
      for (const auto& value : held->values)
        if (value.isTensor() && value.toTensor().device().type() == at::kHPU)
          habana::get_tensor_extra_meta(value.toTensor())->set_tensor_pipelined();
      frames.push_back(std::move(held));
    }
    return frames;
  }

  void scheduleOrdinaryNode(const std::shared_ptr<PreparedFrame>& frame, const PreparedNode& node) {
    if (!node.exchange) {
      torch::jit::Stack input;
      std::vector<at::Tensor> output;
      for (auto index : node.inputs) input.push_back(frame->values.at(index));
      for (auto index : node.outputs)
        output.push_back(habana::eager::HbEagerTensorPool::get_backend_tensor(
            frame->values.at(index).toTensor()));
      auto backend_input = habana::eager::convert_ivalues_to_backend_tensors(input);
      habana::graph::GraphExec::LaunchRecipeTask(
          node.graph, std::move(backend_input), std::move(output), {});
      return;
    }

    torch::jit::Stack bindings;
    for (auto index : node.inputs) bindings.push_back(frame->values.at(index));
    for (auto index : node.outputs) bindings.push_back(frame->values.at(index));
    auto backend = habana::eager::convert_ivalues_to_backend_tensors(bindings);
    habana::HPUDeviceContext::execute_thread().enqueue(
        [frame, node, values = std::move(backend)]() mutable {
          runPreparedExchangeNode(frame->plan->communicator, node, values);
        });
  }

  void scheduleExternalPrefix() {
    size_t ordinal = 0;
    for (const auto& frame : frames_)
      for (const PreparedNode& node : frame->plan->nodes) {
        if (ordinal++ >= prefix_node_count_) return;
        scheduleOrdinaryNode(frame, node);
      }
    TORCH_CHECK(ordinal == prefix_node_count_, "Native decoder external prefix topology changed");
  }

  void scheduleCapture(std::vector<std::shared_ptr<PreparedGroupPlan>> plans,
                       std::vector<torch::jit::Stack> inputs) {
    frames_ = buildFrames(plans, std::move(inputs));
    segment_count_.store(0);
    std::vector<NativeNodeKind> node_kinds;
    for (const auto& plan : plans)
      for (const auto& node : plan->nodes) {
        TORCH_CHECK(!node.peer_only || jointPlanEnabled(), "Exchange-only capture requires the unified command plan");
        node_kinds.push_back({node.exchange, node.peer_only});
      }
    const auto topology = NativeGraphTopology::prepare(node_kinds, plans.size(), jointPlanEnabled(),
                                                      expected_collectives_, external_prefix_);
    prefix_node_count_ = topology.prefixNodes;
    segment_count_.store(topology.computeCount);
    collective_count_.store(topology.consumers.size());
    external_collective_count_.store(topology.externalCollectives);
    prepared_consumers_ = topology.consumers;
    if (mhcOverlapEnabled()) {
      std::vector<NativeDependencyNode> bindings;
      for (const auto& frame : frames_)
        for (const auto& node : frame->plan->nodes) {
          TORCH_CHECK(!node.exchange || node.peer_only, "V4.1 overlap requires ordinary peer exchanges");
          NativeDependencyNode binding;
          binding.exchange = node.exchange;
          auto ranges = [&](const std::vector<int64_t>& slots) {
            std::vector<NativeBufferRange> result;
            for (auto slot : slots) {
              const auto& value = frame->values.at(slot);
              if (!value.isTensor()) continue;
              const auto tensor = value.toTensor();
              if (tensor.device().type() != c10::DeviceType::HPU || !tensor.numel()) continue;
              uint64_t elements = 1;
              for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
                TORCH_CHECK(tensor.stride(dim) >= 0, "Explicit TP binding has a negative stride");
                elements += (tensor.size(dim) - 1) * tensor.stride(dim);
              }
              result.push_back({reinterpret_cast<uint64_t>(tensor.data_ptr()), elements * tensor.element_size()});
            }
            return result;
          };
          binding.inputs = ranges(node.inputs);
          binding.outputs = ranges(node.outputs);
          bindings.push_back(std::move(binding));
        }
      prepared_dependencies_ = prepareNativeDependencies(bindings);
      TORCH_CHECK(prepared_dependencies_.size() == topology.consumers.size(),
                  "Explicit TP dependency coverage differs");
      prepared_consumers_.clear();
      size_t overlapped = 0;
      for (size_t index = 0; index < prepared_dependencies_.size(); ++index) {
        const auto& dep = prepared_dependencies_[index];
        prepared_producers_.push_back(dep.producer);
        prepared_consumers_.push_back(dep.consumer);
        overlapped += dep.producer != UINT32_MAX && dep.consumer > dep.producer + 1;
        std::fprintf(stderr, "NATIVE_TP_DEPENDENCY i=%zu producer=%u consumer=%u last_consumer=%u bytes=%llu\n",
                     index, dep.producer, dep.consumer, dep.lastConsumer,
                     static_cast<unsigned long long>(dep.input.bytes));
      }
      TORCH_CHECK(overlapped > 0, "V4.1 overlap graph has no independent compute between TP producer and consumer");
    }
    hcl_graphs_.assign(topology.consumers.size(), nullptr);

    auto self = shared_from_this();
    scheduleExternalPrefix();
    habana::HPUDeviceContext::execute_thread().enqueue([self]() { self->beginCaptureOnExecute(); });
    size_t collective_index = 0;
    size_t ordinal = 0;
    for (const auto& frame : frames_) {
      for (const PreparedNode& node : frame->plan->nodes) {
        if (ordinal++ < prefix_node_count_) continue;
        if (!node.exchange) {
          torch::jit::Stack input;
          std::vector<at::Tensor> output;
          for (auto index : node.inputs) input.push_back(frame->values.at(index));
          for (auto index : node.outputs)
            output.push_back(habana::eager::HbEagerTensorPool::get_backend_tensor(
                frame->values.at(index).toTensor()));
          auto backend_input = habana::eager::convert_ivalues_to_backend_tensors(input);
          habana::graph::GraphExec::LaunchRecipeTask(
              node.graph, std::move(backend_input), std::move(output), {});
        } else {
          torch::jit::Stack bindings;
          for (auto index : node.inputs) bindings.push_back(frame->values.at(index));
          for (auto index : node.outputs) bindings.push_back(frame->values.at(index));
          auto backend = habana::eager::convert_ivalues_to_backend_tensors(bindings);
          const size_t index = collective_index++;
          habana::HPUDeviceContext::execute_thread().enqueue(
              [self, node, index, values = std::move(backend)]() mutable {
                self->captureExchangeOnExecute(index, node, std::move(values));
              });
        }
      }
    }
    habana::HPUDeviceContext::execute_thread().enqueue([self]() { self->endCaptureOnExecute(); });
  }

  void beginCaptureOnExecute() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Capturing, "Native decoder capture state changed before execution");
    auto& api = RuntimeApis::get();
    auto& device = habana::HPUDeviceContext::get_device();
    synStreamHandle stream = device.get_stream(0);
    checkSynapse(api.syn_create(&syn_graph_, stream), "synNativeComputeGraphCreate");
    checkSynapse(api.syn_begin_capture(syn_graph_), "synNativeComputeGraphBeginCapture");
    syn_capture_active_ = true;
  }

  void captureExchangeOnExecute(size_t index, const PreparedNode& node, torch::jit::Stack values) {
    try {
      TORCH_CHECK(state_.load() == State::Capturing, "Native decoder capture was invalidated");
      TORCH_CHECK(!node.reduction_only,
                  "Native plain AllReduce requires peer transfer and a compiled BF16 sum");
      TORCH_CHECK(index < hcl_graphs_.size() &&
                      (node.peer_only ? values.size() == 2 : values.size() == 7 && node.communication_recipe),
                  "Invalid native TP2 exchange capture binding");
      std::vector<at::Tensor> tensors(values.size());
      for (size_t i = 0; i < tensors.size(); ++i) tensors[i] = values[i].toTensor();
      auto holder = std::make_shared<GenericResourceHolder>();
      std::vector<void*> addresses;
      for (const auto& tensor : tensors) {
        holder->add_tensor(tensor);
        addresses.push_back(tensor.data_ptr());
      }
      auto context = communicator_->getDeviceCtxt();
      context->lock_address(addresses, holder->get_address_lock());
      const auto& locked = *holder->get_address_lock();
      auto& device = habana::HPUDeviceContext::get_device();
      synStreamHandle stream = device.get_stream(0);
      auto& api = RuntimeApis::get();
      const int exchange_mode = node.reduction_only ? 0 : 1;
      const size_t peer_index = node.peer_only ? 1 : 3;
      const hcclResult_t create = api.hcl_create(
          reinterpret_cast<const void*>(locked.at(0)), reinterpret_cast<void*>(locked.at(peer_index)),
          static_cast<size_t>(tensors[0].numel()), hcclBfloat16, hcclSum,
          *(communicator_->GetHcclHandle()), stream, exchange_mode, &hcl_graphs_[index]);
      TORCH_CHECK(create == hcclSuccess, "hcclTp2NativeGraphCreate failed: ", create);
      const hcclResult_t capture = api.hcl_capture(hcl_graphs_[index]);
      TORCH_CHECK(capture == hcclSuccess, "hcclTp2NativeGraphCapture failed: ", capture);
      if (!node.peer_only) launchFusedRecipe(stream, locked.at(0),
                        {locked.at(3), locked.at(1), locked.at(2), locked.at(5), locked.at(4), locked.at(6)},
                        static_cast<uint64_t>(tensors[0].numel()),
                        static_cast<uint64_t>(tensors[2].numel()), node.epsilon, true,
                        node.communication_recipe);
      resources_.push_back(std::move(holder));
    } catch (...) {
      state_.store(State::Invalid);
      if (syn_graph_ != nullptr && syn_capture_active_) {
        RuntimeApis::get().syn_abort_capture(syn_graph_);
        syn_capture_active_ = false;
      }
      throw;
    }
  }

  void endCaptureOnExecute() {
    try {
      auto& api = RuntimeApis::get();
      checkSynapse(api.syn_end_capture(syn_graph_), "synNativeComputeGraphEndCapture");
      syn_capture_active_ = false;
      ComputeGraphInfo info;
      checkSynapse(api.syn_get_info(syn_graph_, &info), "synNativeComputeGraphGetInfo(capture)");
      TORCH_CHECK(info.state == 2 && info.segmentCount == segment_count_.load(),
                  "Native compute capture mismatch: state=", info.state, " segments=", info.segmentCount,
                  " expected=", segment_count_.load());
      TORCH_CHECK(info.globalProgramBytes > 0 && info.arcProgramBytes > 0,
                  "Native compute graph did not retain program memory");
      global_program_bytes_.store(info.globalProgramBytes);
      arc_program_bytes_.store(info.arcProgramBytes);
      uint64_t workspaceBytes = 0;
      checkSynapse(api.syn_get_workspace_bytes(syn_graph_, &workspaceBytes),
                   "synNativeComputeGraphGetWorkspaceBytes(capture)");
      workspace_bytes_.store(workspaceBytes);
      uint64_t commands = 0;
      uint64_t relocations = 0;
      uint64_t hcl_command_bytes = 0;
      uint64_t hcl_ccb_bytes = 0;
      for (auto graph : hcl_graphs_) {
        TORCH_CHECK(graph != nullptr, "Native HCL capture is incomplete");
        HclGraphInfo hcl_info;
        TORCH_CHECK(api.hcl_get_info(graph, &hcl_info) == hcclSuccess,
                    "hcclTp2NativeGraphGetInfo failed during instantiation");
        TORCH_CHECK(hcl_info.state == 2 && hcl_info.capturedCommandCount > 0 &&
                        hcl_info.capturedStreamCount > 0 && hcl_info.dependencyRelocationCount >= 4 &&
                        hcl_info.wqeRelocationCount > 0 &&
                        hcl_info.externalCompletionAddressRelocationCount > 0 &&
                        hcl_info.capturedByteCount > 0 &&
                        hcl_info.maxStreamBytes > 0 && hcl_info.streamCcbBytes > 0,
                    "Native HCL graph lacks executable dependency/WQE/completion relocation metadata");
        commands += hcl_info.capturedCommandCount;
        relocations += hcl_info.capturedRelocationCount;
        hcl_command_bytes += hcl_info.capturedByteCount;
        if (hcl_ccb_bytes == 0) hcl_ccb_bytes = hcl_info.streamCcbBytes;
        TORCH_CHECK(hcl_ccb_bytes == hcl_info.streamCcbBytes,
                    "Native HCL graphs use inconsistent stream CCB capacities");
      }
      captured_command_count_.store(commands);
      captured_relocation_count_.store(relocations);
      hcl_command_bytes_per_replay_.store(hcl_command_bytes);
      hcl_stream_ccb_bytes_.store(hcl_ccb_bytes);
      if (jointPlanEnabled()) {
        api.requirePlan();
        TORCH_CHECK(prepared_consumers_.size() == hcl_graphs_.size(),
                    "Native joint plan dependency coverage mismatch");
        checkSynapse(synStreamSynchronize(habana::HPUDeviceContext::get_device().get_stream(0)),
                     "synStreamSynchronize(native plan instantiate)");
        TORCH_CHECK(api.hcl_batch_create(hcl_graphs_.data(), hcl_graphs_.size(), &hcl_batch_) == hcclSuccess,
                    "hcclTp2NativeBatchCreate(decoder) failed");
        HclGraphInfo last;
        TORCH_CHECK(api.hcl_get_info(hcl_graphs_.back(), &last) == hcclSuccess, "HCL batch completion unavailable");
        if (mhcOverlapEnabled()) {
          api.requirePlanV2();
          checkSynapse(api.syn_prepare_plan_v2(syn_graph_, prepared_producers_.data(), prepared_consumers_.data(),
                       prepared_consumers_.size(), last.completion.longSoIndex, replayNicBatch, hcl_batch_),
                       "synNativeComputeGraphPreparePlanV2(decoder)");
        } else {
          checkSynapse(api.syn_prepare_plan(syn_graph_, prepared_consumers_.data(), prepared_consumers_.size(),
                       last.completion.longSoIndex, replayNicBatch, hcl_batch_),
                       "synNativeComputeGraphPreparePlan(decoder)");
        }
      }
      state_.store(State::Instantiated);
      prepareCompletionAddresses();
      registerOutputs();
    } catch (...) {
      state_.store(State::Invalid);
      if (syn_graph_ != nullptr && syn_capture_active_) {
        RuntimeApis::get().syn_abort_capture(syn_graph_);
        syn_capture_active_ = false;
      }
      throw;
    }
  }

  void stageInputCopiesOnExecute(const std::vector<PendingInputCopy>& copies) {
    auto address = [](const at::Tensor& tensor) {
      return reinterpret_cast<uint64_t>(tensor.storage().data_ptr().get()) +
          tensor.storage_offset() * tensor.element_size();
    };
    auto overlaps = [](uint64_t left, uint64_t leftBytes, uint64_t right, uint64_t rightBytes) {
      return left < right + rightBytes && right < left + leftBytes;
    };
    // Validate the whole transaction after prior producers allocate their
    // storage, and before any DMA can overwrite another staged source.
    for (size_t i = 0; i < copies.size(); ++i) {
      for (size_t j = 0; j < copies.size(); ++j) {
        TORCH_CHECK(!overlaps(address(copies[i].destination), copies[i].bytes,
                              address(copies[j].source), copies[j].bytes),
                    "Native input copy destination overlaps a source");
        if (i < j)
          TORCH_CHECK(!overlaps(address(copies[i].destination), copies[i].bytes,
                                address(copies[j].destination), copies[j].bytes),
                      "Native input copy destinations overlap");
      }
    }
    std::vector<synapse_helpers::device_ptr> destinations;
    destinations.reserve(copies.size());
    for (const auto& copy : copies) {
      destinations.push_back(reinterpret_cast<synapse_helpers::device_ptr>(
          copy.destination.storage().data_ptr().get()));
    }
    auto& device = habana::HPUDeviceContext::get_device();
    // Input storage is also an output of the previous replay's lifetime.
    // The generic D2D helper waits for its source, but does not wait for the
    // destination's old consumer. Protect every destination before the first
    // copy: a ready position upload can otherwise run before the next hidden
    // input's producer and overwrite a still-running decoder's input.
    device.add_wait_events_on_stream(
        destinations, device.get_stream(0, synapse_helpers::default_stream_type::DMA_D2D));
    for (const auto& copy : copies) {
      // This helper retains both allocations and establishes DMA producer
      // dependencies. Full decoder steady replay uses its fixed input table.
      habana_helpers::copy_data_within_device(copy.source, copy.destination, true);
      input_update_copies_.fetch_add(1);
      input_update_bytes_.fetch_add(copy.bytes);
    }
  }

  void prepareInputDependenciesOnExecute(const std::vector<synapse_helpers::device_ptr>& staged_dependencies) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated, "Native graph input preparation after invalidation");
    auto& device = habana::HPUDeviceContext::get_device();
    // The external embedding exchange also reads changing input storage. Its
    // ready dependency must precede that exchange, including when the prefix
    // contains no GraphExec launch to prepare input dependencies for us.
    device.add_wait_events_on_stream(input_dependencies_, device.get_stream(0));
    if (!staged_dependencies.empty())
      device.add_wait_events_on_stream(staged_dependencies, device.get_stream(0));
  }

  void replayOnExecute(uint64_t epoch) {
    RECORD_FUNCTION("vllm_gaudi::native_decoder_publish", std::vector<c10::IValue>());
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_.load() == State::Instantiated,
                "Native decoder graph is not instantiated; no fallback was executed");
    try {
      auto& api = RuntimeApis::get();
      auto& device = habana::HPUDeviceContext::get_device();
      if (hcl_batch_ != nullptr) {
        SyncInfo completion;
        checkSynapse(api.syn_replay_plan(syn_graph_, &completion, joint_statistics_.data(), joint_statistics_.size()),
                     "synNativeComputeGraphReplayPlan(decoder)");
      } else {
      checkSynapse(api.syn_begin_replay(syn_graph_), "synNativeComputeGraphBeginReplay");
      SyncInfo compute_completion;
      SyncInfo collective_completion;
      size_t segment = 0;
      size_t collective = 0;
      size_t ordinal = 0;
      for (const auto& frame : frames_) {
        for (const PreparedNode& node : frame->plan->nodes) {
          if (ordinal++ < prefix_node_count_) continue;
          if (node.exchange) {
            TORCH_CHECK(compute_completion.targetValue != 0 && collective < hcl_graphs_.size(),
                        "Native collective has no preceding compute completion");
            const hcclResult_t status =
                api.hcl_stage(hcl_graphs_[collective++], &compute_completion, &collective_completion);
            TORCH_CHECK(status == hcclSuccess, "hcclTp2NativeGraphStage failed: ", status);
          }
          SyncInfo next_compute;
          const bool submit = segment + 1 == segment_count_.load();
          const SyncInfo* dependency = node.exchange ? &collective_completion : nullptr;
          checkSynapse(api.syn_replay_segment(syn_graph_, segment, dependency, submit, &next_compute),
                       "synNativeComputeGraphReplaySegment");
          compute_completion = next_compute;
          ++segment;
        }
      }
      TORCH_CHECK(segment == segment_count_.load() && collective == hcl_graphs_.size(),
                  "Native decoder replay topology changed");
      TORCH_CHECK(api.hcl_submit(hcl_graphs_.back()) == hcclSuccess, "hcclTp2NativeGraphSubmit failed");
      }
      if (replay_count_.load() == 0) {
        HclGraphInfo replay_info;
        TORCH_CHECK(api.hcl_get_info(hcl_graphs_.back(), &replay_info) == hcclSuccess,
                    "hcclTp2NativeGraphGetInfo failed after first unified replay");
        TORCH_CHECK(replay_info.nativeReplayBytes > 0 && replay_info.nativeSubmissionCount > 0,
                    "Native HCL replay did not advance real stream PI/submission counters");
        hcl_replay_bytes_.store(replay_info.nativeReplayBytes);
        hcl_ccb_wrap_count_.store(replay_info.nativeCcbWrapCount);
        hcl_submission_count_.store(replay_info.nativeSubmissionCount);
      }
      TORCH_CHECK(epoch == executed_epoch_.load() + 1, "Native decoder replay epoch is out of order");
      executed_epoch_.store(epoch);
      ++replay_count_;
      registerOutputs();
    } catch (...) {
      state_.store(State::Invalid);
      throw;
    }
  }

  void prepareCompletionAddresses() {
    auto& outputs = completion_addresses_;
    outputs = input_dependencies_;
    for (const auto& frame : frames_)
      for (auto index : frame->plan->result_slots) {
        const at::Tensor tensor = frame->values.at(index).toTensor();
        outputs.push_back(reinterpret_cast<synapse_helpers::device_ptr>(tensor.storage().data_ptr().get()));
      }
    // Direct-state destinations are mutated aliases rather than graph return
    // values. Register their storage as graph output as well so reset, prefill,
    // bucket replacement, and destruction observe the real final consumer.
    for (const auto& frame : frames_)
      for (auto index : frame->plan->input_slots) {
        const auto& value = frame->values.at(index);
        if (!value.isTensor()) continue;
        const at::Tensor tensor = value.toTensor();
        if (tensor.scalar_type() == at::kFloat && tensor.numel() >= 393216)
          outputs.push_back(reinterpret_cast<synapse_helpers::device_ptr>(tensor.storage().data_ptr().get()));
      }
    std::sort(outputs.begin(), outputs.end());
    outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
  }

  void registerOutputs() {
    auto self = shared_from_this();
    auto& device = habana::HPUDeviceContext::get_device();
    device.register_producer_on_stream(
        std::vector<synapse_helpers::device_ptr>(completion_addresses_), device.get_stream(0), [self]() {});
  }

  std::atomic<State> state_{State::Created};
  mutable std::mutex mutex_;
  RuntimeApis::SynGraph syn_graph_ = nullptr;
  bool syn_capture_active_ = false;
  std::vector<RuntimeApis::HclGraph> hcl_graphs_;
  RuntimeApis::HclBatch hcl_batch_ = nullptr;
  std::array<uint64_t, 12> joint_statistics_ {};
  std::array<uint64_t, 8> retirement_statistics_ {};
  std::vector<uint32_t> prepared_consumers_;
  std::vector<uint32_t> prepared_producers_;
  std::vector<NativeCollectiveDependency> prepared_dependencies_;
  std::vector<std::shared_ptr<PreparedGroupPlan>> plans_;
  std::vector<std::shared_ptr<PreparedFrame>> frames_;
  std::vector<std::vector<FixedInputSignature>> signatures_;
  std::vector<at::Tensor> dynamic_inputs_;
  std::vector<PendingInputCopy> pending_input_copies_;
  std::vector<synapse_helpers::device_ptr> pending_input_dependencies_;
  std::vector<synapse_helpers::device_ptr> input_dependencies_, completion_addresses_;
  std::vector<at::Tensor> state_tensors_;
  std::atomic<uint64_t> workspace_bytes_{0};
  bool fixed_inputs_ready_ = false;
  std::vector<std::shared_ptr<GenericResourceHolder>> resources_;
  std::shared_ptr<habana::HcclCommunicator> communicator_;
  std::atomic<uint64_t> segment_count_{0};
  std::atomic<uint64_t> collective_count_{0};
  std::atomic<uint64_t> external_collective_count_{0};
  std::atomic<uint64_t> captured_command_count_{0};
  std::atomic<uint64_t> captured_relocation_count_{0};
  std::atomic<uint64_t> global_program_bytes_{0};
  std::atomic<uint64_t> arc_program_bytes_{0};
  std::atomic<uint64_t> hcl_command_bytes_per_replay_{0};
  std::atomic<uint64_t> hcl_stream_ccb_bytes_{0};
  std::atomic<uint64_t> hcl_replay_bytes_{0};
  std::atomic<uint64_t> hcl_ccb_wrap_count_{0};
  std::atomic<uint64_t> hcl_submission_count_{0};
  std::atomic<uint64_t> input_epoch_{0};
  std::atomic<uint64_t> input_update_copies_{0};
  std::atomic<uint64_t> input_update_bytes_{0};
  std::atomic<uint64_t> scheduled_epoch_{0};
  std::atomic<uint64_t> executed_epoch_{0};
  size_t prefix_node_count_ = 0;
  std::atomic<uint64_t> replay_count_{0};
};

}  // namespace tp2_native
