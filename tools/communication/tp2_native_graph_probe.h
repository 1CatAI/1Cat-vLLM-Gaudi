// SPDX-License-Identifier: Apache-2.0
// Two-segment device qualification graph: compute -> TP2 exchange -> compute.

namespace tp2_native {

class ProbeExecutionTiming {
 public:
  explicit ProbeExecutionTiming(std::array<uint64_t, 2>& output) : output_(output), start_{clock(CLOCK_MONOTONIC), clock(CLOCK_THREAD_CPUTIME_ID)} {}
  ~ProbeExecutionTiming() {
    output_[1] = clock(CLOCK_THREAD_CPUTIME_ID) - start_[1];
    output_[0] = clock(CLOCK_MONOTONIC) - start_[0];
  }
 private:
  static uint64_t clock(clockid_t source) {
    timespec now{};
    clock_gettime(source, &now);
    return uint64_t(now.tv_sec) * 1000000000 + now.tv_nsec;
  }
  std::array<uint64_t, 2>& output_;
  std::array<uint64_t, 2> start_;
};

class NativeTp2GraphProbe : public std::enable_shared_from_this<NativeTp2GraphProbe> {
 public:
  enum class State { Created, Instantiated, Invalid, Closed };

  NativeTp2GraphProbe(const c10::intrusive_ptr<c10d::Backend>& backend,
                      at::Tensor input, at::Tensor residual, at::Tensor weight, double epsilon)
      : input_(std::move(input)), residual_(std::move(residual)), weight_(std::move(weight)),
        epsilon_(static_cast<float>(epsilon)) {
    auto* group = dynamic_cast<c10d::ProcessGroupEagerHCCL*>(backend.get());
    TORCH_CHECK(group, "Native TP2 graph probe requires ProcessGroupEagerHCCL");
    TORCH_CHECK(input_.device().type() == at::kHPU && input_.scalar_type() == at::kBFloat16 &&
                    input_.sizes() == at::IntArrayRef({1, 5120}) && input_.is_contiguous(),
                "Native TP2 graph probe requires a contiguous [1, 5120] BF16 HPU input");
    TORCH_CHECK(residual_.sizes() == input_.sizes() && residual_.scalar_type() == at::kBFloat16 &&
                    residual_.is_contiguous() && residual_.device() == input_.device(),
                "Native TP2 graph probe residual mismatch");
    TORCH_CHECK(weight_.numel() == 5120 && weight_.scalar_type() == at::kBFloat16 &&
                    weight_.is_contiguous() && weight_.device() == input_.device(),
                "Native TP2 graph probe weight mismatch");
    TORCH_CHECK(tp2ExchangeEnabled(), "Native TP2 graph probe requires the dedicated TP2 exchange algorithm");

    communicator_ = group->lowLatencyCommunicator();
    TORCH_CHECK(communicator_, "Native TP2 graph probe communicator is not initialized");
    pre_residual_ = torch::empty_like(input_);
    pre_normalized_ = torch::empty_like(input_);
    pre_inverse_ = torch::empty({1, 1}, input_.options().dtype(at::kFloat));
    peer_ = torch::empty_like(input_);
    output_residual_ = torch::empty_like(input_);
    output_normalized_ = torch::empty_like(input_);
    output_inverse_ = torch::empty({1, 1}, input_.options().dtype(at::kFloat));

    resources_ = std::make_shared<GenericResourceHolder>();
    tensors_ = {input_, residual_, weight_, pre_residual_, pre_normalized_, pre_inverse_,
                peer_, output_residual_, output_normalized_, output_inverse_};
    std::vector<void*> addresses;
    addresses.reserve(tensors_.size());
    for (const auto& tensor : tensors_) {
      resources_->add_tensor(tensor);
      addresses.push_back(tensor.data_ptr());
      habana::get_tensor_extra_meta(tensor)->set_tensor_pipelined();
    }
    communicator_->getDeviceCtxt()->lock_address(addresses, resources_->get_address_lock());
    for (const auto& tensor : {input_, residual_, weight_})
      input_storage_.push_back(reinterpret_cast<synapse_helpers::device_ptr>(tensor.storage().data_ptr().get()));
    completion_storage_ = input_storage_;
    for (const auto& tensor : {output_normalized_, output_residual_})
      completion_storage_.push_back(
          reinterpret_cast<synapse_helpers::device_ptr>(tensor.storage().data_ptr().get()));
  }

  ~NativeTp2GraphProbe() {
    if (state_ != State::Created && state_ != State::Closed) state_ = State::Invalid;
  }

  void reference() {
    // This unscored oracle executes the exact same compiled math through the
    // ordinary dedicated exchange. It must finish before native capture begins,
    // so reference collectives cannot alter a captured communicator's epochs.
    habana::eager::JoinPendingPipelineThreads();
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_ == State::Created && syn_graph_ == nullptr,
                "Reference execution must precede native capture");
    auto& device = habana::HPUDeviceContext::get_device();
    auto& stream = device.get_stream(0);
    stream_ = stream;
    device.add_wait_events_on_stream(input_storage_, stream);
    const auto& address = *resources_->get_address_lock();
    pre_recipe_ = fusedRecipeCache().get(input_.numel(), input_.size(-1), epsilon_, false);
    post_recipe_ = fusedRecipeCache().get(input_.numel(), input_.size(-1), epsilon_, true);
    launchFusedRecipe(stream_, 0,
                      {address.at(0), address.at(1), address.at(2), address.at(3), address.at(4), address.at(5)},
                      input_.numel(), input_.size(-1), epsilon_, false, pre_recipe_);
    runFusedAllReduceNorm(communicator_, pre_normalized_, residual_, weight_, peer_,
                         output_normalized_, output_residual_, output_inverse_, epsilon_, 0, post_recipe_);
    checkSynapse(synStreamSynchronize(stream_), "synStreamSynchronize(unscored reference)");
  }

  void capture() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_ == State::Created, "Native TP2 graph probe capture has already started");
    auto& api = RuntimeApis::get();
    api.require();
    auto& device = habana::HPUDeviceContext::get_device();
    stream_ = device.get_stream(0);
    const auto& address = *resources_->get_address_lock();
    try {
      checkSynapse(api.syn_create(&syn_graph_, stream_), "synNativeComputeGraphCreate(probe)");
      checkSynapse(api.syn_begin_capture(syn_graph_), "synNativeComputeGraphBeginCapture(probe)");
      syn_capture_active_ = true;
      pre_recipe_ = fusedRecipeCache().get(input_.numel(), input_.size(-1), epsilon_, false);
      launchFusedRecipe(stream_, 0,
                        {address.at(0), address.at(1), address.at(2), address.at(3), address.at(4), address.at(5)},
                        input_.numel(), input_.size(-1), epsilon_, false, pre_recipe_);

      constexpr int exchange_mode = 1;
      const hcclResult_t create = api.hcl_create(
          reinterpret_cast<const void*>(address.at(4)), reinterpret_cast<void*>(address.at(6)), input_.numel(),
          hcclBfloat16, hcclSum, *(communicator_->GetHcclHandle()), stream_, exchange_mode, &hcl_graph_);
      TORCH_CHECK(create == hcclSuccess, "hcclTp2NativeGraphCreate(probe) failed: ", create);
      TORCH_CHECK(api.hcl_capture(hcl_graph_) == hcclSuccess, "hcclTp2NativeGraphCapture(probe) failed");

      post_recipe_ = fusedRecipeCache().get(input_.numel(), input_.size(-1), epsilon_, true);
      launchFusedRecipe(stream_, address.at(4),
                        {address.at(6), address.at(1), address.at(2), address.at(7), address.at(8), address.at(9)},
                        input_.numel(), input_.size(-1), epsilon_, true, post_recipe_);
      checkSynapse(api.syn_end_capture(syn_graph_), "synNativeComputeGraphEndCapture(probe)");
      syn_capture_active_ = false;

      ComputeGraphInfo compute_info;
      HclGraphInfo hcl_info;
      checkSynapse(api.syn_get_info(syn_graph_, &compute_info), "synNativeComputeGraphGetInfo(probe)");
      TORCH_CHECK(api.hcl_get_info(hcl_graph_, &hcl_info) == hcclSuccess,
                  "hcclTp2NativeGraphGetInfo(probe) failed");
      TORCH_CHECK(compute_info.state == 2 && compute_info.segmentCount == 2,
                  "Native TP2 graph probe did not capture exactly two compute segments");
      TORCH_CHECK(compute_info.globalProgramBytes > 0 && compute_info.arcProgramBytes > 0,
                  "Native TP2 graph probe did not retain program storage");
      TORCH_CHECK(hcl_info.state == 2 && hcl_info.capturedCommandCount > 0 &&
                      hcl_info.dependencyRelocationCount >= 4 && hcl_info.wqeRelocationCount > 0 &&
                      hcl_info.externalCompletionAddressRelocationCount > 0 &&
                      hcl_info.capturedByteCount > 0 && hcl_info.maxStreamBytes > 0 &&
                      hcl_info.streamCcbBytes > 0,
                  "Native TP2 graph probe did not capture a replayable collective template");
      captured_commands_ = hcl_info.capturedCommandCount;
      captured_relocations_ = hcl_info.capturedRelocationCount;
      global_program_bytes_ = compute_info.globalProgramBytes;
      arc_program_bytes_ = compute_info.arcProgramBytes;
      captured_hcl_bytes_ = hcl_info.capturedByteCount;
      max_hcl_stream_bytes_ = hcl_info.maxStreamBytes;
      hcl_stream_ccb_bytes_ = hcl_info.streamCcbBytes;
      checkSynapse(synStreamSynchronize(stream_), "synStreamSynchronize(probe capture)");
      const char* batch_mode = std::getenv("HCL_TP2_NATIVE_BATCH");
      if (jointPlanEnabled() || (batch_mode && std::strcmp(batch_mode, "1") == 0)) {
        TORCH_CHECK(api.hcl_batch_create(&hcl_graph_, 1, &hcl_batch_) == hcclSuccess,
                    "hcclTp2NativeBatchCreate(probe) failed");
      }
      if (jointPlanEnabled()) {
        api.requirePlan();
        const uint32_t consumer = 1;
        checkSynapse(api.syn_prepare_plan(syn_graph_, &consumer, 1, hcl_info.completion.longSoIndex,
                                          replayNicBatch, hcl_batch_),
                     "synNativeComputeGraphPreparePlan(probe)");
        joint_plan_ = true;
      }
      state_ = State::Instantiated;
    } catch (...) {
      state_ = State::Invalid;
      if (syn_graph_ != nullptr && syn_capture_active_) {
        api.syn_abort_capture(syn_graph_);
        syn_capture_active_ = false;
      }
      if (hcl_batch_ != nullptr) {
        api.hcl_batch_destroy(hcl_batch_);
        hcl_batch_ = nullptr;
      }
      if (hcl_graph_ != nullptr) {
        api.hcl_destroy(hcl_graph_);
        hcl_graph_ = nullptr;
      }
      if (syn_graph_ != nullptr) {
        api.syn_destroy(syn_graph_);
        syn_graph_ = nullptr;
      }
      throw;
    }
  }

  void replay() {
    // Queue after the input producers, without joining the lowering/execute
    // pipeline on the caller. Later tensor consumers enter the same FIFO.
    auto self = shared_from_this();
    habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([self]() {
      habana::HPUDeviceContext::execute_thread().enqueue([self]() { self->replayOnExecute(); });
    });
  }

 private:
  void replayOnExecute() {
    std::lock_guard<std::mutex> lock(mutex_);
    ProbeExecutionTiming timing(last_execution_timing_);
    TORCH_CHECK(state_ == State::Instantiated, "Native TP2 graph probe is not instantiated");
    auto& api = RuntimeApis::get();
    try {
      auto& device = habana::HPUDeviceContext::get_device();
      device.add_wait_events_on_stream(input_storage_, device.get_stream(0));
      SyncInfo final_compute;
      if (joint_plan_) {
        checkSynapse(api.syn_replay_plan(syn_graph_, &final_compute, joint_statistics_.data(), joint_statistics_.size()),
                     "synNativeComputeGraphReplayPlan(probe)");
      } else {
      checkSynapse(api.syn_begin_replay(syn_graph_), "synNativeComputeGraphBeginReplay(probe)");
      SyncInfo first_compute;
      SyncInfo collective;
      checkSynapse(api.syn_replay_segment(syn_graph_, 0, nullptr, false, &first_compute),
                   "synNativeComputeGraphReplaySegment(probe pre)");
      const char* compare_prepared = std::getenv("HCL_TP2_NATIVE_COMPARE_PREPARED");
      const bool use_prepared = compare_prepared != nullptr && std::strcmp(compare_prepared, "1") == 0;
      TORCH_CHECK(!use_prepared || !hcl_batch_, "Prepared control cannot share a native batch communicator");
      const hcclResult_t hcl_status = hcl_batch_
          ? api.hcl_batch_replay(hcl_batch_, &first_compute, 1, &collective)
          : use_prepared
          ? api.hcl_replay_prepared(hcl_graph_, &first_compute, &collective)
          : api.hcl_stage(hcl_graph_, &first_compute, &collective);
      TORCH_CHECK(hcl_status == hcclSuccess,
                  use_prepared ? "hcclTp2NativeGraphReplayPrepared(probe) failed"
                               : "hcclTp2NativeGraphStage(probe) failed");
      checkSynapse(api.syn_replay_segment(syn_graph_, 1, &collective, true, &final_compute),
                   "synNativeComputeGraphReplaySegment(probe post)");
      if (!use_prepared && !hcl_batch_) {
        TORCH_CHECK(api.hcl_submit(hcl_graph_) == hcclSuccess, "hcclTp2NativeGraphSubmit(probe) failed");
      }
      }
      if (!replay_count_) first_compute_target_ = final_compute.targetValue;
      last_compute_target_ = final_compute.targetValue;
      ++replay_count_;
      batch_replay_count_ += hcl_batch_ != nullptr;
      auto self = shared_from_this();
      // The next input update must also wait for this graph's last reads.
      device.register_producer_on_stream(
          std::vector<synapse_helpers::device_ptr>(completion_storage_), device.get_stream(0), [self]() {});
    } catch (...) {
      state_ = State::Invalid;
      throw;
    }
  }

 public:
  std::vector<at::Tensor> outputs() const { return {output_normalized_, output_residual_}; }

  std::array<uint64_t, 2> lastExecutionTiming() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_execution_timing_;
  }

  std::array<uint64_t, 12> jointInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return joint_statistics_;
  }

  std::array<uint64_t, 19> info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    HclGraphInfo hcl_info;
    TORCH_CHECK(state_ == State::Instantiated && hcl_graph_ != nullptr,
                "Native TP2 graph probe info requires an instantiated graph");
    TORCH_CHECK(RuntimeApis::get().hcl_get_info(hcl_graph_, &hcl_info) == hcclSuccess,
                "hcclTp2NativeGraphGetInfo(probe live) failed");
    return {static_cast<uint64_t>(state_), replay_count_, 2, captured_commands_, captured_relocations_,
            global_program_bytes_, arc_program_bytes_, captured_hcl_bytes_, max_hcl_stream_bytes_,
            hcl_stream_ccb_bytes_, hcl_info.nativeReplayBytes, hcl_info.nativeCcbWrapCount,
            hcl_info.nativeSubmissionCount, hcl_info.wrapCapturedCommandCount,
            hcl_info.completionRingSize, hcl_info.nativeWrapReplayCount, batch_replay_count_,
            first_compute_target_, last_compute_target_};
  }

  void synchronize() {
    habana::eager::JoinPendingPipelineThreads();
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_ == State::Instantiated, "Native TP2 graph probe cannot synchronize in its current state");
    checkSynapse(synStreamSynchronize(stream_), "synStreamSynchronize(probe)");
  }

  void close() {
    habana::eager::JoinPendingPipelineThreads();
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == State::Closed) return;
    auto& api = RuntimeApis::get();
    if (stream_ != nullptr) checkSynapse(synStreamSynchronize(stream_), "synStreamSynchronize(probe close)");
    if (hcl_batch_ != nullptr) {
      TORCH_CHECK(api.hcl_batch_destroy(hcl_batch_) == hcclSuccess, "hcclTp2NativeBatchDestroy(probe) failed");
      hcl_batch_ = nullptr;
    }
    hcclResult_t hcl_status = hcclSuccess;
    if (hcl_graph_ != nullptr) {
      hcl_status = api.hcl_destroy(hcl_graph_);
      hcl_graph_ = nullptr;
    }
    synStatus syn_status = synSuccess;
    if (syn_graph_ != nullptr) {
      syn_status = api.syn_destroy(syn_graph_);
      syn_graph_ = nullptr;
    }
    state_ = hcl_status == hcclSuccess && syn_status == synSuccess ? State::Closed : State::Invalid;
    TORCH_CHECK(state_ == State::Closed, "Native TP2 graph probe close failed");
  }

 private:
  mutable std::mutex mutex_;
  State state_ = State::Created;
  at::Tensor input_, residual_, weight_;
  at::Tensor pre_residual_, pre_normalized_, pre_inverse_, peer_;
  at::Tensor output_residual_, output_normalized_, output_inverse_;
  std::array<at::Tensor, 10> tensors_;
  std::vector<synapse_helpers::device_ptr> input_storage_, completion_storage_;
  std::shared_ptr<GenericResourceHolder> resources_;
  std::shared_ptr<habana::HcclCommunicator> communicator_;
  RuntimeApis::SynGraph syn_graph_ = nullptr;
  RuntimeApis::HclGraph hcl_graph_ = nullptr;
  RuntimeApis::HclBatch hcl_batch_ = nullptr;
  std::array<uint64_t, 2> last_execution_timing_{};
  std::array<uint64_t, 12> joint_statistics_{};
  bool joint_plan_ = false;
  uint64_t batch_replay_count_ = 0;
  uint64_t first_compute_target_ = 0, last_compute_target_ = 0;
  bool syn_capture_active_ = false;
  synStreamHandle stream_ = nullptr;
  FusedRecipe* pre_recipe_ = nullptr;
  FusedRecipe* post_recipe_ = nullptr;
  float epsilon_ = 0;
  uint64_t replay_count_ = 0;
  uint64_t captured_commands_ = 0;
  uint64_t captured_relocations_ = 0;
  uint64_t global_program_bytes_ = 0;
  uint64_t arc_program_bytes_ = 0;
  uint64_t captured_hcl_bytes_ = 0;
  uint64_t max_hcl_stream_bytes_ = 0;
  uint64_t hcl_stream_ccb_bytes_ = 0;
};

}  // namespace tp2_native
