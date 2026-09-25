// SPDX-License-Identifier: Apache-2.0
// Included in the native bridge after the normal exchange implementation.
// This is host-side prepared replay, not a Synapse device graph API.

std::atomic<uint64_t> g_prepared_batches{0}, g_prepared_groups{0};
std::atomic<uint64_t> g_prepared_compute_nodes{0}, g_prepared_exchange_nodes{0};

c10::IValue preparedIValue(py::handle value) {
  if (value.is_none()) return c10::IValue();
  if (THPVariable_Check(value.ptr())) return value.cast<at::Tensor>();
  if (py::isinstance<py::bool_>(value)) return value.cast<bool>();
  if (py::isinstance<py::int_>(value)) return value.cast<int64_t>();
  if (py::isinstance<py::float_>(value)) return value.cast<double>();
  TORCH_CHECK(false, "Prepared bindings accept only tensors, fixed scalars, and None");
}

struct PreparedInputSignature {
  c10::IValue value;
  std::vector<int64_t> sizes, strides;
  int64_t offset = 0;
  explicit PreparedInputSignature(const c10::IValue& input) : value(input) {
    if (value.isTensor()) {
      const auto tensor = value.toTensor();
      sizes = tensor.sizes().vec();
      strides = tensor.strides().vec();
      offset = tensor.storage_offset();
    }
  }
  bool matches(const c10::IValue& input) const {
    if (value.isTensor()) {
      if (!input.isTensor()) return false;
      const auto expected = value.toTensor(), actual = input.toTensor();
      return actual.device() == expected.device() && actual.scalar_type() == expected.scalar_type() &&
          actual.sizes().equals(sizes) && actual.strides().equals(strides) && actual.storage_offset() == offset &&
          // A state destination identifies the decoder group and cache generation.
          // Sharing scratch outputs between different groups would create aliases.
          (!(expected.scalar_type() == at::kFloat && expected.numel() >= 393216) ||
           actual.data_ptr() == expected.data_ptr());
    }
    if (value.isNone()) return input.isNone();
    if (value.isInt()) return input.isInt() && input.toInt() == value.toInt();
    if (value.isDouble()) return input.isDouble() && input.toDouble() == value.toDouble();
    if (value.isBool()) return input.isBool() && input.toBool() == value.toBool();
    return false;
  }
};

struct PreparedNode {
  bool exchange = false;
  bool peer_only = false;
  bool reduction_only = false;
  uint64_t recipe_id = 0;
  habana::graph::GraphExec* graph = nullptr;
  std::vector<int64_t> inputs, outputs;
  uint32_t minimum_tile_bound = 0;
  float epsilon = 0;
  FusedRecipe* communication_recipe = nullptr;
};

class PreparedGroupPlan : public std::enable_shared_from_this<PreparedGroupPlan> {
 public:
  torch::jit::Stack slots;
  std::vector<int64_t> input_slots, result_slots;
  std::vector<PreparedInputSignature> signatures;
  std::vector<PreparedNode> nodes;
  struct ReshapeView {
    int64_t source, output;
    std::vector<int64_t> shape;
  };
  std::vector<ReshapeView> reshape_views;
  std::shared_ptr<habana::HcclCommunicator> communicator;
  bool sealed = false;
  std::atomic<bool> valid{true};

  int64_t add_slot(c10::IValue value, bool input) {
    TORCH_CHECK(!sealed, "Prepared plan is sealed");
    const int64_t index = slots.size();
    slots.push_back(std::move(value));
    if (input) {
      input_slots.push_back(index);
      signatures.emplace_back(slots.back());
    }
    return index;
  }

  void add_compute(uint64_t recipe_id, std::vector<int64_t> inputs, std::vector<int64_t> outputs) {
    TORCH_CHECK(!sealed && !outputs.empty(), "Prepared compute requires preallocated outputs");
    PreparedNode node;
    node.recipe_id = recipe_id;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    nodes.push_back(std::move(node));
  }

  void mark_last_optional_tile(uint32_t minimum) {
    TORCH_CHECK(!sealed && !nodes.empty() && !nodes.back().exchange && minimum > 0 && minimum <= 9,
                "Optional scorer requires an unsealed recipe and tag 1..8 or whole-scorer tag 9");
    nodes.back().minimum_tile_bound = minimum;
  }

  int64_t add_norm_view(int64_t source, const std::vector<int64_t>& shape) {
    TORCH_CHECK(!sealed, "Prepared norm view cannot change a sealed plan");
    const auto tensor = slots.at(source).toTensor();
    auto allowed = [](at::IntArrayRef sizes) {
      return sizes == at::IntArrayRef({1, 5120}) || sizes == at::IntArrayRef({1, 1, 5120});
    };
    TORCH_CHECK(allowed(tensor.sizes()) && allowed(shape) && tensor.scalar_type() == at::kBFloat16 &&
                    tensor.is_contiguous(), "Prepared norm view only changes a leading singleton dimension");
    return add_slot(tensor.view(shape), false);
  }

  int64_t add_reshape_view(int64_t source, const std::vector<int64_t>& shape) {
    TORCH_CHECK(!sealed, "Prepared reshape cannot change a sealed plan");
    const auto tensor = slots.at(source).toTensor();
    TORCH_CHECK(tensor.is_contiguous(), "Prepared reshape requires contiguous storage");
    const auto view = tensor.view(shape);
    TORCH_CHECK(view.numel() == tensor.numel() && view.storage_offset() == tensor.storage_offset(),
                "Prepared reshape must retain the complete source range");
    const auto output = add_slot(view, false);
    reshape_views.push_back({source, output, shape});
    return output;
  }

  void refresh_reshape_views(torch::jit::Stack& values) const {
    for (const auto& view : reshape_views)
      values.at(view.output) = values.at(view.source).toTensor().view(view.shape);
  }

  void add_exchange(std::vector<int64_t> inputs, std::vector<int64_t> outputs, double epsilon) {
    TORCH_CHECK(!sealed && inputs.size() == 3 && outputs.size() == 4, "Invalid prepared exchange bindings");
    PreparedNode node;
    node.exchange = true;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    node.epsilon = static_cast<float>(epsilon);
    nodes.push_back(std::move(node));
  }

  void add_peer_exchange(int64_t input, int64_t output) {
    TORCH_CHECK(!sealed && input != output, "Invalid prepared peer exchange bindings");
    PreparedNode node;
    node.exchange = true;
    node.peer_only = true;
    node.inputs = {input};
    node.outputs = {output};
    nodes.push_back(std::move(node));
  }

  void add_all_reduce(int64_t input, int64_t output) {
    add_peer_exchange(input, output);
    nodes.back().reduction_only = true;
  }

  void prepare(c10d::ProcessGroupEagerHCCL* backend, std::vector<int64_t> results) {
    TORCH_CHECK(!sealed && !nodes.empty() && tp2ExchangeEnabled(), "Prepared plans require dedicated TP2 exchange");
    TORCH_CHECK(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0 &&
                    GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE), "Prepared plans require the eager pipeline");
    TORCH_CHECK(c10::hpu::getCurrentHPUStream().stream() == 0, "Prepared v1 supports the default compute stream");
    // Preparation is the only point that joins host queues. All recipes must
    // have completed their first normal lowering before retaining executors.
    habana::eager::JoinPendingPipelineThreads();
    communicator = backend->lowLatencyCommunicator();
    TORCH_CHECK(communicator, "Prepared communicator is not initialized");
    result_slots = std::move(results);
    for (auto& node : nodes) {
      if (node.exchange) {
        const auto partial = slots.at(node.inputs[0]).toTensor();
        if (node.peer_only) {
          const auto peer = slots.at(node.outputs[0]).toTensor();
          const int64_t hidden = partial.numel();
          const char* v41Flag = std::getenv("VLLM_HPU_DSV41_GRAPH_REPLAY");
          const bool v41 = v41Flag && std::strcmp(v41Flag, "1") == 0;
          const char* batchFlag = std::getenv("VLLM_HPU_DSV41_BATCH_DECODE");
          const int64_t v41Maximum = batchFlag && std::strcmp(batchFlag, "1") == 0 ? 64 * 5120 : 32768;
          const bool v41Shape = v41 && !node.reduction_only && hidden >= 128 &&
                                hidden <= v41Maximum && hidden % 128 == 0;
          TORCH_CHECK(hidden == 4096 || (!node.reduction_only && hidden == 5120) || v41Shape,
                      "Unsupported prepared TP2 hidden width");
          TORCH_CHECK(partial.sizes() == at::IntArrayRef({1, hidden}) &&
                          partial.scalar_type() == at::kBFloat16 && partial.device().type() == at::kHPU,
                      "Prepared peer exchange requires TP2 C1 BF16 hidden states");
          validateTensor(peer, partial, "prepared peer", at::kBFloat16);
          TORCH_CHECK(partial.is_contiguous() && partial.storage().data_ptr().get() != peer.storage().data_ptr().get(),
                      "Prepared peer exchange buffers must be contiguous and distinct");
          continue;
        }
        const auto residual = slots.at(node.inputs[1]).toTensor();
        const auto weight = slots.at(node.inputs[2]).toTensor();
        TORCH_CHECK(partial.sizes() == at::IntArrayRef({1, 5120}) &&
                    partial.scalar_type() == at::kBFloat16 && residual.sizes() == partial.sizes() &&
                    residual.scalar_type() == at::kBFloat16 && weight.numel() == 5120,
                    "Prepared exchange v1 requires contiguous TP2 C1 BF16 hidden states");
        for (auto index : node.inputs)
          TORCH_CHECK(slots.at(index).toTensor().is_contiguous(), "Noncontiguous prepared exchange input");
        for (auto index : node.outputs)
          TORCH_CHECK(slots.at(index).toTensor().is_contiguous(), "Noncontiguous prepared exchange output");
        node.communication_recipe = fusedRecipeCache().get(5120, 5120, node.epsilon, true);
      } else {
        torch::jit::Stack inputs;
        for (auto index : node.inputs) inputs.push_back(slots.at(index));
        node.graph = habana::graph::GraphStorage::get().prepared_static_exec(node.recipe_id, inputs);
      }
    }
    sealed = true;
  }

  bool matches(const torch::jit::Stack& inputs) const {
    if (!sealed || !valid.load() || inputs.size() != signatures.size()) return false;
    for (size_t i = 0; i < inputs.size(); ++i)
      if (!signatures[i].matches(inputs[i])) return false;
    return true;
  }

  std::vector<at::Tensor> outputs() const {
    std::vector<at::Tensor> result;
    for (auto index : result_slots) result.push_back(slots.at(index).toTensor());
    return result;
  }
};

struct PreparedFrame {
  std::shared_ptr<PreparedGroupPlan> plan;
  torch::jit::Stack values;
};

void runPreparedExchangeNode(const std::shared_ptr<habana::HcclCommunicator>& communicator,
                             const PreparedNode& node, const torch::jit::Stack& values) {
  if (node.peer_only) {
    TORCH_CHECK(values.size() == 2, "Invalid prepared peer bindings");
    runTp2ExchangePeer(communicator, values[0].toTensor(), values[1].toTensor(), 0, node.reduction_only);
  } else {
    TORCH_CHECK(values.size() == 7, "Invalid prepared fused norm bindings");
    runFusedAllReduceNorm(communicator,
        values[0].toTensor(), values[1].toTensor(), values[2].toTensor(),
        values[3].toTensor(), values[4].toTensor(), values[5].toTensor(), values[6].toTensor(),
        node.epsilon, 0, node.communication_recipe);
  }
}

void replayPreparedGroups(std::vector<std::shared_ptr<PreparedGroupPlan>> plans,
                           std::vector<torch::jit::Stack> inputs,
                           std::vector<size_t> node_limits = {}) {
  TORCH_CHECK(plans.size() == inputs.size() && !plans.empty(), "Prepared group/input count mismatch");
  if (node_limits.empty()) {
    for (const auto& plan : plans) node_limits.push_back(plan->nodes.size());
  }
  TORCH_CHECK(node_limits.size() == plans.size(), "Prepared node-limit count mismatch");
  for (size_t i = 0; i < plans.size(); ++i)
    TORCH_CHECK(node_limits[i] > 0 && node_limits[i] <= plans[i]->nodes.size(),
                "Prepared node limit must be a nonempty prefix");
  TORCH_CHECK(c10::hpu::getCurrentHPUStream().stream() == 0, "Prepared replay requires default compute stream");
  std::vector<PreparedFrame> frames;
  frames.reserve(plans.size());
  for (size_t i = 0; i < plans.size(); ++i) {
    auto plan = plans[i];
    TORCH_CHECK(plan->matches(inputs[i]), "Prepared input layout or generation changed; reprepare before replay");
    // Logical frontend bindings are immutable templates. GraphExec rewrites
    // backend TensorImpl sizes/offsets; never share those mutable wrappers
    // between recipe nodes or replay frames.
    PreparedFrame frame{plan, plan->slots};
    auto incoming = std::move(inputs[i]);
    for (size_t j = 0; j < incoming.size(); ++j)
      frame.values[plan->input_slots[j]] = std::move(incoming[j]);
    plan->refresh_reshape_views(frame.values);
    for (const auto& value : frame.values)
      if (value.isTensor() && value.toTensor().device().type() == at::kHPU)
        habana::get_tensor_extra_meta(value.toTensor())->set_tensor_pipelined();
    frames.push_back(std::move(frame));
  }
  g_prepared_batches.fetch_add(1, std::memory_order_relaxed);
  g_prepared_groups.fetch_add(frames.size(), std::memory_order_relaxed);
  habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>(
      [frames = std::move(frames), node_limits = std::move(node_limits)]() mutable {
        for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
          auto& frame = frames[frame_index];
          auto held = std::make_shared<PreparedFrame>(std::move(frame));
          for (size_t node_index = 0; node_index < node_limits[frame_index]; ++node_index) {
            const auto& node = held->plan->nodes[node_index];
            if (!node.exchange) {
              torch::jit::Stack input;
              std::vector<at::Tensor> output;
              for (auto index : node.inputs) input.push_back(held->values.at(index));
              for (auto index : node.outputs)
                output.push_back(habana::eager::HbEagerTensorPool::get_backend_tensor(
                    held->values.at(index).toTensor()));
              auto backend_input = habana::eager::convert_ivalues_to_backend_tensors(input);
              habana::graph::GraphExec::LaunchRecipeTask(node.graph, std::move(backend_input), std::move(output), {});
              g_prepared_compute_nodes.fetch_add(1, std::memory_order_relaxed);
            } else {
              // We are already on LOWERING. Queue directly on EXECUTE after
              // the preceding recipe's execute task. Re-entering PipelineTask
              // here would put the exchange behind later compute lowering.
              torch::jit::Stack bindings;
              for (auto index : node.inputs) bindings.push_back(held->values.at(index));
              for (auto index : node.outputs) bindings.push_back(held->values.at(index));
              auto backend_bindings = habana::eager::convert_ivalues_to_backend_tensors(bindings);
              habana::HPUDeviceContext::execute_thread().enqueue([held, node, v = std::move(backend_bindings)]() {
                runPreparedExchangeNode(held->plan->communicator, node, v);
                g_prepared_exchange_nodes.fetch_add(1, std::memory_order_relaxed);
              });
            }
          }
          habana::HPUDeviceContext::execute_thread().enqueue([held]() {
            auto& device = habana::HPUDeviceContext::get_device();
            device.register_producer_on_stream({}, device.get_stream(0), [held]() {});
          });
        }
      });
}
