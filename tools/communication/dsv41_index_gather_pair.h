// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace dsv41_index_pair {

inline std::atomic<uint64_t> launches{0};

inline void run(const std::shared_ptr<habana::HcclCommunicator>& communicator,
                std::array<at::Tensor, 4> tensors) {
  auto context = communicator->getDeviceCtxt();
  const auto stream = communicator->getCommStream();
  auto resources = std::make_shared<GenericResourceHolder>();
  std::vector<void*> addresses;
  for (const auto& tensor : tensors) {
    const auto permutation = std::get<0>(habana_helpers::get_tensor_memory_permutation(tensor));
    TORCH_CHECK(std::is_sorted(permutation.begin(), permutation.end()),
                "Paired index gather requires unpermuted physical tensors");
    resources->add_tensor(tensor);
    addresses.push_back(tensor.data_ptr());
    context->prepare_stream(stream, reinterpret_cast<synapse_helpers::device_ptr>(
        tensor.storage().data_ptr().get()));
  }
  context->lock_address(addresses, resources->get_address_lock());
  const auto& locked = *resources->get_address_lock();
  TORCH_CHECK(hcclGroupStart() == hcclSuccess, "Paired index gather group start failed");
  const auto first = hcclAllGather(reinterpret_cast<const void*>(locked.at(0)),
                                  reinterpret_cast<void*>(locked.at(2)), tensors[0].numel(),
                                  hcclBfloat16, *(communicator->GetHcclHandle()), stream);
  const auto second = first == hcclSuccess
      ? hcclAllGather(reinterpret_cast<const void*>(locked.at(1)),
                      reinterpret_cast<void*>(locked.at(3)), tensors[1].numel(),
                      hcclBfloat16, *(communicator->GetHcclHandle()), stream)
      : first;
  const auto ended = hcclGroupEnd();
  TORCH_CHECK(first == hcclSuccess && second == hcclSuccess && ended == hcclSuccess,
              "Paired index gather failed: ", first, ", ", second, ", ", ended);
  // Preserve stock communication-stream output dependencies and ownership.
  auto& counter = context->get_active_recipe_counter();
  for (size_t index = 2; index < tensors.size(); ++index) {
    counter.increase();
    context->submit_events(stream, reinterpret_cast<synapse_helpers::device_ptr>(
        tensors[index].storage().data_ptr().get()), [resources, &counter]() mutable {
          resources.reset();
          counter.decrease_and_notify();
        });
  }
  launches.fetch_add(1, std::memory_order_relaxed);
}

inline std::tuple<at::Tensor, at::Tensor> enqueue(
    const c10::intrusive_ptr<c10d::Backend>& backend,
    const at::Tensor& query, const at::Tensor& gains,
    const at::Tensor& gathered_query, const at::Tensor& gathered_gains) {
  auto* group = dynamic_cast<c10d::ProcessGroupEagerHCCL*>(backend.get());
  TORCH_CHECK(group && group->getSize() == 4, "Paired index gather requires TP4 HCCL");
  TORCH_CHECK(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0 &&
                  GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
                  GET_ENV_FLAG_NEW(PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE) &&
                  c10::hpu::getCurrentHPUStream().stream() == 0,
              "Paired index gather requires the normal eager default-stream pipelines");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == 8 && query.size(2) == 128 &&
                  query.size(0) >= 1 && query.size(0) <= 6 &&
                  gains.dim() == 2 && gains.size(0) == query.size(0) && gains.size(1) == 8,
              "Paired index gather expects query[C1..C6,8,128] and gains[C1..C6,8]");
  std::array<at::Tensor, 4> tensors = {query, gains, gathered_query, gathered_gains};
  for (size_t index = 0; index < tensors.size(); ++index) {
    const auto& tensor = tensors[index];
    TORCH_CHECK(tensor.device().type() == at::kHPU && tensor.device() == query.device() &&
                    tensor.scalar_type() == at::kBFloat16 && tensor.is_contiguous(),
                "Paired index gather expects contiguous BF16 on one HPU");
    for (size_t previous = 0; previous < index; ++previous)
      TORCH_CHECK(tensor.storage().unsafeGetStorageImpl() !=
                      tensors[previous].storage().unsafeGetStorageImpl(),
                  "Paired index gather buffers cannot alias");
  }
  TORCH_CHECK(gathered_query.numel() == query.numel() * 4 &&
                  gathered_gains.numel() == gains.numel() * 4,
              "Paired index gather outputs must hold all four ranks");
  auto communicator = group->lowLatencyCommunicator();
  TORCH_CHECK(communicator, "Paired index gather communicator is uninitialized");
  for (auto& tensor : tensors) {
    tensor = habana::eager::HbEagerTensorPool::get_backend_tensor(tensor);
    habana::get_tensor_extra_meta(tensor)->set_tensor_pipelined();
  }
  habana::eager::PipelineTask<habana::eager::ThreadType::EXECUTE>(
      [communicator, tensors = std::move(tensors)]() mutable {
        run(communicator, std::move(tensors));
      });
  return {gathered_query, gathered_gains};
}

}  // namespace dsv41_index_pair
