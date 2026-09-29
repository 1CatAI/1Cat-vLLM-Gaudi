// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace dsv41_control {

class PreparedInputs : public std::enable_shared_from_this<PreparedInputs> {
 public:
  PreparedInputs(const at::Tensor& ids, const at::Tensor& positions) {
    TORCH_CHECK(ids.device().type() == at::kHPU && positions.device() == ids.device() &&
                    ids.dim() == 1 && positions.sizes() == ids.sizes() &&
                    ids.numel() >= 1 && ids.numel() <= 128 &&
                    (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong) &&
                    positions.scalar_type() == at::kInt && ids.is_contiguous() &&
                    positions.is_contiguous() && ids.storage_offset() == 0 &&
                    positions.storage_offset() == 0,
                "Prepared input frames require bounded contiguous integer vectors");
    TORCH_CHECK(ids.storage().unsafeGetStorageImpl() != positions.storage().unsafeGetStorageImpl(),
                "Token and position destinations must not alias");
    // Preparation, not hot-path execution: retire initial allocations once.
    habana::eager::JoinPendingPipelineThreads();
    ids_ = habana::eager::HbEagerTensorPool::get_backend_tensor(ids);
    positions_ = habana::eager::HbEagerTensorPool::get_backend_tensor(positions);
    count_ = ids.numel();
    id_bytes_ = habana_helpers::GetNBytes(ids_) / count_;
    TORCH_CHECK((id_bytes_ == 4 || id_bytes_ == 8) &&
                    habana_helpers::GetNBytes(positions_) == count_ * 4,
                "Unsupported control-input physical integer width");
  }

  void upload(const std::vector<int64_t>& tokens, int64_t start) {
    TORCH_CHECK(GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0 &&
                    GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) &&
                    c10::hpu::getCurrentHPUStream().stream() == 0,
                "Prepared control input requires eager default-stream ordering");
    TORCH_CHECK(static_cast<int64_t>(tokens.size()) == count_ && start >= 0 &&
                    start <= static_cast<int64_t>(INT32_MAX) - count_,
                "Control input values do not match the prepared frame");
    auto bytes = std::make_shared<std::vector<uint8_t>>(count_ * (id_bytes_ + 4));
    for (int64_t index = 0; index < count_; ++index) {
      if (id_bytes_ == 4) {
        TORCH_CHECK(tokens[index] >= INT32_MIN && tokens[index] <= INT32_MAX,
                    "Token value does not fit the device integer representation");
        const int32_t token = static_cast<int32_t>(tokens[index]);
        std::memcpy(bytes->data() + index * 4, &token, 4);
      } else {
        std::memcpy(bytes->data() + index * 8, &tokens[index], 8);
      }
      const int32_t position = static_cast<int32_t>(start + index);
      std::memcpy(bytes->data() + count_ * id_bytes_ + index * 4, &position, 4);
    }
    auto owner = shared_from_this();
    // FIFO lowering prevents overtaking an earlier consumer still pending on
    // the lowering pool; a direct frontend execute enqueue would be unsafe.
    habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([owner, bytes]() {
      habana::HPUDeviceContext::execute_thread().enqueue([owner, bytes]() {
        synapse_helpers::device::transfer_manifest transfers;
        for (int index = 0; index < 2; ++index) {
          const auto& destination = index == 0 ? owner->ids_ : owner->positions_;
          synapse_helpers::device::transfer_desc transfer;
          transfer.src = reinterpret_cast<synapse_helpers::device_ptr>(
              bytes->data() + (index == 0 ? 0 : owner->count_ * owner->id_bytes_));
          transfer.dst = reinterpret_cast<synapse_helpers::device_ptr>(destination.data_ptr());
          transfer.dst_event_addr = reinterpret_cast<synapse_helpers::device_ptr>(
              destination.storage().data_ptr().get());
          transfer.bytes_to_transfer = owner->count_ * (index == 0 ? owner->id_bytes_ : 4);
          transfers.push_back(transfer);
        }
        // Stock helper owns address locks, pinned packing, destination waits
        // and publication of both real input producers for compiled consumers.
        habana::HPUDeviceContext::copy_data_to_device(transfers, [owner, bytes]() {}, 0);
      });
    });
    uploads_.fetch_add(1, std::memory_order_relaxed);
  }

  int64_t count() const { return count_; }
  uint64_t uploads() const { return uploads_.load(std::memory_order_relaxed); }
  int64_t payload_bytes() const { return count_ * (id_bytes_ + 4); }

 private:
  at::Tensor ids_, positions_;
  int64_t count_ = 0, id_bytes_ = 0;
  std::atomic<uint64_t> uploads_{0};
};

}  // namespace dsv41_control
