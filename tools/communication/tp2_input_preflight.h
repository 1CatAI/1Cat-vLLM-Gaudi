// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace tp2_input {

// Retain the allocation independently of TensorImpl so set_/resize_ cannot
// retire the captured storage and reuse its identity before validation.
struct InputSnapshot {
  c10::Storage storage;
  uintptr_t address;
  c10::Device device;
  c10::ScalarType dtype;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  static uintptr_t effectiveAddress(const at::Tensor& tensor) {
    // Match storage base plus offset, including pool-backed tensor facades.
    return reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()) +
        tensor.storage_offset() * tensor.element_size();
  }

  explicit InputSnapshot(const at::Tensor& tensor)
      : storage(tensor.storage()), address(effectiveAddress(tensor)),
        device(tensor.device()), dtype(tensor.scalar_type()),
        sizes(tensor.sizes().vec()), strides(tensor.strides().vec()) {}

  bool matchesLayout(const at::Tensor& tensor) const {
    return tensor.defined() && tensor.has_storage() &&
        tensor.layout() == c10::kStrided && tensor.device() == device &&
        tensor.scalar_type() == dtype && tensor.sizes().equals(sizes) &&
        tensor.strides().equals(strides);
  }

  bool matchesAllocation(const at::Tensor& tensor) const {
    return matchesLayout(tensor) &&
        tensor.storage().unsafeGetStorageImpl() == storage.unsafeGetStorageImpl() &&
        effectiveAddress(tensor) == address;
  }
};

class FixedInputPreflight {
 public:
  FixedInputPreflight(const std::vector<at::Tensor>& states,
                      std::vector<at::Tensor> destinations)
      : destinations_(std::move(destinations)) {
    for (const auto& tensor : states) states_.emplace_back(tensor);
    for (const auto& tensor : destinations_) bindings_.emplace_back(tensor);
  }

  pybind11::object updates(pybind11::sequence states,
                           pybind11::sequence sources) const {
    RECORD_FUNCTION("dsv41::native_input_preflight", std::vector<c10::IValue>());
    if (states.size() != states_.size() || sources.size() != bindings_.size())
      return pybind11::none();
    try {
      for (size_t i = 0; i < states_.size(); ++i) {
        if (!states_[i].matchesAllocation(pybind11::cast<at::Tensor>(states[i])))
          return pybind11::none();
      }
      pybind11::list changed;
      for (size_t i = 0; i < bindings_.size(); ++i) {
        const auto& binding = bindings_[i];
        // Destinations are held, but an external resize_/set_ still invalidates
        // the captured graph. No input has been copied at this point.
        if (!binding.matchesAllocation(destinations_[i])) return pybind11::none();
        const auto source = pybind11::cast<at::Tensor>(sources[i]);
        if (!binding.matchesLayout(source)) return pybind11::none();
        if (InputSnapshot::effectiveAddress(source) != binding.address) {
          if (source.storage().unsafeGetStorageImpl() == binding.storage.unsafeGetStorageImpl())
            return pybind11::none();
          changed.append(i);
        }
      }
      return std::move(changed);
    } catch (const pybind11::cast_error&) {
      return pybind11::none();
    }
  }

 private:
  std::vector<at::Tensor> destinations_;
  std::vector<InputSnapshot> states_;
  std::vector<InputSnapshot> bindings_;
};

}  // namespace tp2_input
