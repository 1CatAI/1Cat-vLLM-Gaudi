// SPDX-License-Identifier: Apache-2.0
// Generic Bridge ABI adapter for the concat cache-key patch in bridge.patch.
// Keep the installed frontend intact while qualifying a private runtime bundle.
#include "habana_eager/eager_exec.h"
#include <c10/util/hash.h>
#include <dlfcn.h>
#include <stdexcept>

namespace habana::eager {

size_t EagerExec::calculate_operator_key(
    const UniqueIdxVec& parent_vec,
    torch::jit::Stack& stack) {
  using Original = size_t (*)(EagerExec*, const UniqueIdxVec&, torch::jit::Stack&);
  static const auto original = reinterpret_cast<Original>(dlsym(
      RTLD_NEXT,
      "_ZN6habana5eager9EagerExec22calculate_operator_keyERKNS0_12UniqueIdxVecERSt6vectorIN3c106IValueESaIS7_EE"));
  if (original == nullptr) {
    throw std::runtime_error("The qualified Bridge eager cache-key ABI is unavailable");
  }
  size_t key = original(this, parent_vec, stack);
  if (m_symbol == c10::Symbol::fromQualString("aten::cat")) {
    // Synapse chooses ConcatFcdNode at construction for axis zero. Patching
    // params on a cached graph cannot change that node into a non-FCD concat.
    // Reuse shapes within one axis, never reuse the graph across axes.
    key = at::hash_combine(key, at::IValue::hash(stack.at(1)));
  }
  return key;
}

} // namespace habana::eager
