// SPDX-License-Identifier: Apache-2.0
// A destination-bound state epilogue; recurrent projections stay in the model.
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <habanalabs/perf_lib_layer_params.h>

#include "habana_eager/ops/eager_op.h"
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kFunctional = "custom_op::gdn_state_update";
constexpr auto kOut = "custom_op::gdn_state_update_out";

void validate(const at::Tensor& decayed, const at::Tensor& delta,
              const at::Tensor& key, const at::Tensor& destination) {
  TORCH_CHECK(destination.sizes() == at::IntArrayRef({1, 24, 128, 128}),
              "GDN direct update requires the TP2 C1 active state view");
  TORCH_CHECK(decayed.sizes() == at::IntArrayRef({1, 8, 3, 128, 128}) &&
              delta.sizes() == at::IntArrayRef({1, 8, 3, 128, 1}) &&
              key.sizes() == at::IntArrayRef({1, 8, 1, 1, 128}),
              "GDN direct update has an unsupported grouped layout");
  for (const auto& tensor : {decayed, delta, key, destination}) {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat && tensor.device() == destination.device() &&
                    tensor.is_contiguous() && !tensor.requires_grad(),
                "GDN direct update requires contiguous inference FP32 tensors on one device");
  }
  // The original state's last read is checked by the compiler pass. These
  // inputs must themselves be independent of the destination being written.
  for (const auto& source : {decayed, delta, key}) {
    TORCH_CHECK(at::get_overlap_status(destination, source) == at::MemOverlapStatus::No,
                "GDN direct update sources must not overlap its destination");
  }
}

habana::OutputMetaDataVector output_meta(const at::Stack& stack) {
  return {{at::kFloat, stack.at(3).toTensor().sizes().vec()}};
}

class GdnStateUpdate final : public habana::OpBackend {
 public:
  GdnStateUpdate(int device, c10::ScalarType dtype, bool out)
      : OpBackend(device, NO_TPC + std::string("gdn_state_update"), dtype,
                  out ? std::vector<int>{} : std::vector<int>{0}, {}, {}, out) {
    SetOutputMetaFn(output_meta);
  }

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
    // Match AOT's addcmul decomposition. The vendor addcmul GUID changes
    // rounding relative to the maintained multiply-then-add GDN graph.
    auto product = BuildNode(this, graph, {"mult_fwd_f32", {syn_in(1), syn_in(2)},
        {{{1, 8, 3, 128, 128}, at::kFloat}}});
    auto updated = BuildNode(this, graph, {"add_fwd_f32", {syn_in(0), product.at(0).get()},
        {{{1, 8, 3, 128, 128}, at::kFloat}}});
    auto output = BuildNode(this, graph, {"reshape", {updated.at(0).get()},
        {{stack.at(3).toTensor().sizes().vec(), at::kFloat, 0}}});
    syn_out(0) = std::move(output.at(0));
  }
};

const bool registered = [] {
  habana::custom_op::registerUserCustomOp(kFunctional, "unused_compound_guid",
      [](const at::Stack& stack) {
        return habana::PartialOutputMetaDataVector{{at::kFloat, stack.at(3).toTensor().sizes().vec()}};
      }, nullptr);
  habana::KernelRegistry().add(kFunctional, [](synDeviceId device, c10::ScalarType dtype) {
    return std::make_shared<GdnStateUpdate>(device, dtype, false);
  });
  habana::KernelRegistry().add(kOut, [](synDeviceId device, c10::ScalarType dtype) {
    return std::make_shared<GdnStateUpdate>(device, dtype, true);
  });
  return true;
}();

at::Tensor run(const at::Tensor& decayed, const at::Tensor& delta,
               const at::Tensor& key, const at::Tensor& destination) {
  validate(decayed, delta, key, destination);
  TORCH_CHECK(registered && destination.device().type() == at::kHPU);
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kFunctional);
  return descriptor.execute({decayed, delta, key, destination}).at(0);
}

at::Tensor& run_out(const at::Tensor& decayed, const at::Tensor& delta,
                    const at::Tensor& key, at::Tensor& destination) {
  validate(decayed, delta, key, destination);
  habana::eager::EagerOp<at::Tensor&> op{kOut, {decayed, delta, key, destination},
                                     {destination.sizes().vec()}, 3};
  op.set_eager_op_info({habana::eager::eagerOpKind::InplaceOut, kOut, 1});
  return op.call(destination);
}

at::Tensor cpu(const at::Tensor& decayed, const at::Tensor& delta,
               const at::Tensor& key, const at::Tensor& destination) {
  validate(decayed, delta, key, destination);
  return at::addcmul(decayed, delta, key).view(destination.sizes());
}

at::Tensor& cpu_out(const at::Tensor& decayed, const at::Tensor& delta,
                    const at::Tensor& key, at::Tensor& destination) {
  validate(decayed, delta, key, destination);
  auto grouped = destination.view({1, 8, 3, 128, 128});
  at::addcmul_out(grouped, decayed, delta, key);
  return destination;
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("gdn_state_update(Tensor decayed, Tensor delta, Tensor key, Tensor destination) -> Tensor");
  m.def("gdn_state_update_out(Tensor decayed, Tensor delta, Tensor key, Tensor(a!) destination) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("gdn_state_update", run);
  m.impl("gdn_state_update_out", run_out);
}
TORCH_LIBRARY_IMPL(custom_op, CPU, m) {
  m.impl("gdn_state_update", cpu);
  m.impl("gdn_state_update_out", cpu_out);
}
TORCH_LIBRARY_IMPL(custom_op, Meta, m) {
  m.impl("gdn_state_update", [](const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor& dst) {
    return at::empty(dst.sizes(), dst.options());
  });
  m.impl("gdn_state_update_out", [](const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor& dst)
      -> at::Tensor& { return dst; });
}
