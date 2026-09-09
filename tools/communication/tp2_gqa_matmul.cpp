// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto schema = "custom_op::tp2_gqa_matmul";
void validate(const at::Tensor& x, const at::Tensor& y, bool transpose) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && y.scalar_type() == at::kBFloat16 &&
              x.dim() == 4 && y.dim() == 4 && x.size(0) == y.size(0) &&
              x.size(1) == 2 && y.size(1) == 2 && x.size(2) == 6 &&
              x.size(3) == y.size(transpose ? 3 : 2) && x.size(0) > 0 && x.device() == y.device() &&
              !x.requires_grad() && !y.requires_grad(), "Direct GQA requires static BF16 inference batches");
  TORCH_CHECK((x.size(3) == 256 && y.size(transpose ? 2 : 3) == 128) ||
              (x.size(3) == 128 && y.size(transpose ? 2 : 3) == 256), "Unsupported GQA projection shape");
}
std::vector<int64_t> shape(const at::Tensor& x, const at::Tensor& y, bool transpose) {
  return {x.size(0), x.size(1), x.size(2), y.size(transpose ? 2 : 3)};
}
class GqaMatmul final : public habana::OpBackend {
 public:
  GqaMatmul(int device, c10::ScalarType dtype)
      : OpBackend(device, NO_TPC + std::string("tp2_gqa_matmul"), dtype, {0}, {}, {}, false) {
    SetOutputMetaFn([](const at::Stack& stack) {
      const auto x = stack.at(0).toTensor(), y = stack.at(1).toTensor();
      const bool transpose = stack.at(2).toBool();
      validate(x, y, transpose);
      return habana::OutputMetaDataVector{{at::kBFloat16, shape(x, y, transpose)}};
    });
  }
  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override {
    const auto x = stack.at(0).toTensor(), y = stack.at(1).toTensor();
    const bool transpose = stack.at(2).toBool();
    validate(x, y, transpose);
    synGEMMParams params{false, transpose};
    auto result = BuildNode(this, graph, {"batch_gemm", {syn_in(0), syn_in(1)},
                               {{shape(x, y, transpose), at::kBFloat16, 0}}, &params, sizeof(params)});
    syn_out(0) = std::move(result.at(0));
  }
};
const bool registered = [] {
  habana::custom_op::registerUserCustomOp(schema, "batch_gemm", [](const at::Stack& stack) {
    const auto x = stack.at(0).toTensor(), y = stack.at(1).toTensor();
    const bool transpose = stack.at(2).toBool();
    validate(x, y, transpose);
    return habana::PartialOutputMetaDataVector{{at::kBFloat16,shape(x, y, transpose)}};
  }, nullptr);
  habana::KernelRegistry().add(schema, [](synDeviceId device, c10::ScalarType dtype) {
    return std::make_shared<GqaMatmul>(device, dtype);
  });
  return true;
}();
at::Tensor run(const at::Tensor& x, const at::Tensor& y, bool transpose) {
  validate(x, y, transpose);
  TORCH_CHECK(x.device().type() == at::kHPU && registered, "Direct GQA requires HPU");
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
  return descriptor.execute({x, y, transpose}).at(0);
}
at::Tensor meta(const at::Tensor& x, const at::Tensor& y, bool transpose) {
  validate(x, y, transpose);
  return at::empty(shape(x, y, transpose), x.options());
}
}
TORCH_LIBRARY_FRAGMENT(custom_op, m) {
  m.def("tp2_gqa_matmul(Tensor x, Tensor y, bool transpose_rhs=False) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op, HPU, m) { m.impl("tp2_gqa_matmul", run); }
TORCH_LIBRARY_IMPL(custom_op, Meta, m) { m.impl("tp2_gqa_matmul", meta); }
