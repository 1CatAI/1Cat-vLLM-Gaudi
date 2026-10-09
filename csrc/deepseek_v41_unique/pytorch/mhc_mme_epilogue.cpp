// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "backend/habana_operator.h"

namespace {
constexpr auto name = "custom_deepseek_v41_mhc_mme_epilogue_gaudi2";
constexpr auto schema = "custom_op::custom_deepseek_v41_mhc_mme_epilogue_gaudi2";
void validate(const at::Tensor& projection, const at::Tensor& residual,
              const at::Tensor& scale, const at::Tensor& base) {
    TORCH_CHECK(projection.dim()==2 && projection.size(0)>=2 && projection.size(0)<=6 &&
                (projection.size(1)==24 || projection.size(1)==48) && projection.scalar_type()==at::kFloat,
                "mHC MME epilogue requires FP32 C2-C6 projection");
    TORCH_CHECK(residual.sizes()==at::IntArrayRef({projection.size(0),20480}) &&
                residual.scalar_type()==at::kBFloat16 && scale.scalar_type()==at::kFloat &&
                base.scalar_type()==at::kFloat && scale.numel()==3 && base.numel()==24,
                "mHC MME epilogue requires BF16 residual and actual mHC parameters");
    for (const auto& x : {projection,residual,scale,base})
        TORCH_CHECK(x.device()==projection.device() && x.is_contiguous() && !x.requires_grad());
}
const bool registered = [] {
    habana::custom_op::registerUserCustomOp(schema,name,[](const at::Stack& values) {
        return habana::PartialOutputMetaDataVector{{at::kFloat,{values[0].toTensor().size(0),24}}};
    },nullptr);
    return true;
}();
template<bool Meta>
at::Tensor run(const at::Tensor& projection, const at::Tensor& residual,
               const at::Tensor& scale, const at::Tensor& base) {
    validate(projection,residual,scale,base);
    if constexpr (Meta) return at::empty({projection.size(0),24},projection.options());
    TORCH_CHECK(registered && projection.device().type()==at::kHPU);
    auto descriptor = habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return descriptor.execute({projection,residual,scale,base}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_mhc_mme_epilogue_gaudi2(Tensor projection, Tensor residual, Tensor scale, Tensor base) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl(name,run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl(name,run<true>);}
