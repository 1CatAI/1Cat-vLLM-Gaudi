// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_index_reduce_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto dots=s.at(0).toTensor(),weights=s.at(1).toTensor();
    TORCH_CHECK(dots.scalar_type()==at::kBFloat16&&dots.dim()==3&&dots.size(0)>=1&&dots.size(0)<=64&&
               dots.size(1)==32&&dots.size(2)>=128&&dots.size(2)<=2048&&dots.size(2)%128==0&&
               weights.scalar_type()==at::kBFloat16&&weights.sizes()==at::IntArrayRef({dots.size(0),32}),
               "Invalid ordered index head reduction shape/dtype");
    for(unsigned i=0;i<2;++i){const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.is_contiguous()&&!t.requires_grad()&&t.device()==dots.device(),
                   "Index head reduction needs contiguous operands on one device");}
    return {{at::kFloat,{dots.size(0),dots.size(2)}}};
}
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,schema+11,metadata,
        [](const at::Stack&,size_t& bytes)->std::shared_ptr<void>{bytes=0;return nullptr;});return true;
}();
template<bool Fake> at::Tensor reduce(const at::Tensor& dots,const at::Tensor& weights) {
    const auto meta=metadata({dots,weights});
    if(Fake)return at::empty(meta[0].shape,dots.options().dtype(at::kFloat));
    TORCH_CHECK(registered&&dots.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);
    return op.execute({dots,weights}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_index_reduce_gaudi2(Tensor dots, Tensor weights) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_index_reduce_gaudi2",reduce<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_index_reduce_gaudi2",reduce<true>);}
