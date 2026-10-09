// SPDX-License-Identifier: Apache-2.0
// Add a producer dependency to the shared, unchanged packed-key codec.
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto name="custom_op::custom_deepseek_v41_index_keys_write_ordered_gaudi2";
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto cache=s.at(0).toTensor(),pages=s.at(1).toTensor(),rows=s.at(2).toTensor(),done=s.at(3).toTensor();
    const auto ratio=s.at(4).toInt();
    TORCH_CHECK((ratio==1||ratio==2)&&cache.scalar_type()==at::kByte&&cache.dim()==2&&cache.size(1)==68&&
        cache.size(0)>0&&pages.scalar_type()==at::kInt&&pages.dim()==1&&pages.numel()>0&&pages.numel()<=8192&&
        rows.scalar_type()==at::kInt&&rows.dim()==2&&rows.size(0)>=1&&rows.size(0)<=128&&
        rows.size(1)>=1&&rows.size(1)<=2048&&done.scalar_type()==at::kInt&&done.dim()==2&&
        done.size(0)>=1&&done.size(0)<=6&&done.size(1)==36,"Invalid ordered index keys/writer completion");
    for(unsigned i=0;i<4;++i) {
        auto t=s.at(i).toTensor();
        TORCH_CHECK(t.device()==cache.device()&&t.is_contiguous()&&!t.requires_grad(),"Invalid index-key inputs");
    }
    return {{at::kBFloat16,{rows.size(0),rows.size(1),128}}};
}
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(name,name+11,metadata,
        [](const at::Stack& s,size_t& bytes)->std::shared_ptr<void> {
            bytes=sizeof(int32_t);return std::make_shared<int32_t>(s.at(4).toInt());
        });return true;
}();
template<bool Meta> at::Tensor run(const at::Tensor& cache,const at::Tensor& pages,const at::Tensor& rows,
                                  const at::Tensor& done,int64_t ratio) {
    auto out=metadata({cache,pages,rows,done,ratio});
    if(Meta)return at::empty(out[0].shape,cache.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered&&cache.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(name);
    return op.execute({cache,pages,rows,done,ratio}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_index_keys_write_ordered_gaudi2(Tensor cache, Tensor pages, Tensor rows, Tensor completion, int ratio) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_index_keys_write_ordered_gaudi2",run<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_index_keys_write_ordered_gaudi2",run<true>);}
