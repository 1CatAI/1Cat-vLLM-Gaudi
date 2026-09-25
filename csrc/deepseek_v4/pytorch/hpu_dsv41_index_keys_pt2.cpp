// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr const char* schemas[]={"custom_op::custom_deepseek_v41_index_keys_gaudi2",
    "custom_op::custom_deepseek_v41_index_keys_tiled_gaudi2"};
struct Params { int ratio; };
habana::PartialOutputMetaDataVector metadata(const at::Stack& s) {
    const auto cache=s.at(0).toTensor(),pages=s.at(1).toTensor(),rows=s.at(2).toTensor();
    const auto ratio=s.at(3).toInt();
    TORCH_CHECK((ratio==1||ratio==2)&&cache.scalar_type()==at::kByte&&cache.dim()==2&&cache.size(1)==68&&
                cache.size(0)>0&&cache.size(0)<=1048704&&pages.scalar_type()==at::kInt&&
                (pages.dim()==1||pages.dim()==2)&&pages.size(-1)>0&&pages.size(-1)<=8192&&
                rows.scalar_type()==at::kInt&&rows.dim()==2&&
                rows.size(0)>=1&&rows.size(0)<=128&&rows.size(1)>=1&&rows.size(1)<=2048,
                "Invalid bounded prefill index-key gather shape/dtype/ratio");
    TORCH_CHECK(pages.dim()==1||pages.size(0)==rows.size(0),"Index-key request page table batch mismatch");
    for(unsigned i=0;i<3;++i){const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.device()==cache.device()&&t.is_contiguous()&&!t.requires_grad(),
                    "Index-key inputs must be contiguous on one device");}
    return {{at::kBFloat16,{rows.size(0),rows.size(1),128}}};
}
const bool registered=[] {
    for(const auto* schema:schemas) habana::custom_op::registerUserCustomOp(schema,schema+11,metadata,
        [](const at::Stack& s,size_t& bytes)->std::shared_ptr<void> {
            bytes=sizeof(Params);return std::make_shared<Params>(Params{int(s.at(3).toInt())});
        });return true;
}();
template<bool Fake,bool Tiled=false> at::Tensor gather(const at::Tensor& cache,const at::Tensor& pages,const at::Tensor& rows,int64_t ratio) {
    const auto meta=metadata({cache,pages,rows,ratio});
    if(Fake)return at::empty(meta[0].shape,cache.options().dtype(at::kBFloat16));
    TORCH_CHECK(registered&&cache.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schemas[Tiled?1:0]);
    return op.execute({cache,pages,rows,ratio}).at(0);
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_index_keys_gaudi2(Tensor cache, Tensor pages, Tensor rows, int ratio) -> Tensor");
    m.def("custom_deepseek_v41_index_keys_tiled_gaudi2(Tensor cache, Tensor pages, Tensor rows, int ratio) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {m.impl("custom_deepseek_v41_index_keys_gaudi2",gather<false>);m.impl("custom_deepseek_v41_index_keys_tiled_gaudi2",gather<false,true>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {m.impl("custom_deepseek_v41_index_keys_gaudi2",gather<true>);m.impl("custom_deepseek_v41_index_keys_tiled_gaudi2",gather<true,true>);}
