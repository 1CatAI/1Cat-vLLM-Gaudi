// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr const char* names[]={"custom_op::custom_deepseek_v41_index_scores_gaudi2",
    "custom_op::custom_deepseek_v41_index_threshold_gaudi2", "custom_op::custom_deepseek_v41_index_emit_gaudi2"};
using Meta=habana::PartialOutputMetaDataVector;
struct Params {int ratio,reindex,blocks;};
Meta metadata(unsigned mode,const at::Stack& s) {
    const unsigned tensors=mode==0?6:mode==1?2:4;
    const int ratio=s.at(tensors).toInt(),reindex=s.at(tensors+1).toInt();
    TORCH_CHECK((ratio==1||ratio==2) && (reindex==0||reindex==1),"Invalid indexer ratio/mode");
    const auto first=s.at(0).toTensor();
    for(unsigned i=0;i<tensors;++i) {
        const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.is_contiguous()&&!t.requires_grad()&&t.device()==first.device(),"Indexer tensors must be contiguous on one device");
    }
    if(mode==0) {
        const auto w=s.at(1).toTensor(),cache=s.at(2).toTensor(),pages=s.at(3).toTensor();
        const auto position=s.at(4).toTensor(),candidates=s.at(5).toTensor();
        const int64_t capacity=s.at(8).toInt();
        TORCH_CHECK(first.scalar_type()==at::kBFloat16&&first.sizes()==at::IntArrayRef({1,32,128}) &&
            w.scalar_type()==at::kBFloat16&&w.sizes()==at::IntArrayRef({1,32}) &&
            cache.scalar_type()==at::kByte&&cache.dim()==2&&cache.size(1)==68 &&
            pages.scalar_type()==at::kInt&&pages.dim()==1&&pages.numel()>0 &&
            position.scalar_type()==at::kInt&&position.sizes()==at::IntArrayRef({1}) &&
            candidates.scalar_type()==at::kInt&&candidates.sizes()==at::IntArrayRef({1,2048}) &&
            capacity>=512&&capacity<=1048576&&capacity%64==0 &&
            (reindex?capacity==16384:capacity<=pages.numel()*(128/ratio)),"Invalid C1 paged index scoring contract");
        return {{at::kFloat,{capacity}},{at::kFloat,{((capacity/8+63)/64)*64}}};
    }
    const auto position=s.at(1).toTensor();
    const int blocks=s.at(tensors+2).toInt();
    TORCH_CHECK(first.scalar_type()==at::kFloat&&first.dim()==1&&first.numel()%64==0 &&
        position.scalar_type()==at::kInt&&position.sizes()==at::IntArrayRef({1})&&
        (blocks==0||blocks==1),"Invalid index selection score/length contract");
    if(mode==1)return {{at::kInt,{50}}};
    const auto candidates=s.at(2).toTensor(),stats=s.at(3).toTensor();
    TORCH_CHECK(candidates.scalar_type()==at::kInt&&candidates.sizes()==at::IntArrayRef({1,2048})&&
        stats.scalar_type()==at::kInt&&stats.sizes()==at::IntArrayRef({50}),"Invalid index selection metadata");
    return {{at::kInt,{1,blocks?2048:512}}};
}
const bool registered=[] {
    for(unsigned mode=0;mode<3;++mode)habana::custom_op::registerUserCustomOp(names[mode],names[mode]+11,
        [mode](const at::Stack& s){return metadata(mode,s);},
        [mode](const at::Stack& s,size_t& bytes)->std::shared_ptr<void> {
            const unsigned base=mode==0?6:mode==1?2:4;bytes=sizeof(Params);
            return std::make_shared<Params>(Params{int(s.at(base).toInt()),int(s.at(base+1).toInt()),
                mode==0?0:int(s.at(base+2).toInt())});
        });
    return true;
}();
template<bool Fake,unsigned Mode> std::vector<at::Tensor> execute(const at::Stack& s) {
    const auto meta=metadata(Mode,s);const auto x=s.at(0).toTensor();
    if(Fake) {std::vector<at::Tensor> outputs;
        for(const auto& m:meta)outputs.push_back(at::empty(m.shape,x.options().dtype(m.dtype)));return outputs;}
    TORCH_CHECK(registered&&x.device().type()==at::kHPU);
    auto descriptor=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(names[Mode]);
    return descriptor.execute(s);
}
template<bool Fake> std::tuple<at::Tensor,at::Tensor> score(const at::Tensor& q,const at::Tensor& w,const at::Tensor& cache,
    const at::Tensor& pages,const at::Tensor& pos,const at::Tensor& candidates,int64_t ratio,int64_t reindex,int64_t capacity) {
    auto out=execute<Fake,0>({q,w,cache,pages,pos,candidates,ratio,reindex,capacity});return {out[0],out[1]};
}
template<bool Fake> at::Tensor threshold(const at::Tensor& scores,const at::Tensor& pos,int64_t ratio,int64_t reindex,int64_t blocks) {
    return execute<Fake,1>({scores,pos,ratio,reindex,blocks})[0];
}
template<bool Fake> at::Tensor emit(const at::Tensor& scores,const at::Tensor& pos,const at::Tensor& candidates,
    const at::Tensor& stats,int64_t ratio,int64_t reindex,int64_t blocks) {
    return execute<Fake,2>({scores,pos,candidates,stats,ratio,reindex,blocks})[0];
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_index_scores_gaudi2(Tensor query, Tensor weights, Tensor cache, Tensor pages, Tensor position, Tensor candidates, int ratio, int reindex, int capacity) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_index_threshold_gaudi2(Tensor scores, Tensor position, int ratio, int reindex, int blocks) -> Tensor");
    m.def("custom_deepseek_v41_index_emit_gaudi2(Tensor scores, Tensor position, Tensor candidates, Tensor metadata, int ratio, int reindex, int blocks) -> Tensor");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_index_scores_gaudi2",score<false>);
    m.impl("custom_deepseek_v41_index_threshold_gaudi2",threshold<false>);
    m.impl("custom_deepseek_v41_index_emit_gaudi2",emit<false>);
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_index_scores_gaudi2",score<true>);
    m.impl("custom_deepseek_v41_index_threshold_gaudi2",threshold<true>);
    m.impl("custom_deepseek_v41_index_emit_gaudi2",emit<true>);
}
