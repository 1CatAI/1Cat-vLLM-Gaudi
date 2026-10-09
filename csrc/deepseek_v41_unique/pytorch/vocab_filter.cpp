// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_vocab_filter_gaudi2";
constexpr auto guid="custom_deepseek_v41_vocab_filter_mask_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    const auto scores=s[0].toTensor(),cutoffs=s[1].toTensor();const auto width=s[2].toInt();
    TORCH_CHECK(scores.dim()==2 && scores.size(0)>=8 && scores.size(0)<=48 && scores.size(0)%8==0 &&
                scores.size(1)>=128 && scores.size(1)<=4096 && scores.size(1)%128==0 &&
                cutoffs.sizes()==at::IntArrayRef({scores.size(0)/8,1}) && scores.scalar_type()==at::kFloat &&
                cutoffs.scalar_type()==at::kFloat && cutoffs.device()==scores.device() &&
                scores.is_contiguous() && cutoffs.is_contiguous() && !scores.requires_grad() &&
                !cutoffs.requires_grad() && width>=1 && width<=256,"Bounded filter requires C1-C6 eight-way F32 shards");
    return {{at::kFloat,{scores.size(0),width}},{at::kInt,{scores.size(0),width}},{at::kInt,{scores.size(0),1}}};
}
class Filter final:public habana::OpBackend {
 public:
    Filter(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("vocab_filter"),t,{0,1,2},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto out=metadata(s);int params[]={static_cast<int>(s[0].toTensor().size(1)),static_cast<int>(s[2].toInt())};
        auto masks=BuildNode(this,graph,{guid,{syn_in(0),syn_in(1)},
            {{{s[0].toTensor().size(0),params[0]/32},at::kInt}},params,sizeof(int)});
        auto result=BuildNode(this,graph,{"custom_deepseek_v41_vocab_filter_emit_gaudi2",{syn_in(0),masks[0].get()},
            {{out[0].shape,out[0].dtype,0},{out[1].shape,out[1].dtype,1},{out[2].shape,out[2].dtype,2}},params,sizeof(params)});
        for(int i=0;i<3;++i)syn_out(i)=std::move(result[i]);
    }
};
const bool registered=[] {
    habana::custom_op::registerUserCustomOp(schema,guid,[](const at::Stack& s) {
        const auto m=metadata(s);habana::PartialOutputMetaDataVector out;
        for(const auto& v:m)out.push_back({v.dtype,v.shape});return out;
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Filter>(d,t);});return true;
}();
template<bool Meta> std::tuple<at::Tensor,at::Tensor,at::Tensor> execute(const at::Tensor& scores,
                                                                      const at::Tensor& cutoffs,int64_t width) {
    const at::Stack s{scores,cutoffs,width};const auto m=metadata(s);
    if(Meta)return {at::empty(m[0].shape,scores.options()),at::empty(m[1].shape,scores.options().dtype(at::kInt)),
                    at::empty(m[2].shape,scores.options().dtype(at::kInt))};
    TORCH_CHECK(registered && scores.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);auto result=op.execute(s);
    return {result[0],result[1],result[2]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_vocab_filter_gaudi2(Tensor scores, Tensor cutoffs, int width) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_vocab_filter_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_vocab_filter_gaudi2",execute<true>);}
