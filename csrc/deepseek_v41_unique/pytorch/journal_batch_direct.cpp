// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_journal_batch_direct_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size()==7,"Direct journal needs four sources and three controls");
    const auto x=s[0].toTensor();habana::OutputMetaDataVector out;
    for(int i=0;i<4;++i) {
        const auto source=s[i].toTensor();
        TORCH_CHECK(source.dim()==2 && source.size(0)>=6 && source.numel()/source.size(0)>0 &&
                    (source.element_size()==1 || source.element_size()==2 || source.element_size()==4) &&
                    source.device()==x.device() && source.is_contiguous() && !source.requires_grad(),
                    "Direct journal requires contiguous persistent source rows");
        auto shape=source.sizes().vec();shape[0]=6;out.push_back({source.scalar_type(),shape});
    }
    const auto pos=s[4].toTensor(),pages=s[5].toTensor(),config=s[6].toTensor();
    TORCH_CHECK(pos.sizes()==at::IntArrayRef({6}) && pos.scalar_type()==at::kInt &&
                pages.dim()==1 && pages.numel()>0 && pages.scalar_type()==at::kInt &&
                config.sizes()==at::IntArrayRef({4,2}) && config.scalar_type()==at::kInt &&
                pos.device()==x.device() && pages.device()==x.device() && config.device()==x.device() &&
                pos.is_contiguous() && pages.is_contiguous() && config.is_contiguous(),
                "Direct journal controls must be contiguous device I32 arrays");
    for(int i=0;i<4;++i)out.push_back({at::kInt,{6}});return out;
}

class Batch final:public habana::OpBackend {
public:
    Batch(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("journal_batch_direct"),t,{0,1,2,3,4,5,6,7},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=metadata(s);
        habana::NodeAttr node;node.guid="custom_deepseek_v41_journal_batch_direct_gaudi2";
        int32_t byteCodes=0;
        for(int i=0;i<4;++i) {
            const auto x=s[i].toTensor();const auto columns=x.numel()/x.size(0);
            node.inputs.push_back(syn_in(i));
            node.output_attrs.push_back({{6,columns},x.scalar_type(),i});
            const int code=x.element_size()==4?2:x.element_size()==2?1:0;
            byteCodes|=code<<(i*2);
        }
        for(int i=4;i<7;++i)node.inputs.push_back(syn_in(i));
        for(int i=4;i<8;++i)node.output_attrs.push_back({m[i].shape,m[i].dtype,i});
        node.params=&byteCodes;node.param_size=sizeof(byteCodes);
        auto outputs=BuildNode(this,graph,std::move(node));
        for(int i=0;i<8;++i)syn_out(i)=std::move(outputs[i]);
    }
};
const bool ready=[] {
    habana::custom_op::registerUserCustomOp(schema,schema+11,[](const at::Stack& s) {
        habana::PartialOutputMetaDataVector out;
        for(const auto& m:metadata(s))out.push_back({m.dtype,m.shape});return out;
    },nullptr);
    habana::KernelRegistry().add(schema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Batch>(d,t);});
    return true;
}();
using Results=std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
template<bool Meta> Results execute(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,const at::Tensor& d,
    const at::Tensor& pos,const at::Tensor& pages,const at::Tensor& config) {
    const at::Stack s{a,b,c,d,pos,pages,config};const auto m=metadata(s);
    std::vector<at::Tensor> result;
    if(Meta)for(const auto& meta:m)result.push_back(at::empty(meta.shape,a.options().dtype(meta.dtype)));
    else {
        TORCH_CHECK(ready && a.device().type()==at::kHPU);
        auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);result=op.execute(s);
    }
    return {result[0],result[1],result[2],result[3],result[4],result[5],result[6],result[7]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
 m.def("custom_deepseek_v41_journal_batch_direct_gaudi2(Tensor a, Tensor b, Tensor c, Tensor d, Tensor positions, Tensor pages, Tensor config) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_journal_batch_direct_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_journal_batch_direct_gaudi2",execute<true>);}
