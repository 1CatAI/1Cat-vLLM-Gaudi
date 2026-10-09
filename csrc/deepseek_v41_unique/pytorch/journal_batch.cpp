// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"
namespace {
constexpr auto schema="custom_op::custom_deepseek_v41_journal_batch_gaudi2";
habana::OutputMetaDataVector metadata(const at::Stack& s) {
    TORCH_CHECK(s.size()==11,"Journal batch needs eight byte sources and three controls");
    const auto x=s[0].toTensor();int64_t columns=0;
    for(int i=0;i<8;++i) {
        const auto source=s[i].toTensor();
        TORCH_CHECK(source.dim()==2 && source.size(0)>=6 && source.size(1)>0 &&
                    source.scalar_type()==at::kByte && source.device()==x.device() &&
                    source.is_contiguous() && !source.requires_grad(),
                    "Journal batch requires contiguous byte views of persistent source rows");
        columns=std::max(columns,source.size(1));
    }
    const auto pos=s[8].toTensor(),pages=s[9].toTensor(),config=s[10].toTensor();
    TORCH_CHECK(pos.sizes()==at::IntArrayRef({6}) && pos.scalar_type()==at::kInt &&
                pages.dim()==1 && pages.numel()>0 && pages.scalar_type()==at::kInt &&
                config.sizes()==at::IntArrayRef({8,2}) && config.scalar_type()==at::kInt &&
                pos.device()==x.device() && pages.device()==x.device() && config.device()==x.device() &&
                pos.is_contiguous() && pages.is_contiguous() && config.is_contiguous(),
                "Journal batch controls must be contiguous device I32 arrays");
    columns=(columns+255)/256*256;
    return {{at::kByte,{8,6,columns}},{at::kInt,{8,6}}};
}
class Batch final:public habana::OpBackend {
public:
    Batch(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("journal_batch"),t,{0,1},{},{},false) {
        SetOutputMetaFn(metadata);
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto m=metadata(s);
        habana::NodeAttr node;node.guid="custom_deepseek_v41_journal_batch_gaudi2";
        for(int i=0;i<11;++i)node.inputs.push_back(syn_in(i));
        node.output_attrs={{m[0].shape,m[0].dtype,0},{m[1].shape,m[1].dtype,1}};
        auto outputs=BuildNode(this,graph,std::move(node));
        syn_out(0)=std::move(outputs[0]);syn_out(1)=std::move(outputs[1]);
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
template<bool Meta> std::tuple<at::Tensor,at::Tensor> execute(const at::Tensor& a,const at::Tensor& b,
    const at::Tensor& c,const at::Tensor& d,const at::Tensor& e,const at::Tensor& f,
    const at::Tensor& g,const at::Tensor& h,const at::Tensor& pos,const at::Tensor& pages,const at::Tensor& config) {
    const at::Stack s{a,b,c,d,e,f,g,h,pos,pages,config};const auto m=metadata(s);
    if(Meta)return {at::empty(m[0].shape,a.options()),at::empty(m[1].shape,a.options().dtype(at::kInt))};
    TORCH_CHECK(ready && a.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(schema);auto result=op.execute(s);
    return {result[0],result[1]};
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
 m.def("custom_deepseek_v41_journal_batch_gaudi2(Tensor a, Tensor b, Tensor c, Tensor d, Tensor e, Tensor f, Tensor g, Tensor h, Tensor positions, Tensor pages, Tensor config) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){m.impl("custom_deepseek_v41_journal_batch_gaudi2",execute<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){m.impl("custom_deepseek_v41_journal_batch_gaudi2",execute<true>);}
