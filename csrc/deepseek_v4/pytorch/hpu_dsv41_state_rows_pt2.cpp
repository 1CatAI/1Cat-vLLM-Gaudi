// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/library.h>
#include "hpu_ops/op_backend.h"

namespace {
constexpr auto kWrite="custom_op::custom_deepseek_v41_state_rows_write_gaudi2";
constexpr auto kOrdered="custom_op::custom_deepseek_v41_state_rows_ordered_gaudi2";
constexpr auto kRead="custom_op::custom_deepseek_v41_state_rows_read_gaudi2";
std::string guid(at::ScalarType type,bool read) {
    return std::string("custom_deepseek_v41_state_rows_")+(read?"read_":"write_")+
        (type==at::kByte?"u8":type==at::kBFloat16?"bf16":"f32")+"_gaudi2";
}
habana::PartialOutputMetaDataVector metadata(const at::Stack& s,bool read) {
    const auto cache=s.at(0).toTensor(),rows=s.at(read?1:2).toTensor(),data=s.at(read?2:1).toTensor();
    TORCH_CHECK((cache.scalar_type()==at::kByte||cache.scalar_type()==at::kBFloat16||cache.scalar_type()==at::kFloat)&&
                cache.dim()==2&&cache.size(0)>0&&cache.size(1)>0&&cache.size(1)<=4096&&
                rows.scalar_type()==at::kInt&&rows.dim()==1&&rows.numel()>0&&rows.numel()<=64,
                "State rows require bounded C1 request rows and supported state dtype");
    if(read) { TORCH_CHECK(data.scalar_type()==at::kInt&&data.sizes()==rows.sizes(),
                        "State read requires producer completion for each request row"); }
    else { TORCH_CHECK(data.scalar_type()==cache.scalar_type()&&data.sizes()==at::IntArrayRef({rows.numel(),cache.size(1)}),
                     "State row write value shape/dtype mismatch"); }
    for(unsigned i=0;i<3;++i) {const auto t=s.at(i).toTensor();
        TORCH_CHECK(t.is_contiguous()&&t.device()==cache.device()&&!t.requires_grad(),
                    "State rows require contiguous inference tensors on one device");}
    return read?habana::PartialOutputMetaDataVector{{cache.scalar_type(),{rows.numel(),cache.size(1)}}}:
                habana::PartialOutputMetaDataVector{{at::kInt,{rows.numel()}}};
}
class StateRows : public habana::OpBackend {
    bool read_;
public:
    StateRows(int device,c10::ScalarType dtype,bool read)
        : OpBackend(device,NO_TPC+std::string("dsv41_state_rows"),dtype,{0},{},{},false),read_(read) {
        SetOutputMetaFn([read](const at::Stack& s) {
            const auto meta=metadata(s,read);
            return habana::OutputMetaDataVector{{meta[0].dtype,meta[0].shape}};
        });
    }
    void AddNode(synapse_helpers::graph& graph,const at::Stack& s) override {
        const auto meta=metadata(s,read_);
        syn_out(0)=std::move(BuildNode(this,graph,{guid(s.at(0).toTensor().scalar_type(),read_),
            {syn_in(0),syn_in(1),syn_in(2)},{{meta[0].shape,meta[0].dtype,0}}}).at(0));
    }
};
const bool registered=[] {
    for(const auto schema:{kWrite,kOrdered,kRead}) {
        const bool read=std::string(schema)==kRead;
        habana::custom_op::registerUserCustomOp(schema,guid(at::kByte,read),
            [read](const at::Stack& s){return metadata(s,read);},nullptr);
        habana::KernelRegistry().add(schema,[read](synDeviceId d,c10::ScalarType t){
            return std::make_shared<StateRows>(d,t,read);
        });
    }
    for(bool read:{false,true})for(auto type:{at::kByte,at::kBFloat16,at::kFloat}) {
        const auto name=guid(type,read);
        habana::custom_op::registerUserCustomOp("custom_op::"+name,name,
            [read](const at::Stack& s){return metadata(s,read);},nullptr);
    }
    return true;
}();
template<bool Fake,bool Read> at::Tensor execute(const at::Tensor& cache,const at::Tensor& second,const at::Tensor& third) {
    const at::Stack stack{cache,second,third};const auto output=metadata(stack,Read);
    if(Fake)return at::empty(output[0].shape,cache.options().dtype(output[0].dtype));
    TORCH_CHECK(registered&&cache.device().type()==at::kHPU);
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(Read?kRead:kOrdered);
    return op.execute(stack).at(0);
}
at::Tensor unwrap(const at::Tensor& value) {
    if(!at::functionalization::impl::isFunctionalTensor(value))return value;
    at::functionalization::impl::sync(value);
    return at::functionalization::impl::from_functional_tensor(value);
}
at::Tensor functionalize(const at::Tensor& cache,const at::Tensor& value,const at::Tensor& rows) {
    auto c=unwrap(cache),v=unwrap(value),r=unwrap(rows);
    static auto handle=c10::Dispatcher::singleton().findSchemaOrThrow(kOrdered,"")
        .typed<at::Tensor(const at::Tensor&,const at::Tensor&,const at::Tensor&)>();
    at::Tensor done;
    {at::AutoDispatchSkipFunctionalize guard;done=handle.call(c,v,r);}
    at::functionalization::impl::replace_(cache,c);
    at::functionalization::impl::commit_update(cache);
    at::functionalization::impl::sync(cache);
    return done;
}
}
TORCH_LIBRARY_FRAGMENT(custom_op,m) {
    m.def("custom_deepseek_v41_state_rows_write_gaudi2(Tensor(a!) cache, Tensor value, Tensor rows) -> Tensor");
    m.def("custom_deepseek_v41_state_rows_ordered_gaudi2(Tensor cache, Tensor value, Tensor rows) -> Tensor");
    m.def("custom_deepseek_v41_state_rows_read_gaudi2(Tensor cache, Tensor rows, Tensor completion) -> Tensor");
    for(bool read:{false,true})for(auto type:{at::kByte,at::kBFloat16,at::kFloat}) {
        m.def((guid(type,read)+(read?"(Tensor cache, Tensor rows, Tensor completion) -> Tensor":
                                   "(Tensor cache, Tensor value, Tensor rows) -> Tensor")).c_str());
    }
}
TORCH_LIBRARY_IMPL(custom_op,HPU,m) {
    m.impl("custom_deepseek_v41_state_rows_write_gaudi2",execute<false,false>);
    m.impl("custom_deepseek_v41_state_rows_ordered_gaudi2",execute<false,false>);
    m.impl("custom_deepseek_v41_state_rows_read_gaudi2",execute<false,true>);
    for(bool read:{false,true})for(auto type:{at::kByte,at::kBFloat16,at::kFloat}) {
        if(read)m.impl(guid(type,read).c_str(),execute<false,true>);
        else m.impl(guid(type,read).c_str(),execute<false,false>);
    }
}
TORCH_LIBRARY_IMPL(custom_op,Meta,m) {
    m.impl("custom_deepseek_v41_state_rows_write_gaudi2",execute<true,false>);
    m.impl("custom_deepseek_v41_state_rows_ordered_gaudi2",execute<true,false>);
    m.impl("custom_deepseek_v41_state_rows_read_gaudi2",execute<true,true>);
    for(bool read:{false,true})for(auto type:{at::kByte,at::kBFloat16,at::kFloat}) {
        if(read)m.impl(guid(type,read).c_str(),execute<true,true>);
        else m.impl(guid(type,read).c_str(),execute<true,false>);
    }
}
TORCH_LIBRARY_IMPL(custom_op,Functionalize,m) {
    m.impl("custom_deepseek_v41_state_rows_write_gaudi2",functionalize);
}
