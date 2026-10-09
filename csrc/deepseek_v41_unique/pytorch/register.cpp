// SPDX-License-Identifier: Apache-2.0
// Shared C<=6 expert protocol with qualified SAT decode and device-selected M.
// Compiler placement and complete consumer timing are validated separately.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <synapse_api.h>
#include <dlfcn.h>
#include "backend/helpers/create_tensor.h"
#include "hpu_ops/op_backend.h"

namespace {
using Tensor=synapse_helpers::tensor;
constexpr auto kControlSchema="custom_op::custom_deepseek_v41_unique_control_i32_gaudi2";
constexpr auto kControlGuid="custom_deepseek_v41_unique_control_i32_gaudi2";
constexpr auto kPackSchema="custom_op::custom_deepseek_v41_unique_pack_fp8_gaudi2";
constexpr auto kPackGuid="custom_deepseek_v41_unique_pack_fp8_gaudi2";
constexpr auto kProgramSchema="custom_op::custom_deepseek_v41_unique_program_i32_gaudi2";
constexpr auto kProgramGuid="custom_deepseek_v41_unique_program_i32_gaudi2";
constexpr auto kPipelineSchema="custom_op::custom_deepseek_v41_unique_expert_fp8_gaudi2";
constexpr auto kWeightGuid="custom_deepseek_v41_expert_n256_sat_fp8_gaudi2";
constexpr auto kIdsGuid="custom_deepseek_v41_unique_ids_i32_gaudi2";
constexpr auto kActivateGuid="custom_deepseek_v41_unique_silu_quant_fp8_gaudi2";
constexpr auto kScaleGuid="custom_deepseek_v41_unique_scale_fp8_gaudi2";
constexpr auto kRestoreSchema="custom_op::custom_deepseek_v41_unique_restore_bf16_gaudi2";
constexpr auto kRestoreGuid="custom_deepseek_v41_unique_restore_bf16_gaudi2";


void tensor(const at::Tensor& t,at::ScalarType type,const at::Device& device) {
    // Bridge metadata may omit the index of the current process's sole HPU.
    // Explicit, different device indices remain an invalid contract.
    const bool sameDevice=t.device().type()==device.type() &&
        (!t.device().has_index() || !device.has_index() || t.device().index()==device.index());
    TORCH_CHECK(t.scalar_type()==type && sameDevice && t.is_contiguous() && !t.requires_grad(),
                "Invalid unique expert tensor: expected ", type, " on ", device,
                "; got ", t.scalar_type(), " on ", t.device(), " with sizes ", t.sizes(),
                " and strides ", t.strides(), "; requires_grad=", t.requires_grad());
}

habana::PartialOutputMetaDataVector controlMeta(const at::Stack& s) {
    const auto ids=s.at(0).toTensor(),generation=s.at(1).toTensor();
    tensor(ids,at::kInt,ids.device());tensor(generation,at::kInt,ids.device());
    TORCH_CHECK(ids.dim()==2 && ids.size(0)>=1 && ids.size(0)<=6 && ids.size(1)==6 &&
        generation.sizes()==at::IntArrayRef({2}) && s.at(2).toInt()>=6 && s.at(2).toInt()<=384,
        "Invalid unique expert control input");
    return {{at::kInt,{10,64}},{at::kShort,{11,64}}};
}
class Control final:public habana::OpBackend {
public:Control(int d,c10::ScalarType t):OpBackend(d,kControlGuid,t,{0,1},{},{},false) {
    SetOutputMetaFn([](const at::Stack& s){controlMeta(s);return habana::OutputMetaDataVector{
        {at::kInt,{10,64}},{at::kShort,{11,64}}};});}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        controlMeta(s);int32_t experts=s.at(2).toInt();
        auto out=BuildNode(this,g,{kControlGuid,{syn_in(0),syn_in(1)},
            {{{10,64},at::kInt,0},{{11,64},at::kShort,1}},&experts,sizeof(experts)});
        syn_out(0)=std::move(out.at(0));syn_out(1)=std::move(out.at(1));
    }
};
habana::PartialOutputMetaDataVector packMeta(const at::Stack& s) {
    const auto& x=s.at(0).toTensor();const auto& sx=s.at(1).toTensor();
    const auto& routing=s.at(2).toTensor();const auto& metadata=s.at(3).toTensor();
    tensor(x,at::ScalarType::Float8_e4m3fn,x.device());tensor(sx,at::kFloat,x.device());
    tensor(routing,at::kFloat,x.device());tensor(metadata,at::kInt,x.device());
    TORCH_CHECK(x.dim()==2 && x.size(0)>=1 && x.size(0)<=6 && x.size(1)==5120 &&
        sx.sizes()==at::IntArrayRef({x.size(0),1}) && routing.sizes()==at::IntArrayRef({x.size(0),6}) &&
        metadata.sizes()==at::IntArrayRef({10,64}),"Invalid quantized unique expert input");
    return {{at::ScalarType::Float8_e4m3fn,{36,6,5120}},{at::kFloat,{36,6}},{at::kFloat,{36,6}}};
}
class Pack final:public habana::OpBackend {
public:Pack(int d,c10::ScalarType t):OpBackend(d,kPackGuid,t,{0,1,2},{},{},false) {
        SetOutputMetaFn([](const at::Stack& s){packMeta(s);return habana::OutputMetaDataVector{
            {at::ScalarType::Float8_e4m3fn,{36,6,5120}},{at::kFloat,{36,6}},{at::kFloat,{36,6}}};});}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        packMeta(s);
        auto out=BuildNode(this,g,{kPackGuid,{syn_in(0),syn_in(1),syn_in(2),syn_in(3)},
            {{{36,6,5120},at::ScalarType::Float8_e4m3fn,0},
             {{36,6},at::kFloat,1},{{36,6},at::kFloat,2}}});
        for(unsigned i=0;i<3;++i)syn_out(i)=std::move(out.at(i));
    }
};

habana::PartialOutputMetaDataVector programMeta(const at::Stack& s) {
    const std::vector<std::vector<int64_t>> sizes{{6,65536},{2048,4},{10,64}};
    for(unsigned i=0;i<3;++i){const auto& t=s.at(i).toTensor();tensor(t,at::kInt,t.device());
        TORCH_CHECK(t.sizes()==at::IntArrayRef(sizes[i]),"Invalid FP8 compact program input");}
    return {{at::kInt,{65536}}};
}
class Program final:public habana::OpBackend {
public:Program(int d,c10::ScalarType t):OpBackend(d,kProgramGuid,t,{0},{},{},false) {
        SetOutputMetaFn([](const at::Stack& s){programMeta(s);return habana::OutputMetaDataVector{{at::kInt,{65536}}};});}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        programMeta(s);auto out=BuildNode(this,g,{kProgramGuid,{syn_in(0),syn_in(1),syn_in(2)},{{{65536},at::kInt,0}}});
        syn_out(0)=std::move(out.at(0));
    }
};

std::vector<int64_t> pipelineMeta(const at::Stack& s) {
    const auto device=s.at(0).toTensor().device();
    const at::ScalarType types[]={at::ScalarType::Float8_e4m3fn,at::kFloat,at::kFloat,at::kInt,at::kShort,
        at::kShort,at::kShort,at::kShort,at::kShort,at::kBFloat16,at::kBFloat16,at::kBFloat16,at::kInt};
    for(unsigned i=0;i<13;++i)tensor(s.at(i).toTensor(),types[i],device);
    const auto experts=s.at(5).toTensor().size(0);
    const auto q13=s.at(5).toTensor(),q2=s.at(7).toTensor();
    const auto h=q13.size(2)/64, k=q2.size(2)/64;
    TORCH_CHECK(experts>0 && experts<=384 && h==5120 && k>0 && k<=1280 && k%128==0 &&
        q13.sizes()==at::IntArrayRef({experts,2*k/256,h*64}) &&
        q2.sizes()==at::IntArrayRef({experts,h/256,k*64}) &&
        s.at(0).toTensor().sizes()==at::IntArrayRef({36,6,h}) &&
        s.at(1).toTensor().sizes()==at::IntArrayRef({36,6}) &&
        s.at(2).toTensor().sizes()==at::IntArrayRef({36,6}) &&
        s.at(3).toTensor().sizes()==at::IntArrayRef({10,64}) &&
        s.at(4).toTensor().sizes()==at::IntArrayRef({11,64}) &&
        s.at(6).toTensor().sizes()==at::IntArrayRef({experts,2*k/256,h*4+128}) &&
        s.at(8).toTensor().sizes()==at::IntArrayRef({experts,h/256,k*4+128}) &&
        s.at(9).toTensor().sizes()==at::IntArrayRef({128}) &&
        s.at(10).toTensor().sizes()==at::IntArrayRef({experts,2*k/256,256}) &&
        s.at(11).toTensor().sizes()==at::IntArrayRef({experts,h/256,256}) &&
        s.at(12).toTensor().sizes()==at::IntArrayRef({65536}),"Invalid unique SAT expert geometry");
    return {36,6,5120};
}
class Pipeline final:public habana::OpBackend {
    Tensor make(synapse_helpers::graph& g,const std::vector<int64_t>& sizes,at::ScalarType dtype) {
        std::vector<int64_t> strides(sizes.size(),1);for(int i=int(sizes.size())-2;i>=0;--i)strides[i]=strides[i+1]*sizes[i+1];
        return habana_helpers::create_tensor(sizes,strides,g,false,false,SynInput(0).ref().device_id(),dtype);
    }
    Tensor slice(synapse_helpers::graph& g,synTensor input,const std::vector<int64_t>& sizes,unsigned axis,
                 int64_t first,int64_t count,at::ScalarType dtype) {
        synSliceParams p{};for(unsigned i=0;i<std::size(p.axes);++i){p.axes[i]=i;p.steps[i]=1;p.ends[i]=1;}
        for(unsigned i=0;i<sizes.size();++i)p.ends[i]=sizes[sizes.size()-1-i];
        p.starts[axis]=first;p.ends[axis]=first+count;auto result=sizes;result[sizes.size()-1-axis]=count;
        auto out=BuildNode(this,g,{"slice",{input},{{result,dtype}},&p,sizeof(p)});return std::move(out.at(0));
    }
    Tensor concat(synapse_helpers::graph& g,const std::vector<synTensor>& inputs,const std::vector<int64_t>& sizes) {
        synConcatenateParams p{};p.axis=1;auto out=BuildNode(this,g,{"concat",inputs,{{sizes,at::kBFloat16}},&p,sizeof(p)});
        return std::move(out.at(0));
    }
public:Pipeline(int d,c10::ScalarType t):OpBackend(d,NO_TPC+std::string("dsv41_unique_expert"),t,{0},{},{},false) {
        SetOutputMetaFn([](const at::Stack& s){return habana::OutputMetaDataVector{{at::kBFloat16,pipelineMeta(s)}};});}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {
        pipelineMeta(s);
        const unsigned h=s.at(0).toTensor().size(2), k=s.at(7).toTensor().size(2)/64;
        // All consumers share the qualified SAT decoder. Ordinary compiler
        // slicing owns SRAM placement and lifetimes; no serial RMW arena.
        auto ids=BuildNode(this,g,{kIdsGuid,{syn_in(3),syn_in(4),syn_in(12)},{{{1,36},at::kInt}}});
        auto weight=[&](synTensor selected,unsigned owner,unsigned rows,unsigned columns,unsigned qi,unsigned si,const char* stage){
            synapse_helpers::graph::OpNameContext name(g,"dsv41_unique_sat_s"+std::to_string(owner)+"_"+stage);
            auto out=BuildNode(this,g,{kWeightGuid,{selected,syn_in(qi),syn_in(si),syn_in(9)},
                {{{1,rows,columns},at::ScalarType::Float8_e4m3fn}}});
            return ReshapeHelper(g,out.at(0).get(),{rows,columns},at::ScalarType::Float8_e4m3fn);
        };
        auto gemm=[&](synTensor a,synTensor b,unsigned owner,unsigned rows,unsigned columns){
            synGEMMParams params(false,false);
            synapse_helpers::graph::OpNameContext name(g,
                "dsv41_grouped_v6_m6_s"+std::to_string(owner)+"_p6_k"+std::to_string(rows)+"_matrix");
            auto out=BuildNode(this,g,{"gemm",{a,b},{{{6,columns},at::kFloat}},&params,sizeof(params)});
            return std::move(out.at(0));
        };
        std::vector<Tensor> outputs;std::vector<synTensor> pointers;
        for(unsigned owner=0;owner<36;++owner){
            auto selected=slice(g,ids.at(0).get(),{1,36},0,owner,1,at::kInt);
            auto packed=slice(g,syn_in(0),{36,6,h},2,owner,1,at::ScalarType::Float8_e4m3fn);
            auto activation=ReshapeHelper(g,packed.get(),{6,h},at::ScalarType::Float8_e4m3fn);
            auto w13=weight(selected.get(),owner,h,2*k,5,6,"w13");
            auto p13=gemm(activation.get(),w13.get(),owner,h,2*k);
            int32_t parameter=owner;
            auto middle=BuildNode(this,g,{kActivateGuid,{p13.get(),syn_in(3),syn_in(1),syn_in(10),syn_in(2)},
                {{{6,k},at::ScalarType::Float8_e4m3fn},{{6,1},at::kFloat}},&parameter,sizeof(parameter)});
            auto w2=weight(selected.get(),owner,k,h,7,8,"w2");
            auto p2=gemm(middle.at(0).get(),w2.get(),owner,k,h);
            auto scaled=BuildNode(this,g,{kScaleGuid,{p2.get(),syn_in(3),middle.at(1).get(),syn_in(11)},
                {{{6,h},at::kBFloat16}},&parameter,sizeof(parameter)});
            outputs.push_back(std::move(scaled.at(0)));pointers.push_back(outputs.back().get());
        }
        auto joined=concat(g,pointers,{216,5120});syn_out(0)=ReshapeHelper(g,joined.get(),{36,6,5120},at::kBFloat16,0);
    }
};

habana::PartialOutputMetaDataVector restoreMeta(const at::Stack& s) {
    const auto& x=s.at(0).toTensor();const auto& metadata=s.at(1).toTensor();const auto tokens=s.at(2).toInt();
    tensor(x,at::kBFloat16,x.device());tensor(metadata,at::kInt,x.device());
    TORCH_CHECK(x.sizes()==at::IntArrayRef({36,6,5120}) && metadata.sizes()==at::IntArrayRef({10,64}) && tokens>=1&&tokens<=6,
        "Invalid unique restore shape: matrices=", x.sizes(), ", metadata=", metadata.sizes(),
        ", tokens=", tokens);return {{at::kBFloat16,{tokens,5120}}};
}
class Restore final:public habana::OpBackend {
public:Restore(int d,c10::ScalarType t):OpBackend(d,kRestoreGuid,t,{0},{},{},false){
        SetOutputMetaFn([](const at::Stack& s){return habana::OutputMetaDataVector{{at::kBFloat16,restoreMeta(s)[0].shape}};});}
    void AddNode(synapse_helpers::graph& g,const at::Stack& s) override {auto shape=restoreMeta(s)[0].shape;
        auto out=BuildNode(this,g,{kRestoreGuid,{syn_in(0),syn_in(1)},{{shape,at::kBFloat16,0}}});syn_out(0)=std::move(out.at(0));}
};

const bool registered=[] {
    habana::custom_op::registerUserCustomOp(kControlSchema,kControlGuid,controlMeta,nullptr);
    habana::KernelRegistry().add(kControlSchema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Control>(d,t);});
    habana::custom_op::registerUserCustomOp(kPackSchema,kPackGuid,packMeta,nullptr);
    habana::KernelRegistry().add(kPackSchema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Pack>(d,t);});
    habana::custom_op::registerUserCustomOp(kProgramSchema,kProgramGuid,programMeta,nullptr);
    habana::KernelRegistry().add(kProgramSchema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Program>(d,t);});
    habana::custom_op::registerUserCustomOp(kPipelineSchema,kWeightGuid,[](const at::Stack& s){return habana::PartialOutputMetaDataVector{{at::kBFloat16,pipelineMeta(s)}};},nullptr);
    habana::KernelRegistry().add(kPipelineSchema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Pipeline>(d,t);});
    habana::custom_op::registerUserCustomOp(kRestoreSchema,kRestoreGuid,restoreMeta,nullptr);
    habana::KernelRegistry().add(kRestoreSchema,[](synDeviceId d,c10::ScalarType t){return std::make_shared<Restore>(d,t);});return true;
}();

template<bool Meta> std::tuple<at::Tensor,at::Tensor> control(
    const at::Tensor& ids,const at::Tensor& generation,int64_t experts) {
    at::Stack s{ids,generation,experts};auto metadata=controlMeta(s);
    if(Meta)return {at::empty(metadata[0].shape,ids.options()),at::empty(metadata[1].shape,ids.options().dtype(at::kShort))};
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kControlSchema);
    auto output=op.execute(s);return {output.at(0),output.at(1)};
}
template<bool Meta> std::tuple<at::Tensor,at::Tensor,at::Tensor> pack(
    const at::Tensor& x,const at::Tensor& sx,const at::Tensor& routing,const at::Tensor& metadata) {
    at::Stack s{x,sx,routing,metadata};auto m=packMeta(s);if(Meta)return {
        at::empty(m[0].shape,x.options().dtype(m[0].dtype)),
        at::empty(m[1].shape,x.options().dtype(m[1].dtype)),
        at::empty(m[2].shape,x.options().dtype(m[2].dtype))};
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPackSchema);
    auto outputs=op.execute(s);return {outputs.at(0),outputs.at(1),outputs.at(2)};
}
template<bool Meta> at::Tensor program(const at::Tensor& bank,const at::Tensor& spans,const at::Tensor& metadata) {
    at::Stack s{bank,spans,metadata};programMeta(s);if(Meta)return at::empty({65536},bank.options());
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kProgramSchema);return op.execute(s).at(0);
}
template<bool Meta> at::Tensor pipeline(const at::Tensor& packed,const at::Tensor& sx,const at::Tensor& routing,
    const at::Tensor& metadata,const at::Tensor& controls,const at::Tensor& q13,const at::Tensor& p13,
    const at::Tensor& q2,const at::Tensor& p2,const at::Tensor& lut,const at::Tensor& c13,const at::Tensor& c2,
    const at::Tensor& programTensor) {
    at::Stack s{packed,sx,routing,metadata,controls,q13,p13,q2,p2,lut,c13,c2,programTensor};auto shape=pipelineMeta(s);
    if(Meta)return at::empty(shape,packed.options().dtype(at::kBFloat16));
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kPipelineSchema);return op.execute(s).at(0);
}
template<bool Meta> at::Tensor restore(const at::Tensor& x,const at::Tensor& metadata,int64_t tokens) {
    at::Stack s{x,metadata,tokens};auto m=restoreMeta(s);if(Meta)return at::empty(m[0].shape,x.options());
    auto op=habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(kRestoreSchema);return op.execute(s).at(0);
}
}

TORCH_LIBRARY_FRAGMENT(custom_op,m){
    m.def("custom_deepseek_v41_unique_control_i32_gaudi2(Tensor ids, Tensor generation, int experts) -> (Tensor, Tensor)");
    m.def("custom_deepseek_v41_unique_pack_fp8_gaudi2(Tensor x, Tensor sx, Tensor routing, Tensor metadata) -> (Tensor, Tensor, Tensor)");
    m.def("custom_deepseek_v41_unique_program_i32_gaudi2(Tensor bank, Tensor spans, Tensor metadata) -> Tensor");
    m.def("custom_deepseek_v41_unique_expert_fp8_gaudi2(Tensor packed, Tensor sx, Tensor routing, Tensor metadata, Tensor controls, Tensor q13, Tensor p13, Tensor q2, Tensor p2, Tensor lut, Tensor c13, Tensor c2, Tensor program) -> Tensor");
    m.def("custom_deepseek_v41_unique_restore_bf16_gaudi2(Tensor x, Tensor metadata, int tokens) -> Tensor");}
TORCH_LIBRARY_IMPL(custom_op,HPU,m){
    m.impl("custom_deepseek_v41_unique_control_i32_gaudi2",control<false>);
    m.impl("custom_deepseek_v41_unique_pack_fp8_gaudi2",pack<false>);m.impl("custom_deepseek_v41_unique_program_i32_gaudi2",program<false>);
    m.impl("custom_deepseek_v41_unique_expert_fp8_gaudi2",pipeline<false>);m.impl("custom_deepseek_v41_unique_restore_bf16_gaudi2",restore<false>);}
TORCH_LIBRARY_IMPL(custom_op,Meta,m){
    m.impl("custom_deepseek_v41_unique_control_i32_gaudi2",control<true>);
    m.impl("custom_deepseek_v41_unique_pack_fp8_gaudi2",pack<true>);m.impl("custom_deepseek_v41_unique_program_i32_gaudi2",program<true>);
    m.impl("custom_deepseek_v41_unique_expert_fp8_gaudi2",pipeline<true>);m.impl("custom_deepseek_v41_unique_restore_bf16_gaudi2",restore<true>);}
