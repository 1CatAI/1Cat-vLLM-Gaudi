// SPDX-License-Identifier: Apache-2.0
// Untimed Synapse capability: GEMM -> future receive -> GEMM in one recipe.
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "hccl.h"
#include <dlfcn.h>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <thread>
#include <iostream>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
void network_check(hcclResult_t s,const char* where){if(s!=hcclSuccess)throw std::runtime_error(std::string(where)+": "+std::to_string(s));}
void check(synStatus s,const char* where){if(s!=synSuccess)throw std::runtime_error(std::string(where)+": "+std::to_string(s));}
struct Buffer {uint64_t address=0,bytes=0;};
struct Tensor {synTensor handle=nullptr;std::string name;std::vector<uint64_t> shape;Buffer memory;};
struct NativeSync {uint32_t longSoIndex;uint64_t targetValue;};
template<class F> F resolve(const char* name) {
    auto* address=dlsym(RTLD_DEFAULT,name);
    if(!address)throw std::runtime_error(std::string("Missing capability API ")+name);
    return reinterpret_cast<F>(address);
}
struct NativeReceive {
    using Create=synStatus(*)(void**,synStreamHandle);
    using Action=synStatus(*)(void*);
    using Segment=synStatus(*)(void*,uint64_t,const NativeSync*,uint8_t,NativeSync*);
    using HCreate=hcclResult_t(*)(const void*,void*,size_t,hcclDataType_t,hcclRedOp_t,hcclComm_t,void*,int,void**);
    using HAction=hcclResult_t(*)(void*);
    using HBatchCreate=hcclResult_t(*)(void* const*,size_t,void**);
    using HBatchReplay=hcclResult_t(*)(void*,const NativeSync*,size_t,NativeSync*);
    using EventInfo=synStatus(*)(synEventHandle,NativeSync*,uint32_t*);
    using EventBind=synStatus(*)(synEventHandle,const NativeSync*,uint64_t,uint64_t);
    using EventTemplateBind=synStatus(*)(synEventHandle,synEventHandle,const NativeSync*,uint64_t,uint64_t);
    using Callback=int(*)(void*,const NativeSync*,uint64_t,NativeSync*);
    using Prepare=synStatus(*)(void*,const uint32_t*,const uint32_t*,uint64_t,uint32_t,Callback,void*);
    using PlanReplay=synStatus(*)(void*,NativeSync*,uint64_t*,uint64_t);
    Create create=resolve<Create>("synNativeComputeGraphCreate");
    Action capture=resolve<Action>("synNativeComputeGraphBeginCapture");
    Action end=resolve<Action>("synNativeComputeGraphEndCapture");
    Action begin=resolve<Action>("synNativeComputeGraphBeginReplay");
    Segment segment=resolve<Segment>("synNativeComputeGraphReplaySegment");
    Action destroy=resolve<Action>("synNativeComputeGraphDestroy");
    HCreate hcreate=resolve<HCreate>("hcclTp4NativeGraphCreate");
    HAction hcapture=resolve<HAction>("hcclTp2NativeGraphCapture");
    HBatchCreate batch_create=resolve<HBatchCreate>("hcclTp2NativeBatchCreate");
    HBatchReplay batch_replay=resolve<HBatchReplay>("hcclTp2NativeBatchReplay");
    HAction batch_destroy=resolve<HAction>("hcclTp2NativeBatchDestroy");
    HAction hdestroy=resolve<HAction>("hcclTp2NativeGraphDestroy");
    EventInfo event=resolve<EventInfo>("dsv41ReceiveEventSyncInfo");
    EventBind bind=nullptr;
    EventTemplateBind bind_template=nullptr;
    Prepare prepare=resolve<Prepare>("synNativeComputeGraphPreparePlanV2");
    PlanReplay replay=resolve<PlanReplay>("synNativeComputeGraphReplayPlan");
};
struct FuturePublish {
    NativeReceive* api=nullptr;void* batch=nullptr;
    synStreamHandle nic=nullptr,copy=nullptr;synEventHandle event=nullptr;
    uint64_t delta=0,epoch=0,flag=0;uint64_t callback_count=0;
    bool late_copy=false;
    synEventHandle seed=nullptr;
    static int replay(void* context,const NativeSync* producers,uint64_t count,NativeSync* completions) {
        auto& self=*static_cast<FuturePublish*>(context);
        try {
            if(count!=(self.late_copy?2u:1u)||producers[0].targetValue<=self.delta)
                throw std::runtime_error("Future point coverage");
            NativeSync ready[2]={{producers[0].longSoIndex,producers[0].targetValue-self.delta},{}};
            if(self.late_copy)ready[1]=producers[1];
            network_check(self.api->batch_replay(self.batch,ready,count,completions),"joint NIC batch");
            NativeSync actual{};uint32_t onHcl=0;
            if(self.seed){
                check(self.api->bind_template(self.event,self.seed,completions,
                    completions[0].targetValue-1,completions[count-1].targetValue),"cold-template first point binding");
            } else {
                check(synEventRecord(self.event,self.nic),"joint NIC completion record");
                check(self.api->event(self.event,&actual,&onHcl),"joint completion metadata");
                if(!onHcl||actual.longSoIndex!=completions[count-1].longSoIndex||
                    actual.targetValue!=completions[count-1].targetValue)
                    throw std::runtime_error("Joint NIC completion mismatch");
            }
            if(self.late_copy&&!self.seed){
                check(self.api->bind(self.event,completions,completions[0].targetValue-1,
                    completions[1].targetValue),"bind first point within batch");
            }
            if(self.late_copy||self.seed){
                check(self.api->event(self.event,&actual,&onHcl),"bound first completion metadata");
                if(!onHcl||actual.longSoIndex!=completions[0].longSoIndex||actual.targetValue!=completions[0].targetValue)
                    throw std::runtime_error("First point binding mismatch");
            }
            check(synStreamWaitEvent(self.copy,self.event,0),"joint NIC flag wait");
            check(synMemCopyAsync(self.copy,self.epoch,4,self.flag,DRAM_TO_DRAM),"joint flag publication");
            ++self.callback_count;return 0;
        } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
};
std::vector<char> read(const std::string& path,uint64_t size){
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Missing fixture "+path);
    std::vector<char> b(size);f.read(b.data(),size);if(uint64_t(f.gcount())!=size)throw std::runtime_error("Truncated "+path);return b;
}
int main(int argc,char** argv){
    if(argc!=3)return 2;
    const bool network=std::getenv("DSV41_RECEIVE_NETWORK");
    const bool native=std::getenv("DSV41_RECEIVE_NATIVE_REPLAY");
    const bool joint=std::getenv("DSV41_RECEIVE_JOINT_PLAN");
    const bool production=std::getenv("DSV41_RECEIVE_PRODUCTION_POST");
    const bool late_copy=std::getenv("DSV41_RECEIVE_LATE_POINT_COPY");
    const bool device_epoch=std::getenv("DSV41_RECEIVE_DEVICE_EPOCH");
    const bool cold_template=std::getenv("DSV41_RECEIVE_COLD_TEMPLATE");
    const int rank=network?std::atoi(std::getenv("RANK")):0;
    const int group=network?std::atoi(std::getenv("WORLD_SIZE")):4;
    if(network && (group!=2 && group!=4))return 2;
    if(native && (!network || group!=4))return 2;
    if(joint&&!native)return 2;
    if(production&&!joint)return 2;
    if(late_copy&&!joint)return 2;
    if(cold_template&&!joint)return 2;
    if(device_epoch&&(!joint||!cold_template))return 2;
    uint64_t peer_capacity=128;
    while(2*peer_capacity*uint64_t(group)*6*5120*2<=(uint64_t(48)<<20))peer_capacity*=2;
    const std::string data=argv[1],common_run=argv[2];
    const std::string run=network?common_run+"/rank"+std::to_string(rank):common_run;
    std::filesystem::create_directories(run);
    hcclComm_t communicator=nullptr;
    synDeviceId device=0;synGraphHandle graph=nullptr,consumer_graph=nullptr;
    synRecipeHandle recipe=nullptr,consumer_recipe=nullptr;
    synStreamHandle compute=nullptr,publish=nullptr,copy=nullptr;synEventHandle produced=nullptr;
    std::vector<Buffer> allocated;std::vector<synSectionHandle> sections;std::vector<Tensor> tensors;
    bool acquired=false;
    void* native_graph=nullptr;void* native_nic=nullptr;void* native_batch=nullptr;void* native_late_nic=nullptr;
    synEventHandle capture_done=nullptr,nic_done=nullptr,nic_seed=nullptr;
    std::unique_ptr<NativeReceive> api;
    try {
        std::cerr<<"stage initialize\n";
        check(synInitialize(),"initialize");
        if(native){api=std::make_unique<NativeReceive>();
            if(late_copy)api->bind=resolve<NativeReceive::EventBind>("dsv41ReceiveBindBatchEvent");
            if(cold_template)api->bind_template=resolve<NativeReceive::EventTemplateBind>("dsv41ReceiveBindEventFromTemplate");}
        std::cerr<<"stage graph\n";
        check(synGraphCreate(&graph,synDeviceGaudi2),"graph create");
        auto active_graph=graph;
        auto tensor=[&](const char* name,synDataType type,std::vector<uint64_t> shape,bool persistent){
            Tensor t;t.name=name;t.shape=shape;synTensorDescriptor d{};d.m_dataType=type;
            d.m_dims=shape.size();d.m_name=name;
            for(unsigned i=0;i<shape.size();++i)d.m_sizes[i]=d.m_minSizes[i]=shape[i];
            synSectionHandle section=nullptr;
            if(persistent){check(synSectionCreate(&section,0,active_graph),"section");
                check(synSectionSetPersistent(section,true),"persistent");sections.push_back(section);}
            check(synTensorCreate(&t.handle,&d,section,0),"tensor create");
            t.memory.bytes=std::accumulate(shape.begin(),shape.end(),uint64_t(1),std::multiplies<uint64_t>())*
                (type==syn_type_fp8_143?1:type==syn_type_int32||type==syn_type_single?4:2);
            tensors.push_back(t);return t.handle;
        };
        auto x=tensor("input",syn_type_bf16,{2048,6},true);
        auto w=tensor("producer_weight",syn_type_bf16,{2048,5120},true);
        auto local=tensor("partial",syn_type_bf16,{5120,6},true);
        check(synTensorSetExternal(local,true),"producer external");
        auto peer=tensor("peer_table",syn_type_bf16,{5120,6,uint64_t(group),peer_capacity,2},true);
        auto flags=tensor("ready_flags",syn_type_int32,{1u<<24},true);
        auto epoch=device_epoch?flags:tensor("epoch",syn_type_int32,{1},true);
        auto reduced=tensor("reduced",syn_type_bf16,{5120,6},joint&&!production);
        auto status=tensor("status",syn_type_int32,{40,6},true);
        synTensor norm_output=nullptr;
        if(production){
            auto residual=tensor("residual",syn_type_bf16,{5120,4,6},true);
            auto post=tensor("post",syn_type_single,{4,6},true);
            auto comb=tensor("comb",syn_type_single,{4,4,6},true);
            auto pre=tensor("pre",syn_type_single,{4,6},true);
            auto residual_out=tensor("residual_out",syn_type_bf16,{5120,4,6},true);
            auto collapsed=tensor("collapsed",syn_type_bf16,{5120,6},false);
            auto norm_weight=tensor("norm_weight",syn_type_bf16,{5120},true);
            norm_output=tensor("norm_output",syn_type_bf16,{5120,6},true);
            auto quantized=tensor("quantized",syn_type_fp8_143,{5120,6},true);
            auto scales=tensor("scales",syn_type_single,{1,6},true);
            synTensor post_in[]={reduced,residual,post,comb,pre},post_out[]={residual_out,collapsed};
            check(synNodeCreate(graph,post_in,post_out,5,2,nullptr,0,
                "custom_deepseek_v41_mhc_post_collapse_gaudi2","production_post",nullptr,nullptr),"post consumer");
            synTensor norm_in[]={collapsed,norm_weight},norm_out[]={norm_output,quantized,scales};
            const float norm_params[]={1.0e-20f,1.0f/5120.0f};
            check(synNodeCreate(graph,norm_in,norm_out,2,3,const_cast<float*>(norm_params),sizeof(norm_params),
                "custom_deepseek_v41_ffn_norm_quant_gaudi2","production_norm",nullptr,nullptr),"norm consumer");
        }
        synTensor consumer_input=reduced;
        if(joint){check(synGraphCreate(&consumer_graph,synDeviceGaudi2),"consumer graph create");
            active_graph=consumer_graph;consumer_input=tensor("consumer_input",syn_type_bf16,{5120,6},true);}
        auto nextw=tensor("consumer_weight",production?syn_type_single:syn_type_bf16,
                         {5120,production?384u:1280u},true);
        auto output=tensor("output",production?syn_type_single:syn_type_bf16,{production?384u:1280u,6},true);
        if(production){auto float_input=tensor("float_input",syn_type_single,{5120,6},false);
            check(synNodeCreate(consumer_graph,&consumer_input,&float_input,1,1,nullptr,0,
                "cast_bf16_to_f32","router_input",nullptr,nullptr),"router input cast");
            consumer_input=float_input;}
        synGEMMParams gemm{false,true};
        synTensor producer_input=x;
        if(device_epoch){
            auto flat=tensor("epoch_input",syn_type_bf16,{2048*6,1},false);
            auto moved=tensor("epoch_output",syn_type_bf16,{2048*6,1},false);
            auto shaped=tensor("epoch_shaped",syn_type_bf16,{2048,6},false);
            check(synNodeCreate(graph,&x,&flat,1,1,nullptr,0,"reshape","epoch_flat",nullptr,nullptr),"epoch flatten");
            synTensor epoch_inputs[]={flat,flags};
            check(synNodeCreate(graph,epoch_inputs,&moved,2,1,nullptr,0,
                "custom_deepseek_v41_future_epoch_gaudi2","transport_generation",nullptr,nullptr),"epoch advance");
            check(synNodeCreate(graph,&moved,&shaped,1,1,nullptr,0,"reshape","epoch_shape",nullptr,nullptr),"epoch shape");
            producer_input=shaped;
        }
        synTensor producer[]={producer_input,w};
        check(synNodeCreate(graph,producer,&local,2,1,&gemm,sizeof(gemm),"gemm","producer",nullptr,nullptr),"producer node");
        synTensor received[]={local,peer,flags,epoch},received_out[]={reduced,status};const int params[]={0,rank,65536};
        check(synNodeCreate(graph,received,received_out,4,2,const_cast<int*>(params),sizeof(params),
            "custom_deepseek_v41_bounded_peer_receive_gaudi2","bounded_receive",nullptr,nullptr),"receive node");
        synTensor consumer[]={consumer_input,nextw};
        check(synNodeCreate(joint?consumer_graph:graph,consumer,&output,2,1,&gemm,sizeof(gemm),"gemm","consumer",nullptr,nullptr),"consumer node");
        std::cerr<<"stage compile\n";
        check(synGraphCompile(&recipe,graph,(run+"/receive").c_str(),nullptr),"compile");
        check(synRecipeSerialize(recipe,(run+"/receive.recipe").c_str()),"serialize recipe");
        if(joint){check(synGraphCompile(&consumer_recipe,consumer_graph,(run+"/consumer").c_str(),nullptr),"consumer compile");
            check(synRecipeSerialize(consumer_recipe,(run+"/consumer.recipe").c_str()),"consumer serialize");}
        const char* module=std::getenv("HLS_MODULE_ID");
        if(!module||std::strchr(module,','))throw std::runtime_error("One leased scalar module required");
        check(synDeviceAcquireByModuleId(&device,std::atoi(module)),"acquire");acquired=true;
        for(auto* stream:{&compute,&publish,&copy})check(synStreamCreateGeneric(stream,device,0),"stream create");
        // Default affinity is selected per first job type, not per handle.
        // A first compute, D2D and H2D job can all select affinity zero.
        // Future publication must use a distinct physical D2D queue from the
        // compiler's own recipe transfers, rather than just a new handle.
        if(std::getenv("DSV41_RECEIVE_SEPARATE_AFFINITY")) {
            check(synStreamSetAffinity(device,compute,1),"compute affinity");
            check(synStreamSetAffinity(device,publish,2),"publisher affinity");
            check(synStreamSetAffinity(device,copy,4),"copy affinity");
        }
        if(network) {
            hcclUniqueId identity{};
            const std::string identity_path=common_run+"/hccl-id.bin";
            if(rank==0) {
                network_check(hcclGetUniqueId(&identity),"unique id");
                std::ofstream id(identity_path+".tmp",std::ios::binary);
                id.write(reinterpret_cast<char*>(&identity),sizeof(identity));id.close();
                std::filesystem::rename(identity_path+".tmp",identity_path);
            } else {
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!std::filesystem::exists(identity_path)) {
                    if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("Unique id deadline");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                std::ifstream id(identity_path,std::ios::binary);
                id.read(reinterpret_cast<char*>(&identity),sizeof(identity));
                if(id.gcount()!=sizeof(identity))throw std::runtime_error("Unique id truncated");
            }
            network_check(hcclCommInitRank(&communicator,group,identity,rank),"communicator init");
        }
        auto host_barrier=[&](const std::string& label,bool passed=true) {
            if(!network)return;
            const auto stem=common_run+"/"+label;
            const auto own=stem+"-rank"+std::to_string(rank);
            {std::ofstream ready(own+".tmp");ready<<(passed?1:0);}
            std::filesystem::rename(own+".tmp",own);
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            bool complete=false;
            while(!complete) {
                complete=true;
                for(int r=0;r<group;++r)complete&=std::filesystem::exists(stem+"-rank"+std::to_string(r));
                if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("CPU barrier deadline");
                if(!complete)std::this_thread::yield();
            }
            for(int r=0;r<group;++r){int value=0;std::ifstream ready(stem+"-rank"+std::to_string(r));
                ready>>value;if(!value)throw std::runtime_error("A peer failed the capability contract");}
        };
        auto allocate=[&](uint64_t bytes){Buffer b{0,bytes};check(synDeviceMalloc(device,bytes,0,0,&b.address),"malloc");
            allocated.push_back(b);return b;};
        auto upload=[&](const std::vector<char>& h,Buffer d,synStreamHandle stream){
            check(synHostMap(device,h.size(),h.data()),"host map");
            check(synMemCopyAsync(stream,uint64_t(h.data()),h.size(),d.address,HOST_TO_DRAM),"upload");
            check(synStreamSynchronize(stream),"upload drain");check(synHostUnmap(device,h.data()),"host unmap");};
        std::vector<synLaunchTensorInfo> launch,consumer_launch;
        std::vector<const char*> names;
        auto consumer_tensor=[&](const Tensor& t){return joint&&(t.name=="consumer_input"||t.name=="consumer_weight"||t.name=="output");};
        auto transient_tensor=[&](const Tensor& t){return (t.name=="reduced"&&(!joint||production))||
            t.name=="collapsed"||t.name=="float_input"||t.name.rfind("epoch_",0)==0;};
        for(auto& t:tensors){if(transient_tensor(t))continue;t.memory=allocate(t.memory.bytes);
            if(!consumer_tensor(t))names.push_back(t.name.c_str());}
        std::vector<uint64_t> ids(names.size());check(synTensorRetrieveIds(recipe,names.data(),ids.data(),ids.size()),"tensor ids");
        unsigned index=0,local_index=0,epoch_index=0;
        for(auto& t:tensors){if(transient_tensor(t)||consumer_tensor(t))continue;synLaunchTensorInfo info{};
            info.tensorName=t.name.c_str();info.tensorType=DATA_TENSOR;info.tensorId=ids[index];info.pTensorAddress=t.memory.address;
            for(unsigned i=0;i<t.shape.size();++i)info.tensorSize[i]=t.shape[i];
            if(t.name=="partial")local_index=index;
            if(t.name=="epoch")epoch_index=index;
            launch.push_back(info);++index;}
        auto find=[&](const char* name)->Tensor&{for(auto& t:tensors)if(t.name==name)return t;throw std::runtime_error(name);};
        if(joint){std::vector<const char*> cnames={"consumer_input","consumer_weight","output"};
            std::vector<uint64_t> cids(cnames.size());check(synTensorRetrieveIds(consumer_recipe,cnames.data(),cids.data(),cids.size()),"consumer tensor ids");
            for(unsigned i=0;i<cnames.size();++i){auto& t=find(cnames[i]);synLaunchTensorInfo info{};
                info.tensorName=t.name.c_str();info.tensorType=DATA_TENSOR;info.tensorId=cids[i];
                info.pTensorAddress=i==0?find(production?"norm_output":"reduced").memory.address:t.memory.address;
                for(unsigned d=0;d<t.shape.size();++d){info.tensorSize[d]=t.shape[d];}
                consumer_launch.push_back(info);}}
        uint64_t workspace_size=0;synRecipeAttribute attr=RECIPE_ATTRIBUTE_WORKSPACE_SIZE;
        check(synRecipeGetAttribute(&workspace_size,&attr,1,recipe),"workspace");auto workspace=allocate(workspace_size?workspace_size:64);
        Buffer consumer_workspace{};
        if(joint){check(synRecipeGetAttribute(&workspace_size,&attr,1,consumer_recipe),"consumer workspace");
            consumer_workspace=allocate(workspace_size?workspace_size:64);}
        upload(read(data+"/producer-weight.bin",find("producer_weight").memory.bytes),find("producer_weight").memory,copy);
        upload(read(data+"/consumer-weight.bin",find("consumer_weight").memory.bytes),find("consumer_weight").memory,copy);
        if(production)upload(read(data+"/norm-weight.bin",find("norm_weight").memory.bytes),find("norm_weight").memory,copy);
        auto epoch_values=allocate(16);std::vector<char> epoch_host(16);
        const int values[]={0,2,3,4};std::memcpy(epoch_host.data(),values,16);upload(epoch_host,epoch_values,copy);
        auto source_peers=allocate(4*6*5120*2);
        check(synEventCreate(&produced,device,0),"event");
        if(native){check(synEventCreate(&capture_done,device,0),"capture end event");
            check(synEventCreate(&nic_done,device,0),"NIC event");}
        if(cold_template)check(synEventCreate(&nic_seed,device,0),"cold NIC seed event");
        FuturePublish future_publish{};
        uint64_t steady_ready_delta=0;int device_generation=0;
        if(device_epoch){check(synMemsetD32Async(find("ready_flags").memory.address,0,1u<<24,copy),"init transport counter");
            check(synStreamSynchronize(copy),"counter init drain");}
        std::ofstream result(run+"/capability.json");result<<"{\"timed\":false,\"cases\":[";
        for(unsigned c=0;c<3;++c){
            upload(read(data+"/input-"+std::to_string(c)+".bin",find("input").memory.bytes),find("input").memory,copy);
            if(production)for(const char* name:{"residual","post","comb","pre"}){
                auto& t=find(name);upload(read(data+"/"+name+"-"+std::to_string(c)+".bin",t.memory.bytes),t.memory,copy);}
            if(!network)upload(read(data+"/peers-"+std::to_string(c)+".bin",source_peers.bytes),source_peers,copy);
            const uint64_t parity=device_epoch?0:(c+2)&1,slot=parity;
            const uint64_t field=find("ready_flags").memory.address+slot*4;
            if(!device_epoch)launch[epoch_index].pTensorAddress=epoch_values.address+(c+1)*4;
            const uint64_t counter_address=find("ready_flags").memory.address+((1u<<24)-1)*4ull;
            auto expected_epoch=[&]{return device_epoch?counter_address:launch[epoch_index].pTensorAddress;};
            auto prefill_epoch=[&](synStreamHandle stream){
                if(!device_epoch){check(synMemCopyAsync(stream,expected_epoch(),4,field,DRAM_TO_DRAM),"prefill epoch");return;}
                const int next=device_generation+2;
                std::vector<char> data_bytes(4);std::memcpy(data_bytes.data(),&next,4);
                upload(data_bytes,Buffer{field,4},stream);
            };
            uint64_t ready_delta=0;
            std::vector<char> reference;
            if(network) {
                // Establish exact producer payload on every card before the
                // prefilled NIC reference. The warm result is not timed.
                check(synMemsetD32Async(find("peer_table").memory.address,0,
                    find("peer_table").memory.bytes/4,publish),"warm zeros");
                prefill_epoch(publish);
                check(synStreamSynchronize(publish),"warm inputs");
                check(synEventMapTensor(&produced,1,&launch[local_index],recipe),"warm partial map");
                check(synLaunchWithExternalEvents(compute,launch.data(),launch.size(),workspace.address,recipe,
                    &produced,1,0),"warm local");
                if(joint)check(synLaunch(compute,consumer_launch.data(),consumer_launch.size(),consumer_workspace.address,
                    consumer_recipe,0),"warm consumer");
                check(synStreamSynchronize(compute),"warm producer drain");
                if(device_epoch)device_generation+=2;
                host_barrier("warm-"+std::to_string(c));
            }
            for(unsigned future=0;future<2;++future){
                // Native NIC replay bypasses Stream::addJob. Keep its stream
                // on the NETWORK queue; table initialization and flag DMA
                // belong to the independent copy queue.
                const auto initialize_stream=native?copy:publish;
                check(synMemsetD32Async(find("peer_table").memory.address,0xbeccbecc,
                    find("peer_table").memory.bytes/4,initialize_stream),"stale peers");
                if(!device_epoch)check(synMemsetD32Async(field,0,1,initialize_stream),"zero flag");
                check(synStreamSynchronize(initialize_stream),"initialize tables");
                auto publish_data=[&]{
                    const uint64_t bytes=6*5120*2;
                    if(network)network_check(hcclGroupStart(),"peer group start");
                    for(int r=0;r<group;++r){if(r==rank)continue;
                        const auto destination=find("peer_table").memory.address+((parity*peer_capacity*group+r)*bytes);
                        if(network) {
                            network_check(hcclSend(reinterpret_cast<void*>(find("partial").memory.address),
                                6*5120,hcclBfloat16,r,communicator,publish),"peer send");
                            network_check(hcclRecv(reinterpret_cast<void*>(destination),6*5120,
                                hcclBfloat16,r,communicator,publish),"peer receive");
                        } else check(synMemCopyAsync(publish,source_peers.address+r*bytes,bytes,
                            destination,DRAM_TO_DRAM),"peer copy");
                    }
                    if(network)network_check(hcclGroupEnd(),"peer group end");
                    prefill_epoch(publish);
                };
                if(!future){publish_data();check(synStreamSynchronize(publish),"prefilled drain");}
                host_barrier("launch-"+std::to_string(c)+"-"+std::to_string(future));
                check(synEventMapTensor(&produced,1,&launch[local_index],recipe),"map partial");
                if(native&&!future&&(!device_epoch||c==0)){check(api->create(&native_graph,compute),"native create");
                    check(api->capture(native_graph),"native capture");}
                const auto launch_begin=std::chrono::steady_clock::now();
                NativeSync native_end{};
                if(joint&&future){uint64_t statistics[12]{};
                    check(api->replay(native_graph,&native_end,statistics,12),"joint native plan replay");
                    if(statistics[0]!=(device_epoch?c+1:1)||future_publish.callback_count!=(device_epoch?c+1:1))throw std::runtime_error("Joint replay did not execute one callback");}
                else if(native&&future){check(api->begin(native_graph),"native begin");
                    check(api->segment(native_graph,0,nullptr,1,&native_end),"native compute replay");}
                else check(synLaunchWithExternalEvents(compute,launch.data(),launch.size(),workspace.address,recipe,
                    &produced,1,0),"launch");
                if(native&&!future&&(!device_epoch||c==0)){
                    check(synEventRecord(capture_done,compute),"capture completion record");
                    NativeSync partial{},end{};uint32_t partial_hcl=0,end_hcl=0;
                    check(api->event(produced,&partial,&partial_hcl),"producer external metadata");
                    check(api->event(capture_done,&end,&end_hcl),"capture completion metadata");
                    if(partial_hcl||end_hcl||partial.longSoIndex!=end.longSoIndex||end.targetValue<=partial.targetValue)
                        throw std::runtime_error("External signal must precede compute completion");
                    ready_delta=end.targetValue-partial.targetValue;steady_ready_delta=ready_delta;
                    if(joint)check(synLaunch(compute,consumer_launch.data(),consumer_launch.size(),consumer_workspace.address,
                        consumer_recipe,0),"capture consumer");
                    check(api->end(native_graph),"native end capture");}
                if(device_epoch&&c>0&&!future)check(synLaunch(compute,consumer_launch.data(),consumer_launch.size(),
                    consumer_workspace.address,consumer_recipe,0),"steady reference consumer");
                const auto launch_end=std::chrono::steady_clock::now();
                const auto producer_query=synEventQuery(produced);
                const auto publish_begin=std::chrono::steady_clock::now();
                if(future&&!joint){
                    if(native){
                        if(native_end.targetValue<=ready_delta)throw std::runtime_error("Native producer signal underflow");
                        const NativeSync ready{native_end.longSoIndex,native_end.targetValue-ready_delta};
                        NativeSync received{},event_info{};uint32_t on_hcl=0;
                        network_check(api->batch_replay(native_batch,&ready,1,&received),"native NIC batch replay");
                        check(synEventRecord(nic_done,publish),"native NIC completion record");
                        check(api->event(nic_done,&event_info,&on_hcl),"native NIC event metadata");
                        std::cerr<<"native NIC completed "<<received.longSoIndex<<':'<<received.targetValue
                            <<" event "<<event_info.longSoIndex<<':'<<event_info.targetValue<<" HCL="<<on_hcl
                            <<" external delta="<<ready_delta<<'\n';
                        if(!on_hcl||event_info.longSoIndex!=received.longSoIndex||event_info.targetValue!=received.targetValue)
                            throw std::runtime_error("Native NIC completion event does not match replay");
                        check(synStreamWaitEvent(copy,nic_done,0),"native NIC -> flag dependency");
                        check(synMemCopyAsync(copy,launch[epoch_index].pTensorAddress,4,field,DRAM_TO_DRAM),"native epoch publish");
                    } else {
                      if(!std::getenv("DSV41_RECEIVE_DIAGNOSTIC_NO_WAIT"))
                        check(synStreamWaitEvent(publish,produced,0),"producer dependency");
                      publish_data();
                    }
                }
                const auto publish_end=std::chrono::steady_clock::now();
                uint64_t compute_affinity=0,publish_affinity=0;
                check(synStreamGetAffinity(device,compute,&compute_affinity),"compute affinity query");
                check(synStreamGetAffinity(device,publish,&publish_affinity),"publisher affinity query");
                std::cerr<<"affinity compute="<<compute_affinity<<" publish="<<publish_affinity<<'\n';
                check(synStreamSynchronize(compute),"compute drain");
                if(device_epoch)device_generation+=2;
                check(synStreamSynchronize(publish),"publisher drain");
                if(native)check(synStreamSynchronize(copy),"flag publisher drain");
                if(native&&!future&&(!device_epoch||c==0)){
                    const uint64_t bytes=find("partial").memory.bytes;
                    const uint64_t destination=find("peer_table").memory.address+parity*peer_capacity*group*bytes;
                    network_check(api->hcreate(reinterpret_cast<void*>(find("partial").memory.address),
                        reinterpret_cast<void*>(destination),bytes/2,hcclBfloat16,hcclSum,communicator,publish,2,&native_nic),
                        "native NIC create");
                    network_check(api->hcapture(native_nic),"native NIC capture");
                    check(synStreamSynchronize(publish),"native NIC capture drain");
                    if(late_copy){
                        network_check(api->hcreate(reinterpret_cast<void*>(find("partial").memory.address),
                            reinterpret_cast<void*>(destination+group*bytes),bytes/2,hcclBfloat16,hcclSum,
                            communicator,publish,2,&native_late_nic),"late point NIC create");
                        network_check(api->hcapture(native_late_nic),"late point NIC capture");
                        check(synStreamSynchronize(publish),"late point capture drain");
                    }
                    void* native_points[]={native_nic,native_late_nic};
                    network_check(api->batch_create(native_points,late_copy?2:1,&native_batch),"native NIC batch create");
                    if(joint){NativeSync nic_info{};uint32_t on_hcl=0;
                        check(synEventRecord(nic_done,publish),"prepare NIC event");
                        check(api->event(nic_done,&nic_info,&on_hcl),"prepare NIC metadata");
                        if(!on_hcl)throw std::runtime_error("Joint plan requires network queue");
                        if(cold_template)check(synEventRecord(nic_seed,publish),"cold NIC seed record");
                        future_publish={api.get(),native_batch,publish,copy,nic_done,steady_ready_delta,
                            expected_epoch(),field,0,late_copy,nic_seed};
                        const uint32_t producer_segments[]={0,0},consumer_segments[]={1,1};
                        check(api->prepare(native_graph,producer_segments,consumer_segments,late_copy?2:1,nic_info.longSoIndex,
                            FuturePublish::replay,&future_publish),"prepare joint plan");}
                }
                std::vector<char> answer(find("output").memory.bytes),states(find("status").memory.bytes);
                auto readback=[&](const char* name,std::vector<char>& h){
                    check(synHostMap(device,h.size(),h.data()),"map readback");
                    check(synMemCopyAsync(copy,find(name).memory.address,h.size(),uint64_t(h.data()),DRAM_TO_HOST),"readback");
                    check(synStreamSynchronize(copy),"readback drain");check(synHostUnmap(device,h.data()),"unmap readback");};
                readback("output",answer);readback("status",states);
                if(production)for(const char* name:{"residual_out","norm_output","quantized","scales"}) {
                    std::vector<char> intermediate(find(name).memory.bytes);readback(name,intermediate);
                    answer.insert(answer.end(),intermediate.begin(),intermediate.end());
                }
                unsigned timeout=0,max_spins=0;for(unsigned i=0;i<states.size()/4;++i){int value;std::memcpy(&value,states.data()+i*4,4);timeout+=value<0;max_spins=std::max(max_spins,unsigned(std::abs(value)));}
                int final_flag=0;
                check(synHostMap(device,4,&final_flag),"map final flag");
                check(synMemCopyAsync(copy,field,4,uint64_t(&final_flag),DRAM_TO_HOST),"final flag read");
                check(synStreamSynchronize(copy),"final flag drain");
                check(synHostUnmap(device,&final_flag),"unmap final flag");
                std::ofstream status_file(run+"/tiles-"+std::to_string(c)+"-"+std::to_string(future)+".bin",std::ios::binary);
                status_file.write(states.data(),states.size());
                const auto host_ms=[](auto first,auto last){return std::chrono::duration<double,std::milli>(last-first).count();};
                if(!future)reference=answer;
                const bool equal=!future||reference==answer;
                if(c||future)result<<',';
                result<<"{\"case\":"<<c<<",\"future\":"<<(future?"true":"false")
                    <<",\"launch_host_ms\":"<<host_ms(launch_begin,launch_end)<<",\"publish_enqueue_host_ms\":"<<host_ms(publish_begin,publish_end)<<",\"producer_query_status\":"<<int(producer_query)<<",\"final_flag\":"<<final_flag<<",\"compute_affinity\":"<<compute_affinity<<",\"publish_affinity\":"<<publish_affinity<<",\"device_generation\":"<<device_generation<<",\"max_spins\":"<<max_spins<<",\"timeout_tiles\":"<<timeout<<",\"output_equal\":"<<(equal?"true":"false")<<'}';result.flush();
                host_barrier("checked-"+std::to_string(c)+"-"+std::to_string(future),!timeout&&equal);
                if(timeout||!equal)throw std::runtime_error("Future payload contract or bounded wait failed");
            }
            if(native&&(!device_epoch||c==2)){network_check(api->batch_destroy(native_batch),"native NIC batch destroy");native_batch=nullptr;
                if(native_late_nic){network_check(api->hdestroy(native_late_nic),"late NIC destroy");native_late_nic=nullptr;}
                network_check(api->hdestroy(native_nic),"native NIC destroy");native_nic=nullptr;
                check(api->destroy(native_graph),"native compute destroy");native_graph=nullptr;}
        }
        result<<"],\"passed\":true,\"gain_credit_ms\":0}\n";result.close();
        if(communicator){network_check(hcclCommDestroy(communicator),"communicator destroy");communicator=nullptr;}
        check(synEventDestroy(produced),"destroy event");produced=nullptr;
        if(capture_done){check(synEventDestroy(capture_done),"capture event destroy");capture_done=nullptr;}
        if(nic_done){check(synEventDestroy(nic_done),"NIC event destroy");nic_done=nullptr;}
        if(nic_seed){check(synEventDestroy(nic_seed),"NIC seed destroy");nic_seed=nullptr;}
        for(auto stream:{compute,publish,copy})check(synStreamDestroy(stream),"destroy stream");
        compute=publish=copy=nullptr;
        for(auto b:allocated)check(synDeviceFree(device,b.address,0),"free");
        allocated.clear();
        check(synDeviceRelease(device),"release");acquired=false;
        check(synRecipeDestroy(recipe),"recipe destroy");recipe=nullptr;
        if(consumer_recipe){check(synRecipeDestroy(consumer_recipe),"consumer recipe destroy");consumer_recipe=nullptr;}
        check(synGraphDestroy(graph),"graph destroy");graph=nullptr;
        if(consumer_graph){check(synGraphDestroy(consumer_graph),"consumer graph destroy");consumer_graph=nullptr;}
        for(auto s:sections)check(synSectionDestroy(s),"section destroy");
        check(synDestroy(),"destroy");std::cout<<"Capability complete; no timing or performance credit\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';
        // Complete the diagnostic array on early failure; preserve raw cases.
        {
            std::ifstream existing(run+"/capability.json");
            const std::string contents((std::istreambuf_iterator<char>(existing)),{});
            if(!contents.empty() && contents.back()!='\n') {
                std::ofstream failed(run+"/capability.json",std::ios::app);
                failed<<"],\"passed\":false,\"gain_credit_ms\":0}\n";
            }
        }
        std::ofstream error(run+"/failure.json");error<<"{\"failed\":true,\"reason\":\""<<e.what()<<"\",\"timed\":false}\n";
        const int exit_code=acquired?3:2;
        if(acquired) {
            for(auto stream:{compute,publish,copy})if(stream)synStreamSynchronize(stream);
            if(native_batch)api->batch_destroy(native_batch);
            if(native_late_nic)api->hdestroy(native_late_nic);
            if(native_nic)api->hdestroy(native_nic);
            if(native_graph)api->destroy(native_graph);
            if(capture_done)synEventDestroy(capture_done);
            if(nic_done)synEventDestroy(nic_done);
            if(nic_seed)synEventDestroy(nic_seed);
            if(produced)synEventDestroy(produced);
            for(auto stream:{compute,publish,copy})if(stream)synStreamDestroy(stream);
            for(auto b:allocated)synDeviceFree(device,b.address,0);
            if(communicator){hcclCommAbort(communicator);communicator=nullptr;}
            synDeviceRelease(device);
        }
        if(recipe)synRecipeDestroy(recipe);
        if(consumer_recipe)synRecipeDestroy(consumer_recipe);
        if(graph)synGraphDestroy(graph);
        if(consumer_graph)synGraphDestroy(consumer_graph);
        for(auto section:sections)synSectionDestroy(section);
        synDestroy();
        return exit_code;
    }
}
