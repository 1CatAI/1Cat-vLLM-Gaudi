// SPDX-License-Identifier: Apache-2.0
// Isolated producer/consumer native-plan validation. No default C1 registration.
#include <torch/extension.h>
#include <dlfcn.h>
#include <hccl.h>
#include <habanalabs/synapse_api.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/helpers/generic_resource_holder.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_eager/eager_pipeline_utils.h"
#include "python_packages/habana_frameworks/torch/distributed/hccl/process_group_eager_hccl.hpp"
#include "dsv41_future_receive_publisher.h"

namespace {
struct Sync { uint32_t longSoIndex=0;uint64_t targetValue=0; };
struct Info {
    int state=0;uint64_t segments=0,replays=0;uint32_t completion=0;uint64_t target=0;
    uint64_t global_bytes=0,arc_bytes=0;
};
template<class T>T symbol(const char* name){auto result=reinterpret_cast<T>(dlsym(RTLD_DEFAULT,name));
    TORCH_CHECK(result,"Missing isolated native API: ",name);return result;}
void check(synStatus s){TORCH_CHECK(s==synSuccess,"Future native compute failure: ",s);}
void check(hcclResult_t s){TORCH_CHECK(s==hcclSuccess,"Future native NIC failure: ",s);}
class Plan:public std::enable_shared_from_this<Plan> {
    using Publisher=Dsv41FutureReceivePublisher<Sync>;
    using Create=synStatus(*)(void**,synStreamHandle);
    using Action=synStatus(*)(void*);
    using Get=synStatus(*)(void*,Info*);
    using Prepare=synStatus(*)(void*,const uint32_t*,const uint32_t*,uint64_t,uint32_t,
        int(*)(void*,const Sync*,uint64_t,Sync*),void*);
    using Replay=synStatus(*)(void*,Sync*,uint64_t*,uint64_t);
    using HCreate=hcclResult_t(*)(const void*,void*,size_t,hcclDataType_t,hcclRedOp_t,hcclComm_t,void*,int,void**);
    using HAction=hcclResult_t(*)(void*);
    using HBatchCreate=hcclResult_t(*)(void* const*,size_t,void**);
    using EventInfo=synStatus(*)(synEventHandle,Sync*,uint32_t*);
public:
    Plan(){
        create_=symbol<Create>("synNativeComputeGraphCreate");begin_=symbol<Action>("synNativeComputeGraphBeginCapture");
        end_=symbol<Action>("synNativeComputeGraphEndCapture");destroy_=symbol<Action>("synNativeComputeGraphDestroy");
        get_=symbol<Get>("synNativeComputeGraphGetInfo");prepare_=symbol<Prepare>("synNativeComputeGraphPreparePlanV2");
        replay_=symbol<Replay>("synNativeComputeGraphReplayPlan");hcreate_=symbol<HCreate>("hcclTp4NativeGraphCreate");
        hcapture_=symbol<HAction>("hcclTp2NativeGraphCapture");hdestroy_=symbol<HAction>("hcclTp2NativeGraphDestroy");
        batch_create_=symbol<HBatchCreate>("hcclTp2NativeBatchCreate");batch_destroy_=symbol<HAction>("hcclTp2NativeBatchDestroy");
    }
    void begin(){
        habana::eager::JoinPendingPipelineThreads();auto self=shared_from_this();
        habana::HPUDeviceContext::execute_thread().enqueue([self]{
            TORCH_CHECK(!self->graph_,"Repeated future capture");
            self->compute_=habana::HPUDeviceContext::get_device().get_stream(0);
            check(self->create_(&self->graph_,self->compute_));check(self->begin_(self->graph_));
        });habana::eager::JoinPendingPipelineThreads();
    }
    void end(uint64_t segments){
        habana::eager::JoinPendingPipelineThreads();auto self=shared_from_this();
        habana::HPUDeviceContext::execute_thread().enqueue([self,segments]{
            check(self->end_(self->graph_));check(self->get_(self->graph_,&self->info_));
            TORCH_CHECK(self->info_.state==2&&self->info_.segments==segments,"Captured recipe count differs");
            check(synStreamSynchronize(self->compute_));
        });habana::eager::JoinPendingPipelineThreads();
    }
    void prepare(const c10::intrusive_ptr<c10d::Backend>& backend,const std::vector<at::Tensor>& packets,
        const at::Tensor& peers,const at::Tensor& flags,const std::vector<uint32_t>& producers,
        const std::vector<uint32_t>& consumers,const std::vector<uint64_t>& deltas,
        const std::vector<at::Tensor>& retained,bool future){
        habana::eager::JoinPendingPipelineThreads();
        auto* group=dynamic_cast<c10d::ProcessGroupEagerHCCL*>(backend.get());
        TORCH_CHECK(group&&group->getSize()==4,"Future capability requires initialized TP4 EagerHCCL");
        communicator_=group->lowLatencyCommunicator();TORCH_CHECK(communicator_,"Uninitialized peer communicator");
        TORCH_CHECK(graph_&&!publisher_&&!packets.empty()&&packets.size()<=128&&
            packets.size()==producers.size()&&packets.size()==consumers.size()&&packets.size()==deltas.size(),
            "Incomplete future bindings");
        const auto rows=packets[0].size(0);
        TORCH_CHECK(peers.scalar_type()==at::kBFloat16&&peers.dim()==5&&peers.size(0)==2&&
            peers.size(2)==4&&peers.size(3)==rows&&peers.size(4)==5120&&peers.is_contiguous()&&
            flags.scalar_type()==at::kInt&&flags.sizes()==at::IntArrayRef({1<<24})&&flags.is_contiguous());
        TORCH_CHECK(peers.nbytes()>(48u<<20)&&peers.size(1)>=128,"Future tables must remain in HBM");
        resources_=std::make_shared<GenericResourceHolder>();
        tensors_=retained;tensors_.insert(tensors_.end(),packets.begin(),packets.end());
        tensors_.push_back(peers);tensors_.push_back(flags);
        std::vector<void*> addresses;
        for(const auto& tensor:tensors_){TORCH_CHECK(tensor.device()==peers.device(),"Retained owner is on a different device");
            resources_->add_tensor(tensor);addresses.push_back(tensor.data_ptr());
            habana::get_tensor_extra_meta(tensor)->set_tensor_pipelined();}
        communicator_->getDeviceCtxt()->lock_address(addresses,resources_->get_address_lock());
        auto self=shared_from_this();
        habana::HPUDeviceContext::execute_thread().enqueue([self,packets,peers,flags,producers,consumers,deltas,future]{
            const auto& locked=*self->resources_->get_address_lock();
            const size_t base=self->tensors_.size()-packets.size()-2;
            const auto table=locked.at(base+packets.size()),flag=locked.at(base+packets.size()+1);
            const auto device=static_cast<synDeviceId>(peers.device().index());
            check(synStreamCreateGeneric(&self->nic_,device,0));check(synStreamSetAffinity(device,self->nic_,2));
            check(synStreamCreateGeneric(&self->copy_,device,0));check(synStreamSetAffinity(device,self->copy_,4));
            uint64_t affinity=0;check(synStreamGetAffinity(device,self->compute_,&affinity));
            TORCH_CHECK(!(affinity&6),"Future publisher aliases physical compute/DMA queue");
            const uint64_t bytes=packets[0].nbytes();
            std::vector<Publisher::Point> points;
            for(size_t i=0;i<packets.size();++i){
                TORCH_CHECK(packets[i].sizes()==packets[0].sizes()&&packets[i].scalar_type()==at::kBFloat16&&
                    packets[i].size(1)==5120&&producers[i]<consumers[i]&&consumers[i]<self->info_.segments&&
                    deltas[i]==(future?1u:0u),"Unqualified future point geometry/external order");
                void* graph=nullptr;
                check(self->hcreate_(reinterpret_cast<void*>(locked.at(base+i)),
                    reinterpret_cast<void*>(table+i*4*bytes),packets[i].numel(),hcclBfloat16,hcclSum,
                    *self->communicator_->GetHcclHandle(),self->nic_,2,&graph));
                self->nic_graphs_.push_back(graph);check(self->hcapture_(graph));
                check(synStreamSynchronize(self->nic_));
                synEventHandle event=nullptr;
                if(future){check(synEventCreate(&event,device,0));self->events_.push_back(event);}
                points.push_back({deltas[i],future?flag+((1u<<24)-1)*4ull:0,future?flag+i*8ull:0,event});
            }
            check(self->batch_create_(self->nic_graphs_.data(),self->nic_graphs_.size(),&self->batch_));
            check(synEventCreate(&self->seed_,device,0));check(synEventRecord(self->seed_,self->nic_));
            Sync completion{};uint32_t onHcl=0;
            check(symbol<EventInfo>("dsv41ReceiveEventSyncInfo")(self->seed_,&completion,&onHcl));
            TORCH_CHECK(onHcl,"NIC cold template names a non-network queue");
            Publisher::Api api{symbol<Publisher::BatchReplay>("hcclTp2NativeBatchReplay"),
                symbol<Publisher::Bind>("dsv41ReceiveBindEventFromTemplate")};
            self->publisher_=std::make_unique<Publisher>(api,self->batch_,self->seed_,self->copy_,std::move(points));
            check(self->prepare_(self->graph_,producers.data(),consumers.data(),producers.size(),
                completion.longSoIndex,Publisher::replay,self->publisher_.get()));
        });habana::eager::JoinPendingPipelineThreads();
    }
    void replay(){
        TORCH_CHECK(publisher_,"Future plan not prepared");auto self=shared_from_this();
        habana::eager::PipelineTask<habana::eager::ThreadType::LOWERING>([self]{
            habana::HPUDeviceContext::execute_thread().enqueue([self]{Sync complete;uint64_t statistics[12]{};
                check(self->replay_(self->graph_,&complete,statistics,12));self->publisher_->check();});
        });
    }
    std::vector<uint64_t> info(){habana::eager::JoinPendingPipelineThreads();check(get_(graph_,&info_));
        return {info_.segments,info_.replays,info_.global_bytes,info_.arc_bytes};}
    void close(){habana::eager::JoinPendingPipelineThreads();if(!graph_)return;
        check(synStreamSynchronize(compute_));if(nic_)check(synStreamSynchronize(nic_));
        if(copy_)check(synStreamSynchronize(copy_));publisher_.reset();
        if(batch_){check(batch_destroy_(batch_));batch_=nullptr;}
        for(auto graph:nic_graphs_){check(hdestroy_(graph));}
        nic_graphs_.clear();
        for(auto event:events_){check(synEventDestroy(event));}
        events_.clear();
        if(seed_){check(synEventDestroy(seed_));}
        seed_=nullptr;
        if(nic_){check(synStreamDestroy(nic_));}
        nic_=nullptr;
        if(copy_){check(synStreamDestroy(copy_));}
        copy_=nullptr;
        check(destroy_(graph_));graph_=nullptr;tensors_.clear();resources_.reset();communicator_.reset();
    }
private:
    Create create_;Action begin_,end_,destroy_;Get get_;Prepare prepare_;Replay replay_;
    HCreate hcreate_;HAction hcapture_,hdestroy_,batch_destroy_;HBatchCreate batch_create_;
    void* graph_=nullptr;void* batch_=nullptr;std::vector<void*> nic_graphs_;
    synStreamHandle compute_=nullptr,nic_=nullptr,copy_=nullptr;synEventHandle seed_=nullptr;
    std::vector<synEventHandle> events_;std::vector<at::Tensor> tensors_;Info info_;
    std::shared_ptr<habana::HcclCommunicator> communicator_;
    std::shared_ptr<GenericResourceHolder> resources_;std::unique_ptr<Publisher> publisher_;
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module){
    pybind11::class_<Plan,std::shared_ptr<Plan>>(module,"Plan").def(pybind11::init<>())
        .def("begin",&Plan::begin).def("end",&Plan::end).def("prepare",&Plan::prepare)
        .def("replay",&Plan::replay).def("info",&Plan::info).def("close",&Plan::close);
}
