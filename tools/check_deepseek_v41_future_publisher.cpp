// SPDX-License-Identifier: Apache-2.0
#include "communication/dsv41_future_receive_publisher.h"
#include <cassert>
#include <iostream>
struct Sync {uint32_t longSoIndex;uint64_t targetValue;};
using Publisher = Dsv41FutureReceivePublisher<Sync>;
static bool failBind = false;
static unsigned batches = 0,binds = 0,copies = 0;
hcclResult_t replayBatch(void*,const Sync* input,size_t count,Sync* output) {
    ++batches;assert(count==2);assert(input[0].targetValue==55 && input[1].targetValue==60);
    output[0]={1448,10};output[1]={1448,11};return hcclSuccess;
}
synStatus bind(synEventHandle,synEventHandle,const Sync* info,uint64_t before,uint64_t end) {
    ++binds;assert(info->targetValue==10 && before==9 && end==11);
    return failBind?synFail:synSuccess;
}
synStatus wait(const synStreamHandle,const synEventHandle,const unsigned int flags) {
    assert(flags==0);return synSuccess;
}
synStatus copy(const synStreamHandle,const uint64_t source,const uint64_t bytes,
               const uint64_t destination,const synDmaDir direction) {
    ++copies;assert(source==4096 && destination==8192 && bytes==4 && direction==DRAM_TO_DRAM);
    return synSuccess;
}
int main() {
    Publisher::Api api{replayBatch,bind,wait,copy};
    auto handle=reinterpret_cast<synEventHandle>(uint64_t(1));
    auto stream=reinterpret_cast<synStreamHandle>(uint64_t(2));
    Publisher publisher(api,reinterpret_cast<void*>(uint64_t(3)),handle,stream,
                        {{5,4096,8192,handle},{0,0,0,nullptr}});
    Sync producers[]={{80,60},{80,60}},done[2]{};
    assert(Publisher::replay(&publisher,producers,2,done)==0);
    publisher.check();assert(batches==1 && binds==1 && copies==1);
    failBind=true;
    assert(Publisher::replay(&publisher,producers,2,done)==0); // Drain already-queued NIC safely.
    bool threw=false;try{publisher.check();}catch(const std::runtime_error&){threw=true;}
    assert(threw && publisher.failedPoint()==0 && copies==1);
    producers[0].targetValue=5;
    assert(Publisher::replay(&publisher,producers,2,done)!=0);assert(batches==2);
    std::cout<<"{\"passed\":true,\"mixed_points\":true,\"flag_error_drains_and_rejects_outputs\":true}\n";
}
