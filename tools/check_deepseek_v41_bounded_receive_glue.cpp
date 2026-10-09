// SPDX-License-Identifier: Apache-2.0
#include "tpc_kernel_lib_interface.h"
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tpc_lib_api;
void require(bool value){if(!value)throw std::runtime_error("Router pair glue contract failed");}
void shape(Tensor& t,decltype(DATA_F32) type,std::initializer_list<uint64_t> dims){
    t={};t.geometry.dataType=type;t.geometry.dims=dims.size();
    unsigned i=0;for(auto size:dims){t.geometry.maxSizes[i]=size;t.geometry.minSizes[i++]=size;}
}
int main(int argc,char** argv){
    require(argc==2);void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);require(lib);
    for(const char* symbol:{"GetLibVersion","GetKernelGuids","InstantiateTpcKernel",
        "GetShapeInference","GetSupportedDataLayouts","GetSuggestedManipulation"})require(dlsym(lib,symbol));
    auto call=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));require(call);
    for(uint64_t rows:{2u,6u})for(uint64_t group:{2u,4u}) {
        std::array<Tensor,4> input{};std::array<Tensor,2> output{};
        std::array<TensorAccessPattern,4> ia{};std::array<TensorAccessPattern,2> oa{};
        HabanaKernelParams p{};HabanaKernelInstantiation out{};
        out.inputTensorAccessPattern=ia.data();out.outputTensorAccessPattern=oa.data();
        p.deviceId=DEVICE_ID_GAUDI2;p.inputTensors=input.data();p.outputTensors=output.data();
        p.inputTensorNr=4;p.outputTensorNr=2;
        std::strcpy(p.guid.name,"custom_deepseek_v41_bounded_peer_receive_gaudi2");
        int params[]={127,int(group)-1,65536};p.nodeParams.nodeParams=params;
        p.nodeParams.nodeParamsSize=sizeof(params);
        uint64_t capacity=128;
        while(5120*rows*group*capacity*4<=(uint64_t(48)<<20))capacity*=2;
        shape(input[0],DATA_BF16,{5120,rows});shape(input[1],DATA_BF16,{5120,rows,group,capacity,2});
        shape(input[2],DATA_I32,{1u<<24});shape(input[3],DATA_I32,{1});
        shape(output[0],DATA_BF16,{5120,rows});shape(output[1],DATA_I32,{40,rows});
        const auto saved=input;
        require(call(&p,&out)==GLUE_INSUFFICIENT_ELF_BUFFER);
        std::vector<char> elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
        require(call(&p,&out)==GLUE_SUCCESS);require(std::memcmp(input.data(),saved.data(),sizeof(input))==0);
        require(out.kernel.paramsNr==3);require(out.indexSpaceGeometry[0]==40);
        require(out.indexSpaceGeometry[1]==rows);
        for(unsigned i=1;i<4;++i)require(out.inputTensorAccessPattern[i].allRequired);
        shape(input[3],DATA_I32,{1u<<24});require(call(&p,&out)==GLUE_SUCCESS);
        input[3].geometry.maxSizes[0]--;require(call(&p,&out)!=GLUE_SUCCESS);
        shape(input[3],DATA_I32,{1});
        params[1]=group;require(call(&p,&out)!=GLUE_SUCCESS);params[1]=group-1;
        params[2]=65537;require(call(&p,&out)!=GLUE_SUCCESS);params[2]=65536;
        if(capacity>128){input[1].geometry.maxSizes[3]=128;
            require(call(&p,&out)!=GLUE_SUCCESS);input[1].geometry.maxSizes[3]=capacity;}
        input[2].geometry.maxSizes[0]--;require(call(&p,&out)!=GLUE_SUCCESS);
    }
    {
        std::array<Tensor,2> input{};std::array<Tensor,1> output{};
        std::array<TensorAccessPattern,2> ia{};std::array<TensorAccessPattern,1> oa{};
        HabanaKernelParams p{};HabanaKernelInstantiation out{};
        p.deviceId=DEVICE_ID_GAUDI2;p.inputTensors=input.data();p.outputTensors=output.data();
        p.inputTensorNr=2;p.outputTensorNr=1;
        std::strcpy(p.guid.name,"custom_deepseek_v41_future_epoch_gaudi2");
        out.inputTensorAccessPattern=ia.data();out.outputTensorAccessPattern=oa.data();
        shape(input[0],DATA_BF16,{6*2048,1});shape(input[1],DATA_I32,{1u<<24});
        shape(output[0],DATA_BF16,{6*2048,1});
        require(call(&p,&out)==GLUE_INSUFFICIENT_ELF_BUFFER);
        std::vector<char> elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
        require(call(&p,&out)==GLUE_SUCCESS);require(out.indexSpaceGeometry[0]==96);
        require(out.inputTensorAccessPattern[1].allRequired);
        input[0].geometry.maxSizes[0]--;require(call(&p,&out)!=GLUE_SUCCESS);
    }
    std::cout<<"{\"passed\":true,\"rows\":[2,6],\"tp_sizes\":[2,4],\"descriptors_unchanged\":true}"<<std::endl;
}
