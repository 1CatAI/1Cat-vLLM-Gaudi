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
    auto call=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));require(call);
    for(uint64_t rows:{2u,6u})for(unsigned mode:{0u,1u}){
        std::array<Tensor,6> input{};std::array<Tensor,2> output{};
        std::array<TensorAccessPattern,6> ia{};std::array<TensorAccessPattern,2> oa{};
        HabanaKernelParams p{};HabanaKernelInstantiation out{};
        out.inputTensorAccessPattern=ia.data();out.outputTensorAccessPattern=oa.data();
        p.deviceId=DEVICE_ID_GAUDI2;p.inputTensors=input.data();p.outputTensors=output.data();
        p.inputTensorNr=mode?6:2;p.outputTensorNr=2;
        std::strcpy(p.guid.name,mode?"custom_deepseek_v41_router_pair_finish_gaudi2":
                                     "custom_deepseek_v41_router_pair_input_gaudi2");
        if(!mode){shape(input[0],DATA_BF16,{5120,rows});shape(input[1],DATA_F32,{1,rows});
            shape(output[0],DATA_F8_143,{5120,2*rows});shape(output[1],DATA_F32,{1,rows});
        }else{shape(input[0],DATA_F32,{1024,rows*2});shape(input[1],DATA_F32,{384});
            shape(input[2],DATA_F32,{384});shape(input[3],DATA_I8,{rows});
            shape(input[4],DATA_F32,{384,2});shape(input[5],DATA_F32,{1,rows});
            shape(output[0],DATA_I32,{6,rows});shape(output[1],DATA_F32,{6,rows});}
        const auto saved=input;
        require(call(&p,&out)==GLUE_INSUFFICIENT_ELF_BUFFER);
        std::vector<char> elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
        require(call(&p,&out)==GLUE_SUCCESS);require(std::memcmp(input.data(),saved.data(),sizeof(input))==0);
        require(out.kernel.paramsNr==0);require(out.indexSpaceGeometry[mode?0:1]==rows);
        if(mode){require(out.inputTensorAccessPattern[0].mapping[0].end_b==895);
            require(out.inputTensorAccessPattern[0].mapping[1].end_b==rows);
        }else{require(out.indexSpaceGeometry[0]==40);require(out.outputTensorAccessPattern[0].mapping[1].end_b==rows);}
        input[0].geometry.maxSizes[0]--;require(call(&p,&out)!=GLUE_SUCCESS);
    }
    std::cout<<"{\"passed\":true,\"rows\":[2,6],\"descriptors_unchanged\":true}"<<std::endl;
}
