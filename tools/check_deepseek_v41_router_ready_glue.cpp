// SPDX-License-Identifier: Apache-2.0
// CPU-only contract: exact inherited ELF, real product mapping, immutable inputs.
#include "tpc_kernel_lib_interface.h"
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace tpc_lib_api;
void require(bool condition) {if(!condition)throw std::runtime_error("Prepared Router glue contract failed");}
void shape(Tensor& t,TensorDataType type,std::initializer_list<uint64_t> sizes) {
    t={};t.geometry.dataType=type;t.geometry.dims=sizes.size();unsigned i=0;
    for(auto size:sizes){t.geometry.maxSizes[i]=t.geometry.minSizes[i]=size;++i;}
}
std::vector<char> instantiate(pfnInstantiateTpcKernel fn,HabanaKernelParams& p,
                              HabanaKernelInstantiation& out) {
    auto status=fn(&p,&out);
    if(status!=GLUE_INSUFFICIENT_ELF_BUFFER) {
        std::cerr<<"ELF query status="<<status<<" guid="<<p.guid.name<<std::endl;
        require(false);
    }
    std::vector<char> elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
    require(fn(&p,&out)==GLUE_SUCCESS);return elf;
}
int main(int argc,char** argv) {
    require(argc==3);auto alias=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),parent=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
    require(alias&&parent);
    auto fn=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(alias,"InstantiateTpcKernel"));
    auto inherited=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(parent,"InstantiateTpcKernel"));
    require(fn&&inherited);
    for(unsigned rows:{2u,6u}) {
        std::array<Tensor,6> input{};std::array<Tensor,2> output{};
        shape(input[0],DATA_F32,{512,rows});shape(input[1],DATA_F32,{384});shape(input[2],DATA_F32,{384});
        shape(input[3],DATA_I8,{rows});shape(input[4],DATA_F32,{384,1});shape(input[5],DATA_F32,{1,rows});
        shape(output[0],DATA_I32,{6,rows});shape(output[1],DATA_F32,{6,rows});
        HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.inputTensorNr=6;p.outputTensorNr=2;
        p.inputTensors=input.data();p.outputTensors=output.data();
        std::strcpy(p.guid.name,"custom_deepseek_v41_router_ready_scaled_gaudi2");
        const auto saved=input;
        std::array<TensorAccessPattern,6> in_access{};std::array<TensorAccessPattern,2> out_access{};
        HabanaKernelInstantiation actual{};actual.inputTensorAccessPattern=in_access.data();
        actual.outputTensorAccessPattern=out_access.data();
        auto actual_elf=instantiate(fn,p,actual);
        require(std::memcmp(saved.data(),input.data(),sizeof(input))==0);
        require(actual.indexSpaceRank==1&&actual.indexSpaceGeometry[0]==rows&&actual.kernel.paramsNr==0);
        const auto mapping=in_access[0].mapping[0];require(mapping.a==0&&mapping.start_b==0&&mapping.end_b==383);
        shape(input[0],DATA_F32,{1792,rows});std::strcpy(p.guid.name,"custom_deepseek_v41_router_shared_scaled_gaudi2");
        std::array<TensorAccessPattern,6> old_in{};std::array<TensorAccessPattern,2> old_out{};
        HabanaKernelInstantiation original{};original.inputTensorAccessPattern=old_in.data();
        original.outputTensorAccessPattern=old_out.data();
        auto original_elf=instantiate(inherited,p,original);require(actual_elf==original_elf);
        std::strcpy(p.guid.name,"custom_deepseek_v41_router_ready_scaled_gaudi2");
        for(unsigned width:{384u,513u}) {
            shape(input[0],DATA_F32,{width,rows});require(fn(&p,&actual)==GLUE_INCOMPATIBLE_INPUT_SIZE);
        }
    }
    std::cout<<"{\"passed\":true,\"rows\":[2,6],\"actual_product_columns\":512,"
                 "\"access_columns\":[0,383],\"inherited_ELF_byte_exact\":true,"
                 "\"caller_descriptors_unchanged\":true,\"hardware_used\":false}"<<std::endl;
}
