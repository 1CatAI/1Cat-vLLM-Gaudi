// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type,std::initializer_list<uint64_t> sizes) {
    Tensor t{};t.geometry.dataType=type;t.geometry.dims=sizes.size();unsigned d=0;
    for(auto n:sizes){t.geometry.maxSizes[d]=t.geometry.minSizes[d]=n;t.permutation[d]=d;++d;}
    return t;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    auto lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!lib){puts(dlerror());return 2;}
    auto fn=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));
    for(unsigned width:{640u,1280u})for(bool down:{false,true}) {
        const unsigned k=down?width:5120,n=down?5120:width*2,slots=15;
        Tensor in[]={tensor(DATA_I32,{slots,1}),tensor(DATA_BF16,{n,k,128})};
        Tensor out[]={tensor(DATA_BF16,{n,k,slots})};
        HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.maxAvailableTpc=24;
        p.inputTensorNr=2;p.outputTensorNr=1;p.inputTensors=in;p.outputTensors=out;
        std::strcpy(p.guid.name,"custom_deepseek_v41_expert_cached_gather_bf16_gaudi2");
        TensorAccessPattern ia[2]{},oa[1]{};HabanaKernelInstantiation r{};
        r.inputTensorAccessPattern=ia;r.outputTensorAccessPattern=oa;
        if(fn(&p,&r)!=GLUE_INSUFFICIENT_ELF_BUFFER)return 1;
        std::vector<char> elf(r.kernel.elfSize);r.kernel.kernelElf=elf.data();
        if(fn(&p,&r)!=GLUE_SUCCESS || r.indexSpaceRank!=3 ||
           r.indexSpaceGeometry[0]!=n/128 || r.indexSpaceGeometry[1]!=k/32 ||
           r.indexSpaceGeometry[2]!=slots || ia[0].mapping[0].indexSpaceDim!=2 ||
           ia[1].mapping[2].a!=0 || ia[1].mapping[2].end_b!=127 ||
           oa[0].mapping[2].a!=1)return 1;
        printf("W%s K%u N%u slots%u: exact rectangular ownership\n",down?"2":"13",k,n,slots);
    }
}
