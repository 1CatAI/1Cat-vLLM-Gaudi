// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type,std::initializer_list<uint64_t> sizes) {
    Tensor value{};value.geometry.dataType=type;value.geometry.dims=sizes.size();unsigned d=0;
    for(auto n:sizes){value.geometry.maxSizes[d]=value.geometry.minSizes[d]=n;value.permutation[d]=d;++d;}
    return value;
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    auto privateLib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),baseLib=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
    if(!privateLib||!baseLib){puts(dlerror());return 2;}
    auto candidate=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(privateLib,"InstantiateTpcKernel"));
    auto reference=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(baseLib,"InstantiateTpcKernel"));
    for(unsigned heads:{8u,16u})for(unsigned rows:{2u,6u})for(bool inverse:{false,true}) {
        Tensor inputs[]={tensor(DATA_BF16,{512,heads,rows}),tensor(DATA_I32,{rows}),tensor(DATA_F32,{64,32768})};
        Tensor outputs[]={tensor(DATA_BF16,{512,heads,rows})};
        HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.apiVersion=1;p.maxAvailableTpc=24;
        p.inputTensorNr=3;p.outputTensorNr=1;p.inputTensors=inputs;p.outputTensors=outputs;
        std::strcpy(p.guid.name,inverse?"custom_deepseek_v41_rope_inverse_coherent_bf16_gaudi2":
                                          "custom_deepseek_v41_rope_coherent_bf16_gaudi2");
        TensorAccessPattern inAP[3]{},outAP[1]{};HabanaKernelInstantiation result{};
        result.inputTensorAccessPattern=inAP;result.outputTensorAccessPattern=outAP;
        if(candidate(&p,&result)!=GLUE_INSUFFICIENT_ELF_BUFFER)return 1;
        std::vector<char> elf(result.kernel.elfSize);result.kernel.kernelElf=elf.data();
        if(candidate(&p,&result)!=GLUE_SUCCESS)return 1;
        if(result.indexSpaceRank!=3 || result.indexSpaceGeometry[1]!=heads || result.indexSpaceGeometry[2]!=1 ||
            inAP[0].mapping[2].a!=0 || inAP[0].mapping[2].end_b!=rows-1 ||
            outAP[0].mapping[2].a!=0 || outAP[0].mapping[2].end_b!=rows-1 ||
            inAP[1].mapping[0].a!=0 || inAP[1].mapping[0].end_b!=rows-1)return 1;
        std::strcpy(p.guid.name,inverse?"custom_deepseek_v41_rope_inverse_bf16_gaudi2":
                                          "custom_deepseek_v41_rope_bf16_gaudi2");
        result.kernel.elfSize=0;
        if(reference(&p,&result)!=GLUE_INSUFFICIENT_ELF_BUFFER)return 1;
        std::vector<char> base(result.kernel.elfSize);result.kernel.kernelElf=base.data();
        if(reference(&p,&result)!=GLUE_SUCCESS)return 1;
        printf("rows%u heads%u inverse%u newELF%u\n",rows,heads,inverse,unsigned(elf!=base));
    }
}
