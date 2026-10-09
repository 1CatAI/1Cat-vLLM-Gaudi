// SPDX-License-Identifier: Apache-2.0
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type,uint64_t width,uint64_t rows) {
    Tensor value{};value.geometry.dataType=type;value.geometry.dims=2;
    value.geometry.maxSizes[0]=value.geometry.minSizes[0]=width;
    value.geometry.maxSizes[1]=value.geometry.minSizes[1]=rows;
    value.permutation[0]=0;value.permutation[1]=1;return value;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    auto library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library){puts(dlerror());return 2;}
    auto fn=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(library,"InstantiateTpcKernel"));
    for(uint64_t width:{1280u,5120u})for(uint64_t rows:{2u,6u}) {
        Tensor input=tensor(DATA_BF16,width,rows),output=tensor(DATA_F8_143,width,rows);
        float reciprocal=16.0f;
        HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.apiVersion=1;p.maxAvailableTpc=24;
        p.inputTensorNr=p.outputTensorNr=1;p.inputTensors=&input;p.outputTensors=&output;
        p.nodeParams.nodeParams=&reciprocal;p.nodeParams.nodeParamsSize=sizeof(reciprocal);
        std::strcpy(p.guid.name,"custom_deepseek_v41_fixed_dense_quant_gaudi2");
        TensorAccessPattern inAP{},outAP{};HabanaKernelInstantiation out{};
        out.inputTensorAccessPattern=&inAP;out.outputTensorAccessPattern=&outAP;
        if(fn(&p,&out)!=GLUE_INSUFFICIENT_ELF_BUFFER)return 1;
        std::vector<char> elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
        if(fn(&p,&out)!=GLUE_SUCCESS || out.indexSpaceRank!=2 ||
           out.indexSpaceGeometry[0]!=width/32 || out.indexSpaceGeometry[1]!=rows ||
           inAP.mapping[0].a!=32 || inAP.mapping[0].end_b!=31 || inAP.mapping[1].a!=1 ||
           outAP.mapping[0].a!=32 || outAP.mapping[1].a!=1 || out.kernel.paramsNr!=1 ||
           std::memcmp(out.kernel.scalarParams,&reciprocal,sizeof(float)))return 1;
        printf("K%lu C%lu: disjoint group32 maps and fixed reciprocal pass\n",width,rows);
    }
}
