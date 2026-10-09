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
    if(argc!=2)return 2;
    const bool shared=false;
    if(argc==3&&!shared)return 2;
    auto library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library){puts(dlerror());return 2;}
    for(const auto* symbol:{"GetLibVersion","GetKernelGuids","InstantiateTpcKernel",
                            "GetShapeInference","GetSupportedDataLayouts","GetSuggestedManipulation"})
        if(!dlsym(library,symbol)){printf("Missing ABI: %s\n",symbol);return 2;}
    auto function=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(library,"InstantiateTpcKernel"));
    if(!function)return 2;
    for(unsigned tokens:{2u,5u,6u}) for(unsigned width:{640u,1152u}) {
        const auto slots=tokens*6;
        Tensor input[]={tensor(DATA_F32,{256,1,6,20,tokens}),tensor(DATA_I32,{slots,1}),
                        tensor(DATA_F32,{1,slots}),tensor(DATA_BF16,{256,20,384}),
                        tensor(DATA_BF16,{5120,tokens}),tensor(DATA_F32,{1,tokens}),
                        tensor(DATA_F32,{5120,1})};
        Tensor output[]={tensor(DATA_BF16,{5120,1,tokens})};
        HabanaKernelParams p{};p.apiVersion=1;p.deviceId=DEVICE_ID_GAUDI2;p.maxAvailableTpc=24;
        std::strcpy(p.guid.name,shared?"custom_deepseek_v41_w2_reduce_shared_scale_n256_gaudi2":
                                     "custom_deepseek_v41_w2_channel_reduce_gaudi2");
        p.inputTensorNr=shared?7:4;p.outputTensorNr=1;p.inputTensors=input;p.outputTensors=output;
        TensorAccessPattern in_ap[7]{},out_ap[1]{};
        HabanaKernelInstantiation out{};out.inputTensorAccessPattern=in_ap;out.outputTensorAccessPattern=out_ap;
        const auto initial=function(&p,&out);
        if(initial!=GLUE_INSUFFICIENT_ELF_BUFFER||!out.kernel.elfSize) {
            printf("C%u first status%u ELF%u\n",tokens,unsigned(initial),out.kernel.elfSize);return 1;
        }
        std::vector<char> elf(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
        const auto status=function(&p,&out);
        const bool pass=status==GLUE_SUCCESS&&out.indexSpaceRank==2&&out.indexSpaceGeometry[0]==20&&
            out.indexSpaceGeometry[1]==tokens&&in_ap[0].mapping[0].a==0&&in_ap[0].mapping[0].end_b==255&&
            in_ap[0].mapping[2].a==0&&in_ap[0].mapping[3].a==1&&in_ap[0].mapping[4].a==1&&out_ap[0].mapping[0].a==256&&out_ap[0].mapping[2].a==1;
        printf("C%u status%u ELF%u pass%u\n",tokens,unsigned(status),out.kernel.elfSize,pass);
        if(!pass)return 1;
        Tensor decode_input[]={tensor(DATA_I32,{6,tokens}),tensor(DATA_I16,{width*64,20,384}),
            tensor(DATA_I16,{width*4,20,384}),tensor(DATA_BF16,{128}),tensor(DATA_I16,{128,20,384})};
        Tensor decode_output[]={tensor(DATA_F8_143,{256,width,6,20,tokens})};
        HabanaKernelParams q=p;std::strcpy(q.guid.name,"custom_deepseek_v41_w2_channel_decode_gaudi2");
        q.inputTensorNr=5;q.inputTensors=decode_input;q.outputTensors=decode_output;
        out.kernel.elfSize=0;out.kernel.kernelElf=nullptr;
        const auto decodeInitial=function(&q,&out);
        printf("decode initial%u ELF%u shapeK%llu\n",unsigned(decodeInitial),out.kernel.elfSize,static_cast<unsigned long long>(decode_output[0].geometry.maxSizes[1]));
        if(decodeInitial!=GLUE_INSUFFICIENT_ELF_BUFFER)return 1;
        elf.resize(out.kernel.elfSize);out.kernel.kernelElf=elf.data();
        if(function(&q,&out)!=GLUE_SUCCESS||out.indexSpaceGeometry[0]!=20||
           out.indexSpaceGeometry[1]!=6||out.indexSpaceGeometry[2]!=width/128||out.indexSpaceGeometry[3]!=tokens||
           out_ap[0].allRequired||out_ap[0].mapping[0].a!=0||out_ap[0].mapping[1].a!=128||
           out_ap[0].mapping[2].a!=1||out_ap[0].mapping[3].a!=1||out_ap[0].mapping[4].a!=1)return 1;
        printf("C%u decoder affine mappings pass\n",tokens);
        if(shared) {
            if(in_ap[4].mapping[0].a!=256||in_ap[4].mapping[1].indexSpaceDim!=1||
               in_ap[4].mapping[1].a!=1||in_ap[5].mapping[0].a!=0||
               in_ap[5].mapping[1].a!=1||in_ap[6].mapping[0].a!=256||in_ap[6].mapping[1].a!=0)return 1;
            input[5].geometry.dataType=DATA_BF16;
            if(function(&p,&out)!=GLUE_INCOMPATIBLE_INPUT_SIZE)return 1;
            input[5].geometry.dataType=DATA_F32;
        }
        input[1].geometry.dims=1;
        if(function(&p,&out)!=GLUE_INCOMPATIBLE_INPUT_SIZE)return 1;
    }
    return 0;
}
