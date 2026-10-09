// SPDX-License-Identifier: Apache-2.0
// CPU-only validation of the private SWA-cache glue contract.
#include <tpc_kernel_lib_interface.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>
using namespace tpc_lib_api;
static Tensor tensor(TensorDataType type, std::initializer_list<uint64_t> sizes) {
    Tensor t{}; t.geometry.dataType=type; t.geometry.dims=sizes.size();
    unsigned d=0;
    for(auto n:sizes){t.geometry.maxSizes[d]=t.geometry.minSizes[d]=n;t.permutation[d]=d;++d;}
    return t;
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    auto* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!lib){std::fprintf(stderr,"%s\n",dlerror());return 2;}
    auto instantiate=reinterpret_cast<pfnInstantiateTpcKernel>(dlsym(lib,"InstantiateTpcKernel"));
    if(!instantiate)return 2;
    bool passed=true;
    for(unsigned rows:{2u,5u,6u})for(unsigned mode=0;mode<11;++mode) {
        std::vector<Tensor> inputs,outputs;
        const char* name=nullptr;
        int ratio=1;
        if(mode==0) {
            name="custom_deepseek_v41_swa_cached_publish_gaudi2";
            inputs={tensor(DATA_BF16,{512,256}),tensor(DATA_U8,{288,32768}),
                    tensor(DATA_I32,{512,rows}),tensor(DATA_I32,{rows}),tensor(DATA_I32,{512}),
                    tensor(DATA_I32,{rows}),tensor(DATA_I32,{16,rows})};
            outputs={tensor(DATA_BF16,{512,640,rows}),tensor(DATA_F32,{512,640,rows}),
                     tensor(DATA_F32,{640,rows}),tensor(DATA_BF16,{512,640,rows})};
        } else if(mode==1) {
            name="custom_deepseek_v41_swa_cached_reuse_gaudi2";
            inputs={tensor(DATA_BF16,{512,256}),tensor(DATA_BF16,{512,640,rows}),
                    tensor(DATA_F32,{640,rows}),tensor(DATA_I32,{rows}),tensor(DATA_I32,{rows}),
                    tensor(DATA_I32,{16,rows})};
            outputs={tensor(DATA_BF16,{512,128,rows}),tensor(DATA_F32,{512,128,rows}),tensor(DATA_F32,{128,rows})};
        } else if(mode==2) {
            name="custom_deepseek_v41_swa_batch_cache_write_gaudi2";
            inputs={tensor(DATA_U8,{528,256}),tensor(DATA_BF16,{512,rows}),
                    tensor(DATA_I32,{rows}),tensor(DATA_BF16,{512,256})};
            outputs={tensor(DATA_I32,{16,rows})};
        } else if(mode==3) {
            name="custom_deepseek_v41_expert_pair_silu_gaudi2";
            const auto routes=rows*6;
            ratio=640;
            inputs={tensor(DATA_F32,{2560,1,routes/2}),tensor(DATA_I32,{routes,1}),
                    tensor(DATA_F32,{1,routes}),tensor(DATA_BF16,{256,5,384}),tensor(DATA_F32,{routes,1})};
            outputs={tensor(DATA_F8_143,{640,1,routes}),tensor(DATA_F32,{1,1,routes})};
        }
        if(mode==4) {
            name="custom_deepseek_v41_compressor_sequence_bf16_gaudi2";
            inputs={tensor(DATA_F32,{512,8}),tensor(DATA_F32,{512,8}),
                    tensor(DATA_F32,{512,rows}),tensor(DATA_F32,{512,rows}),tensor(DATA_I32,{rows})};
            outputs={tensor(DATA_BF16,{512,rows})};
        }
        if(mode==5) {
            name="custom_deepseek_v41_silu_full_rows_gaudi2";
            const auto routes=rows*6;
            inputs={tensor(DATA_F32,{1280,1,routes}),tensor(DATA_I32,{routes,1}),
                    tensor(DATA_F32,{1,routes}),tensor(DATA_BF16,{256,5,384}),tensor(DATA_F32,{routes,1})};
            outputs={tensor(DATA_F8_143,{640,1,routes}),tensor(DATA_F32,{1,1,routes})};
        }
        if(mode==6) {
            name="custom_deepseek_v41_weighted_finish_mask_gaudi2";
            inputs={tensor(DATA_F32,{129280,rows}),tensor(DATA_F32,{129280,rows}),tensor(DATA_I32,{1,rows})};
            outputs={tensor(DATA_F32,{129280,rows}),tensor(DATA_I32,{64,rows}),
                     tensor(DATA_I32,{64,rows}),tensor(DATA_I32,{64,rows})};
        }
        if(mode==7) {
            name="custom_deepseek_v41_swa_source_reuse_gaudi2";
            inputs={tensor(DATA_U8,{528,256}),tensor(DATA_BF16,{512,640,rows}),tensor(DATA_F32,{640,rows}),
                    tensor(DATA_I32,{rows}),tensor(DATA_I32,{rows})};
            outputs={tensor(DATA_BF16,{512,128,rows}),tensor(DATA_F32,{512,128,rows}),tensor(DATA_F32,{128,rows})};
        }
        if(mode==8) {
            name="custom_deepseek_v41_mla_adjacent_softmax_gaudi2";
            inputs={tensor(DATA_F32,{640,16,rows}),tensor(DATA_F32,{640,rows}),tensor(DATA_F32,{16}),
                    tensor(DATA_F32,{1})};
            outputs={tensor(DATA_BF16,{640,32,rows})};
        }
        if(mode==9) {
            name="custom_deepseek_v41_mla_adjacent_finish_gaudi2";
            inputs={tensor(DATA_F32,{512,32,rows})};outputs={tensor(DATA_BF16,{512,16,rows})};
        }
        if(mode==10) {
            name="custom_deepseek_v41_swa_keys_reuse_gaudi2";
            inputs={tensor(DATA_U8,{528,256}),tensor(DATA_BF16,{512,640,rows}),tensor(DATA_F32,{640,rows}),
                    tensor(DATA_I32,{rows}),tensor(DATA_I32,{rows})};
            outputs={tensor(DATA_BF16,{512,128,rows}),tensor(DATA_F32,{128,rows})};
        }
        const auto before=inputs;
        HabanaKernelParams params{};
        params.apiVersion=1;params.deviceId=DEVICE_ID_GAUDI2;params.maxAvailableTpc=24;
        std::strcpy(params.guid.name,name);params.inputTensors=inputs.data();params.inputTensorNr=inputs.size();
        params.outputTensors=outputs.data();params.outputTensorNr=outputs.size();
        if(mode==0 || mode==3){params.nodeParams.nodeParams=&ratio;params.nodeParams.nodeParamsSize=sizeof(ratio);}
        TensorAccessPattern input_ap[16]{},output_ap[16]{};
        std::vector<char> elf(1024*1024);
        HabanaKernelInstantiation result{};result.inputTensorAccessPattern=input_ap;result.outputTensorAccessPattern=output_ap;
        result.kernel.kernelElf=elf.data();result.kernel.elfSize=elf.size();
        const auto status=instantiate(&params,&result);
        const bool unchanged=std::memcmp(before.data(),inputs.data(),inputs.size()*sizeof(Tensor))==0;
        bool split_valid=true;
        if(result.preferredSplitDim) {
            auto check=[&](const std::vector<Tensor>& tensors,TensorAccessPattern* ap) {
                for(unsigned i=0;i<tensors.size();++i) {
                    bool valid=ap[i].allRequired;
                    for(unsigned d=0;d<tensors[i].geometry.dims;++d)
                        valid |= ap[i].mapping[d].indexSpaceDim==result.preferredSplitDim-1 && ap[i].mapping[d].allRequired;
                    split_valid &= valid;
                }
            };
            check(inputs,input_ap);check(outputs,output_ap);
        }
        std::printf("{\"kernel\":\"%s\",\"rows\":%u,\"status\":%d,\"input_descriptors_unchanged\":%s,\"preferred_split_contract_valid\":%s,\"elf_bytes\":%u,\"index_space\":[%llu,%llu]}\n",
                    name,rows,status,unchanged?"true":"false",split_valid?"true":"false",result.kernel.elfSize,
                    (unsigned long long)result.indexSpaceGeometry[0],(unsigned long long)result.indexSpaceGeometry[1]);
        passed &= status==GLUE_SUCCESS && unchanged && split_valid;
    }
    dlclose(lib);return passed?0:1;
}
