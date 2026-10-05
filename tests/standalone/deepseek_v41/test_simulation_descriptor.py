# SPDX-License-Identifier: Apache-2.0
"""Exercise the FP32 descriptor through the actual Gaudi2 instruction simulator."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_fp32_simulation_round_trip(tmp_path):
    headers = Path('/usr/lib/habanatools/include')
    if not shutil.which('tpc-clang') or not (headers / 'tpc_test_core_api.h').exists():
        pytest.skip('Gaudi2 instruction simulator SDK is not installed')
    root = Path(__file__).resolve().parents[3]
    kernel = tmp_path / 'echo.c'
    kernel.write_text('void main(tensor x,tensor y){v_f32_st_tnsr((int5){0},y,v_f32_ld_tnsr_b((int5){0},x));}\n')
    subprocess.run(['tpc-clang', '-O2', '-march=gaudi2', '-c', str(kernel), '-o', str(tmp_path/'echo.o')],
                   check=True, capture_output=True, text=True)
    source = tmp_path / 'test.cpp'
    source.write_text(r"""
#include "deepseek_v41_simulation_descriptor.hpp"
#include <array>
#include <fstream>
#include <iterator>
#include <cstring>
#include <vector>
int main(int argc,char** argv) {
    using namespace tpc_lib_api; using namespace tpc_tests;
    std::ifstream stream(argv[1],std::ios::binary);
    std::vector<char> elf{std::istreambuf_iterator<char>(stream),{}};
    std::array<float,64> input{},output{};
    for(int i=0;i<64;++i)input[i]=(i-31)*0.125f;
    Tensor tensor{};tensor.geometry.dataType=DATA_F32;tensor.geometry.dims=1;tensor.geometry.maxSizes[0]=64;
    auto a=gaudi_validation::descriptor(input.data(),tensor),b=gaudi_validation::descriptor(output.data(),tensor);
    if((a.configuration&15)!=7)return 2;
    if(gaudi_validation::tensor_config_type(DATA_BF16)!=6 ||
       gaudi_validation::tensor_config_type(DATA_F8_143)!=10)return 3;
    HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.maxAvailableTpc=1;
    p.inputTensorNr=p.outputTensorNr=1;p.inputTensors=p.outputTensors=&tensor;
    strcpy(p.guid.name,"descriptor_echo");
    HabanaKernelInstantiation g{};TensorAccessPattern ia{},oa{};
    g.inputTensorAccessPattern=&ia;g.outputTensorAccessPattern=&oa;
    g.indexSpaceRank=1;g.indexSpaceGeometry[0]=1;g.kernel.kernelElf=elf.data();g.kernel.elfSize=elf.size();
    std::vector<TensorDesc2> descriptors{a,b};VPEStats stats;TestConfigurations cfg{};
    cfg.dontTestIndexSpaceMapping=true;cfg.disableShuffleIndexSpacePartition=true;
    RunSimulation(p,g,descriptors,stats,e_accessPatternIgnoreMode,{},{},cfg);
    if(!stats.instructionsExecuted || std::memcmp(input.data(),output.data(),sizeof(input)))return 4;
    tensor.geometry.dataType=DATA_PACKED_NF4;
    try {gaudi_validation::descriptor(input.data(),tensor);return 5;}catch(const std::invalid_argument&) {}
    return 0;
}
""")
    binary = tmp_path / 'test'
    subprocess.run(['g++', '-O2', '-I'+str(headers), '-I'+str(root/'tools'), str(source),
                    '-L/usr/lib/habanatools', '-Wl,-rpath,/usr/lib/habanatools',
                    '-ltpc_tests_core_ext', '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary), str(tmp_path/'echo.o')], check=True, capture_output=True, text=True, timeout=30)


def test_integer_vector_and_scalar_addressing(tmp_path):
    """Wrong integer encodings can copy vectors yet misaddress scalar stores."""
    headers = Path('/usr/lib/habanatools/include')
    if not shutil.which('tpc-clang') or not (headers / 'tpc_test_core_api.h').exists():
        pytest.skip('Gaudi2 instruction simulator SDK is not installed')
    root = Path(__file__).resolve().parents[3]
    kernel = tmp_path / 'integers.c'
    kernel.write_text(r'''
void main(tensor i8,tensor u8,tensor i16,tensor u16,tensor i32,tensor u32,
          tensor oi8,tensor ou8,tensor oi16,tensor ou16,tensor oi32,tensor ou32,tensor deltas) {
    v_i8_st_tnsr((int5){0},oi8,v_i8_ld_tnsr_b((int5){0},i8));
    v_u8_st_tnsr((int5){0},ou8,v_u8_ld_tnsr_b((int5){0},u8));
    v_i16_st_tnsr((int5){0},oi16,v_i16_ld_tnsr_b((int5){0},i16));
    v_u16_st_tnsr((int5){0},ou16,v_u16_ld_tnsr_b((int5){0},u16));
    v_i32_st_tnsr((int5){0},oi32,v_i32_ld_tnsr_b((int5){0},i32));
    v_u32_st_tnsr((int5){0},ou32,v_u32_ld_tnsr_b((int5){0},u32));
#define ADDRESS(T,I) { \
    const uint32_t_pair_t first=get_addr(gen_addr((int5){0},T)); \
    const uint32_t_pair_t second=get_addr(gen_addr((int5){1},T)); \
    s_u32_st_g(gen_addr((int5){2*I},deltas),first.v1); \
    s_u32_st_g(gen_addr((int5){2*I+1},deltas),second.v1); }
    ADDRESS(i8,0);ADDRESS(u8,1);ADDRESS(i16,2);ADDRESS(u16,3);ADDRESS(i32,4);ADDRESS(u32,5);
}
''')
    subprocess.run(['tpc-clang', '-O2', '-march=gaudi2', '-c', str(kernel), '-o', str(tmp_path/'integers.o')],
                   check=True, capture_output=True, text=True)
    source = tmp_path / 'integers.cpp'
    source.write_text(r'''
#include "deepseek_v41_simulation_descriptor.hpp"
#include <array>
#include <fstream>
#include <iterator>
#include <cstring>
#include <vector>
int main(int argc,char**argv) {
    using namespace tpc_lib_api;using namespace tpc_tests;
    std::ifstream stream(argv[1],std::ios::binary);std::vector<char>elf{std::istreambuf_iterator<char>(stream),{}};
    Tensor input[6]{},output[7]{};const TensorDataType types[]={DATA_I8,DATA_U8,DATA_I16,DATA_U16,DATA_I32,DATA_U32};
    const uint32_t sizes[]={1,1,2,2,4,4},encodings[]={0,3,1,4,2,5};
    for(int i=0;i<6;++i){input[i].geometry.dataType=types[i];input[i].geometry.dims=1;input[i].geometry.maxSizes[0]=256/sizes[i];output[i]=input[i];}
    output[6].geometry.dataType=DATA_U32;output[6].geometry.dims=1;output[6].geometry.maxSizes[0]=12;
    for(int trial=0;trial<5;++trial) {
        std::array<std::array<uint8_t,256>,6>in{},out{};std::array<uint32_t,12>deltas{};
        std::vector<TensorDesc2>descriptors;
        for(int i=0;i<6;++i){for(int j=0;j<256;++j)in[i][j]=(j*17+i*31+trial*7)%256;auto d=gaudi_validation::descriptor(in[i].data(),input[i]);if((d.configuration&15)!=encodings[i])return 2;descriptors.push_back(d);}
        for(int i=0;i<6;++i)descriptors.push_back(gaudi_validation::descriptor(out[i].data(),output[i]));
        descriptors.push_back(gaudi_validation::descriptor(deltas.data(),output[6]));
        HabanaKernelParams p{};p.deviceId=DEVICE_ID_GAUDI2;p.maxAvailableTpc=1;p.inputTensorNr=6;p.outputTensorNr=7;p.inputTensors=input;p.outputTensors=output;strcpy(p.guid.name,"integer_descriptor_echo");
        HabanaKernelInstantiation g{};TensorAccessPattern ia[6]{},oa[7]{};g.inputTensorAccessPattern=ia;g.outputTensorAccessPattern=oa;g.indexSpaceRank=1;g.indexSpaceGeometry[0]=1;g.kernel.kernelElf=elf.data();g.kernel.elfSize=elf.size();
        VPEStats stats;TestConfigurations cfg{};cfg.dontTestIndexSpaceMapping=true;cfg.disableShuffleIndexSpacePartition=true;
        RunSimulation(p,g,descriptors,stats,e_accessPatternIgnoreMode,{},{},cfg);
        if(!stats.instructionsExecuted||std::memcmp(in.data(),out.data(),sizeof(in)))return 3;
        for(int i=0;i<6;++i)if(deltas[2*i+1]-deltas[2*i]!=sizes[i])return 4;
    }
    return 0;
}
''')
    binary = tmp_path / 'integers'
    subprocess.run(['g++', '-O2', '-I'+str(headers), '-I'+str(root/'tools'), str(source),
                    '-L/usr/lib/habanatools', '-Wl,-rpath,/usr/lib/habanatools',
                    '-ltpc_tests_core_ext', '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary), str(tmp_path/'integers.o')], check=True, capture_output=True, text=True, timeout=30)
