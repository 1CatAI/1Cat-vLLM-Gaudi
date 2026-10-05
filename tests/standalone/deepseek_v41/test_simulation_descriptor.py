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
