// SPDX-License-Identifier: Apache-2.0
// Run with tpc_tests_core_ext; no Gaudi device is acquired. Compare the fused
// producer with the existing native BF16 norm followed by dense quantization.
#include <tpc_test_core_api.h>
#include <dlfcn.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace tpc_lib_api;
using namespace tpc_tests;
using Instantiate = GlueCodeReturn (*)(HabanaKernelParams*, HabanaKernelInstantiation*);

Tensor tensor(TensorDataType type, std::initializer_list<uint64_t> dimensions) {
    Tensor result{}; result.geometry.dataType = type; result.geometry.dims = dimensions.size();
    unsigned index = 0;
    for (auto size : dimensions) result.geometry.maxSizes[index++] = size;
    return result;
}
TensorDesc2 descriptor(void* address, const Tensor& tensor) {
    TensorDesc2 result{}; result.baseAddrUnion.baseAddr = reinterpret_cast<uint64_t>(address);
    const unsigned type = tensor.geometry.dataType == DATA_BF16 ? 6 :
                          tensor.geometry.dataType == DATA_F8_143 ? 10 : 0;
    result.configuration = type | (0x1f << 8) | ((tensor.geometry.dims-1) << 16);
    uint32_t stride = 1;
    for (unsigned i = 0; i < 5; ++i) {
        result.dimDescriptors[i].size = i < tensor.geometry.dims ? tensor.geometry.maxSizes[i] : 1;
        result.dimDescriptors[i].stride = stride; stride *= result.dimDescriptors[i].size;
    }
    return result;
}
VPEStats run(Instantiate instantiate, const char* guid, std::vector<Tensor> inputs,
             std::vector<Tensor> outputs, std::vector<void*> buffers,
             std::vector<float> params = {}) {
    HabanaKernelParams p{}; p.deviceId = DEVICE_ID_GAUDI2; p.maxAvailableTpc = 1;
    p.inputTensorNr = inputs.size(); p.outputTensorNr = outputs.size();
    p.inputTensors = inputs.data(); p.outputTensors = outputs.data();
    p.nodeParams.nodeParams = params.data(); p.nodeParams.nodeParamsSize = params.size()*sizeof(float);
    std::strcpy(p.guid.name, guid);
    std::vector<TensorAccessPattern> ia(inputs.size()), oa(outputs.size());
    HabanaKernelInstantiation g{}; g.inputTensorAccessPattern = ia.data(); g.outputTensorAccessPattern = oa.data();
    if (instantiate(&p, &g) != GLUE_INSUFFICIENT_ELF_BUFFER) throw std::runtime_error("host contract failed");
    std::vector<char> elf(g.kernel.elfSize); g.kernel.kernelElf = elf.data();
    if (instantiate(&p, &g) != GLUE_SUCCESS) throw std::runtime_error("ELF contract failed");
    std::vector<TensorDesc2> descriptors;
    auto tensors = inputs;
    tensors.insert(tensors.end(), outputs.begin(), outputs.end());
    for (unsigned i = 0; i < tensors.size(); ++i) descriptors.push_back(descriptor(buffers.at(i), tensors[i]));
    VPEStats stats; TestConfigurations config{};
    config.dontTestIndexSpaceMapping = true; config.disableShuffleIndexSpacePartition = true;
    RunSimulation(p, g, descriptors, stats, e_accessPatternIgnoreMode, {}, {}, config);
    return stats;
}
uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("kernel library required");
    auto library = dlopen(argv[1], RTLD_NOW);
    if (!library) throw std::runtime_error(dlerror());
    auto instantiate = reinterpret_cast<Instantiate>(dlsym(library, "InstantiateTpcKernel"));
    std::mt19937 random(30); std::normal_distribution<float> normal;
    unsigned failures = 0;
    for (unsigned width : {1280u, 5120u}) for (unsigned rows : {1u, 2u, 6u}) {
        const unsigned count = rows*width;
        std::vector<uint16_t> x(count), weight(width), reference(count), candidate(count);
        std::vector<uint8_t> q_reference(count), q_candidate(count);
        std::vector<float> s_reference(rows), s_candidate(rows);
        for (auto& w : weight) w = bf16(1 + normal(random)*.125f);
        auto x_shape = tensor(DATA_BF16, {width, rows}); auto w_shape = tensor(DATA_BF16, {width});
        auto q_shape = tensor(DATA_F8_143, {width, rows}); auto s_shape = tensor(DATA_F32, {1, rows});
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
            const float factor = pattern == 1 ? 0 : pattern == 2 ? .0001f : pattern == 3 ? 1024 : 1;
            for (auto& value : x) value = bf16(normal(random)*factor);
            const auto norm = run(instantiate, "custom_deepseek_v41_attention_norm_bf16_gaudi2",
                {x_shape, w_shape}, {x_shape}, {x.data(),weight.data(),reference.data()}, {1e-6f});
            const auto quant = run(instantiate, "custom_deepseek_v41_dense_quant_gaudi2",
                {x_shape}, {q_shape,s_shape}, {reference.data(),q_reference.data(),s_reference.data()});
            const bool publish = width == 5120;
            const auto fused = run(instantiate, publish ? "custom_deepseek_v41_attention_norm_quant_gaudi2" :
                "custom_deepseek_v41_qnorm_quant_gaudi2", {x_shape,w_shape},
                publish ? std::vector<Tensor>{q_shape,s_shape,x_shape} : std::vector<Tensor>{q_shape,s_shape},
                publish ? std::vector<void*>{x.data(),weight.data(),q_candidate.data(),s_candidate.data(),candidate.data()} :
                std::vector<void*>{x.data(),weight.data(),q_candidate.data(),s_candidate.data()}, {1e-6f,1.f/width});
            const bool exact = (!publish || reference == candidate) && q_reference == q_candidate && s_reference == s_candidate;
            failures += !exact;
            std::cout << "{\"rows\":" << rows << ",\"width\":" << width << ",\"pattern\":" << pattern << ",\"exact\":" << exact
                      << ",\"separate_vliw\":" << norm.instructionsExecuted+quant.instructionsExecuted
                      << ",\"fused_vliw\":" << fused.instructionsExecuted << "}\n";
        }
    }
    return failures ? 1 : 0;
}
