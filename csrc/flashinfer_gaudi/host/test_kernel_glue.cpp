// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
#include "silu_and_mul_bf16_gaudi2.hpp"

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId, uint32_t*, tpc_lib_api::GuidInfo*);

int main() {
    using namespace tpc_lib_api;
    uint32_t count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS && count == 2);
    GuidInfo guids[2]{};
    count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS && count == 2);
    assert(guids[0].name[0] == '\0');
    count = 1;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_FAILED && count == 2);
    assert(guids[0].name[0] == '\0');
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS);
    assert(std::strcmp(guids[1].name, SiluAndMulBf16Gaudi2::name) == 0);
    assert(GetKernelGuids(DEVICE_ID_GAUDI, &count, guids) == GLUE_SUCCESS && count == 0);
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, nullptr, nullptr) == GLUE_FAILED);

    auto params = std::make_unique<HabanaKernelParams>();
    auto instance = std::make_unique<HabanaKernelInstantiation>();
    Tensor inputs[1]{}, outputs[1]{};
    TensorAccessPattern inputPatterns[1]{}, outputPatterns[1]{};
    params->inputTensors = inputs;
    params->outputTensors = outputs;
    instance->inputTensorAccessPattern = inputPatterns;
    instance->outputTensorAccessPattern = outputPatterns;
    params->inputTensorNr = 1;
    params->outputTensorNr = 1;
    auto& input = params->inputTensors[0].geometry;
    auto& output = params->outputTensors[0].geometry;
    input.dims = output.dims = 2;
    input.dataType = output.dataType = DATA_BF16;
    input.maxSizes[0] = 256;
    output.maxSizes[0] = 128;
    input.maxSizes[1] = output.maxSizes[1] = 2;
    SiluAndMulBf16Gaudi2 kernel;
    assert(kernel.GetGcDefinitions(nullptr, instance.get()) == GLUE_FAILED);
    assert(kernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(instance->kernel.elfSize > 0);
    std::vector<unsigned char> elf(instance->kernel.elfSize);
    instance->kernel.kernelElf = elf.data();
    assert(kernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_SUCCESS);
    assert(instance->indexSpaceRank == 2 && instance->indexSpaceGeometry[0] == 1 &&
           instance->indexSpaceGeometry[1] == 2);
    assert(std::memcmp(elf.data(), "\177ELF", 4) == 0);
    input.maxSizes[0] = 192;
    assert(kernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    input.dataType = DATA_F32;
    assert(kernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_DATA_TYPE);
    return 0;
}
