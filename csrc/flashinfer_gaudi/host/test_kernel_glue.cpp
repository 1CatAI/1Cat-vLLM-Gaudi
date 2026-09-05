// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
#include "silu_and_mul_bf16_gaudi2.hpp"
#include "silu_mul_quant_bf16_gaudi2.hpp"

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId, uint32_t*, tpc_lib_api::GuidInfo*);
extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);

int main() {
    using namespace tpc_lib_api;
    uint32_t count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS && count == 3);
    GuidInfo guids[3]{};
    count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS && count == 3);
    assert(guids[0].name[0] == '\0');
    count = 1;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_FAILED && count == 3);
    assert(guids[0].name[0] == '\0');
    count = 2;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_FAILED && count == 3);
    assert(guids[0].name[0] == '\0' && guids[1].name[0] == '\0');
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS);
    assert(std::strcmp(guids[1].name, SiluAndMulBf16Gaudi2::name) == 0);
    assert(std::strcmp(guids[2].name, SiluMulQuantBf16Gaudi2::name) == 0);
    assert(GetKernelGuids(DEVICE_ID_GAUDI, &count, guids) == GLUE_SUCCESS && count == 0);
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, nullptr, nullptr) == GLUE_FAILED);

    auto params = std::make_unique<HabanaKernelParams>();
    auto instance = std::make_unique<HabanaKernelInstantiation>();
    assert(InstantiateTpcKernel(nullptr, instance.get()) == GLUE_FAILED);
    assert(InstantiateTpcKernel(params.get(), nullptr) == GLUE_FAILED);
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

    Tensor quantOutputs[2]{}, quantInputs[2]{};
    TensorAccessPattern quantInputPatterns[2]{};
    TensorAccessPattern quantPatterns[2]{};
    params->outputTensors = quantOutputs;
    params->outputTensorNr = 2;
    instance->outputTensorAccessPattern = quantPatterns;
    input.dataType = DATA_BF16;
    input.maxSizes[0] = 4096;
    quantInputs[0].geometry = input;
    quantInputs[1].geometry = input;
    quantInputs[1].geometry.maxSizes[0] = 2048;
    params->inputTensors = quantInputs;
    params->inputTensorNr = 2;
    instance->inputTensorAccessPattern = quantInputPatterns;
    quantOutputs[0].geometry = input;
    quantOutputs[0].geometry.dataType = DATA_F8_143;
    quantOutputs[0].geometry.maxSizes[0] = 2048;
    quantOutputs[1].geometry = input;
    quantOutputs[1].geometry.dataType = DATA_F32;
    quantOutputs[1].geometry.maxSizes[0] = 1;
    SiluMulQuantBf16Gaudi2 quantKernel;
    assert(quantKernel.GetGcDefinitions(nullptr, instance.get()) == GLUE_FAILED);
    assert(quantKernel.GetGcDefinitions(params.get(), nullptr) == GLUE_FAILED);
    params->inputTensorNr = 1;
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_INPUT_COUNT);
    params->inputTensorNr = 2;
    params->outputTensorNr = 1;
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    params->outputTensorNr = 2;
    instance->kernel.elfSize = 0;
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INSUFFICIENT_ELF_BUFFER);
    elf.resize(instance->kernel.elfSize);
    instance->kernel.kernelElf = elf.data();
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_SUCCESS);
    assert(instance->indexSpaceRank == 1 && instance->indexSpaceGeometry[0] == 2);
    assert(quantPatterns[0].mapping[0].a == 0 && quantPatterns[0].mapping[0].end_b == 2047);
    assert(quantPatterns[1].mapping[0].end_b == 0 && quantPatterns[1].mapping[1].a == 1);
    assert(quantInputPatterns[0].mapping[0].start_b == 2048 && quantInputPatterns[0].mapping[0].end_b == 4095);
    assert(quantInputPatterns[1].mapping[0].start_b == 0 && quantInputPatterns[1].mapping[0].end_b == 2047);
    assert(std::memcmp(elf.data(), "\177ELF", 4) == 0);
    instance->kernel.kernelElf = nullptr;
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_FAILED);
    instance->kernel.kernelElf = elf.data();
    quantOutputs[1].geometry.maxSizes[0] = 2;
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    quantOutputs[1].geometry.maxSizes[0] = 1;
    quantOutputs[0].geometry.dataType = DATA_BF16;
    assert(quantKernel.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_DATA_TYPE);
    return 0;
}
