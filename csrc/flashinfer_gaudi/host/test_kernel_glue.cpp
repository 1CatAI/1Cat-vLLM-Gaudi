// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
#include "dflash2_score_select_i64_bf16_f32_gaudi2.hpp"
#include "dflash2_select_path_i64_f32_gaudi2.hpp"
#include "gdn_mtp_packed_f32_gaudi2.hpp"
#include "gdn_mtp_prepared_f32_gaudi2.hpp"
#include "gdn_packed_decode_f32_gaudi2.hpp"
#include "silu_and_mul_bf16_gaudi2.hpp"
#include "silu_mul_quant_bf16_gaudi2.hpp"
#include "block_fp8_dequant_gaudi2.hpp"

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(
    tpc_lib_api::DeviceId, uint32_t*, tpc_lib_api::GuidInfo*);
extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);

int main() {
    using namespace tpc_lib_api;
    constexpr uint32_t expectedKernelCount = 8;
    uint32_t count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS && count == expectedKernelCount);
    GuidInfo guids[expectedKernelCount]{};
    count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS && count == expectedKernelCount);
    assert(guids[0].name[0] == '\0');
    for (uint32_t capacity = 1; capacity < expectedKernelCount; ++capacity) {
        count = capacity;
        assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_FAILED && count == expectedKernelCount);
        for (uint32_t i = 0; i < expectedKernelCount; ++i) {
            assert(guids[i].name[0] == '\0');
        }
    }
    count = expectedKernelCount;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids) == GLUE_SUCCESS);
    char expectedName[MAX_NODE_NAME]{};
    GdnPackedDecodeF32Gaudi2{}.GetKernelName(expectedName);
    assert(std::strcmp(guids[0].name, expectedName) == 0);
    GdnMtpPackedF32Gaudi2{}.GetKernelName(expectedName);
    assert(std::strcmp(guids[1].name, expectedName) == 0);
    DFlash2SelectPathI64F32Gaudi2{}.GetKernelName(expectedName);
    assert(std::strcmp(guids[2].name, expectedName) == 0);
    DFlash2ScoreSelectI64Bf16F32Gaudi2{}.GetKernelName(expectedName);
    assert(std::strcmp(guids[3].name, expectedName) == 0);
    GdnMtpPreparedF32Gaudi2{}.GetKernelName(expectedName);
    assert(std::strcmp(guids[4].name, expectedName) == 0);
    assert(std::strcmp(guids[5].name, SiluAndMulBf16Gaudi2::name) == 0);
    assert(std::strcmp(guids[6].name, SiluMulQuantBf16Gaudi2::name) == 0);
    assert(std::strcmp(guids[7].name, BlockFp8DequantGaudi2::name) == 0);
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

    Tensor block_inputs[2]{}, block_outputs[1]{};
    TensorAccessPattern block_input_patterns[2]{}, block_output_patterns[1]{};
    params->inputTensors = block_inputs;
    params->outputTensors = block_outputs;
    params->inputTensorNr = 2;
    params->outputTensorNr = 1;
    instance->inputTensorAccessPattern = block_input_patterns;
    instance->outputTensorAccessPattern = block_output_patterns;
    for (auto* tensor : {&block_inputs[0], &block_inputs[1], &block_outputs[0]}) tensor->geometry.dims = 2;
    block_inputs[0].geometry.dataType = DATA_F8_143;
    block_inputs[0].geometry.maxSizes[0] = 384;
    block_inputs[0].geometry.maxSizes[1] = 256;
    block_outputs[0].geometry = block_inputs[0].geometry;
    block_outputs[0].geometry.dataType = DATA_BF16;
    block_inputs[1].geometry.dataType = DATA_F32;
    block_inputs[1].geometry.maxSizes[0] = 3;
    block_inputs[1].geometry.maxSizes[1] = 2;
    BlockFp8DequantGaudi2 block;
    assert(block.GetGcDefinitions(nullptr, instance.get()) == GLUE_FAILED);
    assert(block.GetGcDefinitions(params.get(), nullptr) == GLUE_FAILED);
    instance->kernel.elfSize = 0;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_INSUFFICIENT_ELF_BUFFER);
    elf.resize(instance->kernel.elfSize);
    instance->kernel.kernelElf = elf.data();
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_SUCCESS);
    assert(instance->indexSpaceRank == 2 && instance->indexSpaceGeometry[0] == 2 && instance->indexSpaceGeometry[1] == 2);
    assert(block_input_patterns[0].mapping[0].a == 256 && block_input_patterns[0].mapping[1].end_b == 127);
    assert(block_input_patterns[1].mapping[0].a == 2 && block_input_patterns[1].mapping[1].end_b == 0);
    assert(block_output_patterns[0].mapping[1].a == 128);
    assert(std::memcmp(elf.data(), "\177ELF", 4) == 0);
    instance->kernel.kernelElf = nullptr;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_FAILED);
    instance->kernel.kernelElf = elf.data();
    params->inputTensorNr = 1;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_INPUT_COUNT);
    params->inputTensorNr = 2;
    params->outputTensorNr = 2;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    params->outputTensorNr = 1;
    block_inputs[1].geometry.maxSizes[1] = 3;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    block_inputs[1].geometry.maxSizes[1] = 2;
    block_inputs[0].geometry.maxSizes[0] = 383;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    block_inputs[0].geometry.dataType = DATA_BF16;
    assert(block.GetGcDefinitions(params.get(), instance.get()) == GLUE_INCOMPATIBLE_DATA_TYPE);
    return 0;
}
