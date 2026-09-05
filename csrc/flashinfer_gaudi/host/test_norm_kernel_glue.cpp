// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <limits>
#include <vector>
#include "add_rmsnorm_quant_bf16_gaudi2.hpp"

int main() {
    using namespace tpc_lib_api;
    HabanaKernelParams params{};
    HabanaKernelInstantiation instance{};
    Tensor inputs[3]{}, outputs[4]{};
    TensorAccessPattern input_patterns[3]{}, output_patterns[4]{};
    AddRmsNormQuantParams scalars{1e-6f, 1.0f / 5120, 0.004180908203125f};
    params.deviceId = DEVICE_ID_GAUDI2;
    params.nodeParams = {&scalars, sizeof(scalars)};
    params.inputTensorNr = 3;
    params.outputTensorNr = 4;
    params.inputTensors = inputs;
    params.outputTensors = outputs;
    instance.inputTensorAccessPattern = input_patterns;
    instance.outputTensorAccessPattern = output_patterns;
    for (int i = 0; i < 3; ++i) {
        inputs[i].geometry.dims = i == 2 ? 1 : 2;
        inputs[i].geometry.dataType = DATA_BF16;
        inputs[i].geometry.maxSizes[0] = 5120;
        inputs[i].geometry.maxSizes[1] = i == 2 ? 1 : 8;
    }
    for (int i = 0; i < 4; ++i) {
        outputs[i].geometry.dims = 2;
        outputs[i].geometry.dataType = i == 0 ? DATA_F8_143 : i == 1 ? DATA_F32 : DATA_BF16;
        outputs[i].geometry.maxSizes[0] = i == 1 ? 1 : 5120;
        outputs[i].geometry.maxSizes[1] = 8;
    }
    AddRmsNormQuantBf16Gaudi2 kernel;
    assert(kernel.GetGcDefinitions(nullptr, &instance) == GLUE_FAILED);
    assert(kernel.GetGcDefinitions(&params, nullptr) == GLUE_FAILED);
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
    std::vector<unsigned char> elf(instance.kernel.elfSize);
    instance.kernel.kernelElf = elf.data();
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_SUCCESS);
    assert(std::memcmp(elf.data(), "\177ELF", 4) == 0);
    assert(instance.indexSpaceRank == 1 && instance.indexSpaceGeometry[0] == 8);
    assert(instance.kernel.paramsNr == 3 && std::memcmp(instance.kernel.scalarParams, &scalars, sizeof(scalars)) == 0);
    assert(input_patterns[0].mapping[0].a == 0 && input_patterns[0].mapping[0].end_b == 5119);
    assert(input_patterns[0].mapping[1].a == 1 && input_patterns[1].mapping[1].a == 1);
    assert(input_patterns[2].mapping[0].end_b == 5119 && input_patterns[2].mapping[1].a == 0);
    assert(output_patterns[1].mapping[0].end_b == 0 && output_patterns[1].mapping[1].a == 1);
    const auto small_elf = elf;
    for (unsigned width : {8192u, 8320u, 17408u}) {
        for (auto& tensor : inputs) tensor.geometry.maxSizes[0] = width;
        for (int index = 0; index < 4; ++index) outputs[index].geometry.maxSizes[0] = index == 1 ? 1 : width;
        scalars.inverse_width = 1.0f / width;
        instance.kernel.kernelElf = nullptr;
        instance.kernel.elfSize = 0;
        assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
        std::vector<unsigned char> variant(instance.kernel.elfSize);
        instance.kernel.kernelElf = variant.data();
        assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_SUCCESS);
        assert((variant == small_elf) == (width <= 8192));
        assert(input_patterns[0].mapping[0].end_b == static_cast<int>(width - 1));
    }
    for (auto& tensor : inputs) tensor.geometry.maxSizes[0] = 5120;
    for (int index = 0; index < 4; ++index) outputs[index].geometry.maxSizes[0] = index == 1 ? 1 : 5120;
    scalars.inverse_width = 1.0f / 5120;
    instance.kernel.elfSize = elf.size();
    instance.kernel.kernelElf = nullptr;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_FAILED);
    instance.kernel.kernelElf = elf.data();
    params.inputTensorNr = 2;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_COUNT);
    params.inputTensorNr = 3;
    params.outputTensorNr = 3;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    params.outputTensorNr = 4;
    params.nodeParams.nodeParamsSize = 4;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_FAILED);
    params.nodeParams.nodeParamsSize = sizeof(scalars);
    for (float epsilon : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min()}) {
        scalars.epsilon = epsilon;
        assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    }
    scalars.epsilon = 1e-6f;
    scalars.inverse_width = 1.0f;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    scalars.inverse_width = 1.0f / 5120;
    scalars.inverse_range = 1.0f;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    scalars.inverse_range = 1.0f / 240.0f;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_SUCCESS);
    scalars.inverse_range = 0.004180908203125f;
    inputs[0].geometry.maxSizes[0] = 5119;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    inputs[0].geometry.maxSizes[0] = 5120;
    inputs[1].geometry.maxSizes[1] = 7;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    inputs[1].geometry.maxSizes[1] = 8;
    inputs[2].geometry.dataType = DATA_F32;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_DATA_TYPE);
    inputs[2].geometry.dataType = DATA_BF16;
    outputs[1].geometry.maxSizes[0] = 8;
    assert(kernel.GetGcDefinitions(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    return 0;
}
