// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <vector>
#include "tpc_kernel_lib_interface.h"

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId, uint32_t*, tpc_lib_api::GuidInfo*);
extern "C" tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(
    tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
extern "C" tpc_lib_api::GlueCodeReturn GetSupportedDataLayouts(
    const tpc_lib_api::HabanaKernelParams*, tpc_lib_api::NodeDataLayouts*, uint32_t*);

int main() {
    using namespace tpc_lib_api;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, nullptr, nullptr) == GLUE_FAILED);
    uint32_t count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS);
    assert(count > 39);
    std::vector<GuidInfo> guids(count);
    uint32_t capacity = 1;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &capacity, guids.data()) == GLUE_FAILED);
    assert(capacity == count);
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids.data()) == GLUE_SUCCESS);
    uint32_t custom_count = 0;
    bool selected_kv_seen = false;
    bool device_engram_seen = false;
    for (const auto& guid : guids) {
        if (std::strstr(guid.name, "deepseek_v4")) {
            ++custom_count;
            selected_kv_seen |= std::strcmp(
                guid.name, "custom_deepseek_v41_selected_kv_bf16_gaudi2") == 0;
            device_engram_seen |= std::strcmp(
                guid.name, "custom_deepseek_v41_engram_hash_gather_bf16_gaudi2") == 0;
            HabanaKernelParams query{};
            query.guid = guid;
            uint32_t layouts_count = 0;
            assert(GetSupportedDataLayouts(&query, nullptr, &layouts_count) == GLUE_SUCCESS);
            assert(layouts_count == 1);
            TensorDataLayout input{}, output{};
            NodeDataLayouts layouts{};
            layouts.inputs = &input;
            layouts.outputs = &output;
            layouts.inputTensorNr = layouts.outputTensorNr = 1;
            assert(GetSupportedDataLayouts(&query, &layouts, &layouts_count) == GLUE_SUCCESS);
            for (unsigned i = 0; i < MAX_TENSOR_DIM; ++i) {
                assert(input.layout[i] == 'x' && output.layout[i] == 'x');
            }
        }
    }
    assert(custom_count >= 52);
    assert(selected_kv_seen);
    assert(device_engram_seen);
    {
        HabanaKernelParams p{};
        HabanaKernelInstantiation q{};
        Tensor inputs[5]{}, outputs[3]{};
        TensorAccessPattern ia[5]{}, oa[3]{};
        p.inputTensors = inputs; p.inputTensorNr = 5;
        p.outputTensors = outputs; p.outputTensorNr = 3;
        q.inputTensorAccessPattern = ia; q.outputTensorAccessPattern = oa;
        std::strcpy(p.guid.name, "custom_deepseek_v41_paged_mla_gather_gaudi2");
        auto set = [](Tensor& t, TensorDataType type, std::initializer_list<uint64_t> dims) {
            t.geometry.dataType = type; t.geometry.dims = dims.size();
            unsigned d = 0;
            for (auto value : dims) t.geometry.maxSizes[d++] = value;
        };
        set(inputs[0], DATA_U8, {528, 256});
        set(inputs[1], DATA_U8, {288, 6179072});
        set(inputs[2], DATA_I32, {768, 1});
        set(inputs[3], DATA_I32, {640, 1});
        set(inputs[4], DATA_I32, {1});
        set(outputs[0], DATA_BF16, {512, 640, 1});
        set(outputs[1], DATA_F32, {512, 640, 1});
        set(outputs[2], DATA_F32, {640, 1});
        assert(InstantiateTpcKernel(&p, &q) == GLUE_INSUFFICIENT_ELF_BUFFER);
        assert(q.indexSpaceGeometry[0] == 640 && q.indexSpaceGeometry[1] == 1);
        assert(ia[2].allRequired && !ia[3].allRequired);
        assert(oa[0].mapping[0].end_b == 511 && oa[1].mapping[1].a == 1);
        inputs[3].geometry.maxSizes[1] = 2;
        assert(InstantiateTpcKernel(&p, &q) == GLUE_INCOMPATIBLE_INPUT_SIZE);
        inputs[3].geometry.maxSizes[1] = 1;
        outputs[1].geometry.dataType = DATA_BF16;
        assert(InstantiateTpcKernel(&p, &q) == GLUE_INCOMPATIBLE_OUTPUT_SIZE);
    }
    HabanaKernelParams params{};
    HabanaKernelInstantiation instance{};
    std::strcpy(
        params.guid.name,
        "custom_deepseek_v4_mxfp4_prepared_gate_normal_bf16_gaudi2");
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_COUNT);

    // A final W13 window must address the last two blocks without reading
    // past the compressed row or changing the MME reduction dimension.
    Tensor inputs[4]{}, outputs[1]{};
    params = {};
    std::strcpy(params.guid.name, "custom_deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2");
    params.inputTensors = inputs; params.inputTensorNr = 4;
    params.outputTensors = outputs; params.outputTensorNr = 1;
    auto shape = [](Tensor& tensor, TensorDataType type, std::initializer_list<uint64_t> sizes) {
        tensor.geometry.dataType = type; tensor.geometry.dims = sizes.size();
        unsigned dim = 0;
        for (auto size : sizes) tensor.geometry.maxSizes[dim++] = size;
    };
    shape(inputs[0], DATA_I32, {36, 1});
    shape(inputs[1], DATA_I16, {163840, 18, 384});
    shape(inputs[2], DATA_BF16, {20480, 18, 384});
    shape(inputs[3], DATA_BF16, {128});
    shape(outputs[0], DATA_BF16, {256, 5120, 36});
    int32_t window[2] = {16, 2};
    params.nodeParams.nodeParams = window; params.nodeParams.nodeParamsSize = sizeof(window);
    TensorAccessPattern inputPatterns[4]{}, outputPatterns[1]{};
    instance = {};
    instance.inputTensorAccessPattern = inputPatterns;
    instance.outputTensorAccessPattern = outputPatterns;
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(instance.indexSpaceGeometry[0] == 2 && instance.indexSpaceGeometry[2] == 40);
    assert(instance.kernel.paramsNr == 1 && instance.kernel.scalarParams[0] == 16);
    assert(inputPatterns[1].mapping[1].start_b == 16 && inputPatterns[1].mapping[1].a == 1);
    assert(outputPatterns[0].mapping[0].start_b == 0);
    window[0] = 17;
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    window[0] = -1;
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    // Head-vector attention must retain complete indirect KV ranges and
    // mask the final score tile without widening its token interval.
    Tensor headInputs[5]{}, headOutputs[2]{};
    TensorAccessPattern headIA[5]{}, headOA[2]{};
    params = {}; instance = {};
    params.inputTensors = headInputs; params.outputTensors = headOutputs;
    instance.inputTensorAccessPattern = headIA; instance.outputTensorAccessPattern = headOA;
    std::strcpy(params.guid.name, "custom_deepseek_v41_attn_scores_f32_gaudi2");
    params.inputTensorNr = 5; params.outputTensorNr = 1;
    shape(headInputs[0], DATA_BF16, {512, 32, 6}); shape(headInputs[1], DATA_BF16, {512, 768});
    shape(headInputs[2], DATA_I32, {639, 6}); shape(headInputs[3], DATA_F32, {1});
    shape(headInputs[4], DATA_I32, {6}); shape(headOutputs[0], DATA_F32, {32, 639, 6});
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(instance.indexSpaceGeometry[0] == 32 && instance.indexSpaceGeometry[1] == 80);
    assert(headIA[1].sparseAccess && headIA[1].mapping[1].end_b == 767);
    assert(headOA[0].mapping[1].a == 8 && headOA[0].mapping[2].indexSpaceDim == 2);
    std::strcpy(params.guid.name, "custom_deepseek_v41_attn_recurrence_f32_gaudi2");
    params.inputTensorNr = 4; params.outputTensorNr = 2;
    shape(headInputs[0], DATA_F32, {32, 639, 6}); shape(headInputs[1], DATA_I32, {639, 6});
    shape(headInputs[2], DATA_F32, {32}); shape(headInputs[3], DATA_I32, {6});
    shape(headOutputs[0], DATA_F32, {32, 2, 639, 6}); shape(headOutputs[1], DATA_F32, {32, 6});
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    int32_t kvRows = 768;
    params.nodeParams.nodeParams = &kvRows; params.nodeParams.nodeParamsSize = sizeof(kvRows);
    instance.kernel.elfSize = 0;
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(instance.kernel.paramsNr == 1 && instance.kernel.scalarParams[0] == 768);
    kvRows = 0;
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
    std::strcpy(params.guid.name, "custom_deepseek_v41_attn_values_f32_gaudi2");
    params.inputTensorNr = 5; params.outputTensorNr = 1;
    shape(headInputs[0], DATA_BF16, {512, 768}); shape(headInputs[1], DATA_I32, {639, 6});
    shape(headInputs[2], DATA_I32, {6}); shape(headInputs[3], DATA_F32, {32, 2, 639, 6});
    shape(headInputs[4], DATA_F32, {32, 6}); shape(headOutputs[0], DATA_F32, {32, 512, 6});
    instance.kernel.elfSize = 0;
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
    assert(instance.indexSpaceGeometry[0] == 64 && instance.indexSpaceGeometry[2] == 6);
    assert(headIA[0].sparseAccess && headOA[0].mapping[1].a == 8);

    // Compact TP4 scales must cover both original group bytes and the final
    // channel-code row, including when Synapse slices the K index space.
    for (const auto& test : std::vector<std::vector<uint64_t>>{
             {1280, 5120, 6}, {5120, 640, 6}, {1280, 5120, 1}}) {
        const auto n = test[0], k = test[1], slots = test[2];
        for (const auto* guid : {"custom_deepseek_v41_expert_n256_fp8_gaudi2",
                                 "custom_deepseek_v41_expert_n256_bf16_gaudi2"}) {
            Tensor expertInputs[4]{}, expertOutputs[1]{};
            TensorAccessPattern expertIA[4]{}, expertOA[1]{};
            params = {}; instance = {};
            params.inputTensors = expertInputs; params.inputTensorNr = 4;
            params.outputTensors = expertOutputs; params.outputTensorNr = 1;
            instance.inputTensorAccessPattern = expertIA; instance.outputTensorAccessPattern = expertOA;
            std::strcpy(params.guid.name, guid);
            shape(expertInputs[0], DATA_I32, {slots, 1});
            shape(expertInputs[1], DATA_I16, {k * 64, n / 256, 384});
            shape(expertInputs[2], DATA_I16, {k * 4 + 128, n / 256, 384});
            shape(expertInputs[3], DATA_BF16, {128});
            shape(expertOutputs[0], std::strstr(guid, "_fp8_") ? DATA_F8_143 : DATA_BF16, {n, k, slots});
            assert(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
            assert(expertIA[2].mapping[0].a == 0 && expertIA[2].mapping[0].end_b == k * 4 + 127);
            const unsigned tile = std::strstr(guid, "_fp8_") && slots % 2 == 0 ? 2 : 1;
            assert(expertOA[0].mapping[1].a == 128 && expertOA[0].mapping[2].a == tile);
            expertInputs[2].geometry.maxSizes[0] -= 1;
            assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_SIZE);
        }
    }

    // Keep routed expert identity explicit when the TP4 decoder processes a
    // bounded pair. Odd slot counts and the TP2 shapes retain their old map.
    for (const auto& test : std::vector<std::vector<uint64_t>>{
             {1280, 5120, 6, 2}, {5120, 640, 6, 2}, {1280, 5120, 12, 2},
             {5120, 640, 36, 2}, {1280, 5120, 1, 1}, {1280, 5120, 3, 1},
             {2560, 5120, 6, 1}, {5120, 1280, 6, 1}}) {
        const auto n = test[0], k = test[1], slots = test[2], tile = test[3];
        Tensor expertInputs[4]{}, expertOutputs[1]{};
        TensorAccessPattern expertIA[4]{}, expertOA[1]{};
        params = {}; instance = {};
        params.inputTensors = expertInputs; params.inputTensorNr = 4;
        params.outputTensors = expertOutputs; params.outputTensorNr = 1;
        instance.inputTensorAccessPattern = expertIA; instance.outputTensorAccessPattern = expertOA;
        std::strcpy(params.guid.name, "custom_deepseek_v41_expert_n256_fp8_gaudi2");
        shape(expertInputs[0], DATA_I32, {slots, 1});
        shape(expertInputs[1], DATA_I16, {k * 64, n / 256, 384});
        shape(expertInputs[2], DATA_I16, {k * 8, n / 256, 384});
        shape(expertInputs[3], DATA_BF16, {128});
        shape(expertOutputs[0], DATA_F8_143, {n, k, slots});
        assert(InstantiateTpcKernel(&params, &instance) == GLUE_INSUFFICIENT_ELF_BUFFER);
        assert(instance.indexSpaceRank == 3 && instance.indexSpaceGeometry[1] == slots / tile);
        assert(expertIA[0].mapping[0].a == tile && expertIA[0].mapping[0].end_b == tile - 1);
        assert(expertOA[0].mapping[2].a == tile && expertOA[0].mapping[2].end_b == tile - 1);
        assert(expertIA[1].mapping[2].end_b == 383 && expertIA[2].mapping[2].end_b == 383);
    }
}
