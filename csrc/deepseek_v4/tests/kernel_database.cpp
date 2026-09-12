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
    for (const auto& guid : guids) {
        if (std::strstr(guid.name, "deepseek_v4")) {
            ++custom_count;
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
    assert(custom_count == 72);
    for (const char* required : {
             "custom_deepseek_v41_swa_decoded_write_bf16_gaudi2",
             "custom_deepseek_v41_fp4_decoded_write_bf16_gaudi2",
             "custom_deepseek_v41_decoded_attn_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_vec_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_vec_ordered_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_vec_cache_bf16_gaudi2",
             "custom_deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2",
             "custom_deepseek_v41_mxfp4_prepared_dequant_k128n_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_valid_ordered_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_valid_cache_ordered_bf16_gaudi2",
             "custom_deepseek_v41_rope_bf16_gaudi2",
             "custom_deepseek_v41_c1_indices_i32_gaudi2",
             "custom_deepseek_v41_fp4_pack_g16_bf16_gaudi2",
             "custom_deepseek_v41_fp4_pack_g32_bf16_gaudi2",
             "custom_deepseek_v41_fp4_cache_write_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_cache_ordered_bf16_gaudi2",
             "custom_deepseek_v41_swa_pack_bf16_gaudi2",
             "custom_deepseek_v41_swa_pack_write_bf16_gaudi2",
             "custom_deepseek_v41_selected_kv_ordered_bf16_gaudi2"}) {
        unsigned matches = 0;
        for (const auto& guid : guids) matches += std::strcmp(guid.name, required) == 0;
        assert(matches == 1);
    }
    HabanaKernelParams params{};
    HabanaKernelInstantiation instance{};
    std::strcpy(
        params.guid.name,
        "custom_deepseek_v4_mxfp4_prepared_gate_normal_bf16_gaudi2");
    assert(InstantiateTpcKernel(&params, &instance) == GLUE_INCOMPATIBLE_INPUT_COUNT);
}
