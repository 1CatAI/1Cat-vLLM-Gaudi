// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_permuted_gaudi2.hpp"
#include "deepseek_v41_expert_n256_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_prefill_permuted_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_permuted_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillPermutedGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    const unsigned capacity = instance->kernel.elfSize;
    // The tensor geometry and access intervals are unchanged. Only columns
    // within each N256 output tile use the converter's even/odd lane order.
    const auto status = DeepseekV41ExpertN256Gaudi2(
        DeepseekV41ExpertN256Gaudi2::DeadNormalBF16).GetGcDefinitions(params, instance);
    if (status != GLUE_SUCCESS && status != GLUE_INSUFFICIENT_ELF_BUFFER) return status;
    const auto* start = &_binary___deepseek_v41_prefill_permuted_bf16_gaudi2_o_start;
    const auto* end = &_binary___deepseek_v41_prefill_permuted_bf16_gaudi2_o_end;
    instance->kernel.elfSize = end - start;
    if (capacity < instance->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(instance->kernel.kernelElf, start, instance->kernel.elfSize);
    return GLUE_SUCCESS;
}
