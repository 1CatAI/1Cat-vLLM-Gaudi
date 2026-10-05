// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV41ExpertTokenWideGaudi2 {
 public:
    explicit DeepseekV41ExpertTokenWideGaudi2(unsigned routes = 2, bool unroll = false, bool aligned = false) : routes_(routes), unroll_(unroll), aligned_(aligned) {}
    static constexpr const char* name = "custom_deepseek_v41_expert_token_wide_sat_fp8_gaudi2";
    static constexpr const char* six_name = "custom_deepseek_v41_expert_token_wide6_sat_fp8_gaudi2";
    static constexpr const char* three_name = "custom_deepseek_v41_expert_token_wide3_sat_fp8_gaudi2";
    static constexpr const char* unroll3_name = "custom_deepseek_v41_expert_token_wide3_unroll_sat_fp8_gaudi2";
    static constexpr const char* unroll6_name = "custom_deepseek_v41_expert_token_wide6_unroll_sat_fp8_gaudi2";
    static constexpr const char* aligned_name = "custom_deepseek_v41_expert_token_wide6_aligned_sat_fp8_gaudi2";
    unsigned routes_;
    bool unroll_;
    bool aligned_;
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
