// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41PrefillQScaleRopeGaudi2 {
    bool bf16_product_;
public:
    static constexpr const char* name = "custom_deepseek_v41_prefill_q_scale_rope_gaudi2";
    static constexpr const char* bf16_name = "custom_deepseek_v41_prefill_q_scale_rope_bf16_gaudi2";
    explicit DeepseekV41PrefillQScaleRopeGaudi2(bool bf16_product = false) : bf16_product_(bf16_product) {}
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
