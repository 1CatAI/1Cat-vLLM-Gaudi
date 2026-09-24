// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41PrefillTopkGaudi2 {
    bool emit_;
public:
    explicit DeepseekV41PrefillTopkGaudi2(bool emit) : emit_(emit) {}
    static constexpr const char* threshold_name = "custom_deepseek_v41_prefill_topk_threshold_gaudi2";
    static constexpr const char* emit_name = "custom_deepseek_v41_prefill_topk_emit_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
