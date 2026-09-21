// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV41FinalCollapseNormGaudi2 {
public:
    static constexpr const char* name =
        "custom_deepseek_v41_final_collapse_norm_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*,
        tpc_lib_api::HabanaKernelInstantiation*);
};
