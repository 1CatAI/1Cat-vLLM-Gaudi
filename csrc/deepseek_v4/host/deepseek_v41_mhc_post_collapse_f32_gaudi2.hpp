// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"
class DeepseekV41MhcPostCollapseF32Gaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_mhc_post_collapse_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
