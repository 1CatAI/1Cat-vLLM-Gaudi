// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV41MhcGatesGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_mhc_gates_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};
