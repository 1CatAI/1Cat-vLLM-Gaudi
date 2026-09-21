// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV41DensePairScaleGaudi2 {
public:
    static constexpr const char* name =
        "custom_deepseek_v41_dense_pair_scale_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*,
        tpc_lib_api::HabanaKernelInstantiation*);
};
