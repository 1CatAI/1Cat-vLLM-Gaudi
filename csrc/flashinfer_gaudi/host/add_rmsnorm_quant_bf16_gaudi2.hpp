// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
#include "../norm_quant_params.h"

class AddRmsNormQuantBf16Gaudi2 {
public:
    static constexpr const char* name = kAddRmsNormQuantGuid;
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};
