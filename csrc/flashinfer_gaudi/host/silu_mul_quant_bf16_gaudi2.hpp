// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class SiluMulQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_silu_mul_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};
