// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41ExpertPairSiluQuantGaudi2 {
 public:
    explicit DeepseekV41ExpertPairSiluQuantGaudi2(bool flat=false) : flat_(flat) {}
    bool flat_;
    static constexpr const char* flat_name="custom_deepseek_v41_expert_flat_silu_quant_gaudi2";
    static constexpr const char* name="custom_deepseek_v41_expert_pair_silu_quant_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,tpc_lib_api::HabanaKernelInstantiation*);
};
