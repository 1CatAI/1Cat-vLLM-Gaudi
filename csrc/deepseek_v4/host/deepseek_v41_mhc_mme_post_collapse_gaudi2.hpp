// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41MhcMmePostCollapseGaudi2 {
    int mode_;
public:
    explicit DeepseekV41MhcMmePostCollapseGaudi2(int mode=0) : mode_(mode) {}
    static constexpr const char* name="custom_deepseek_v41_mhc_mme_post_collapse_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
