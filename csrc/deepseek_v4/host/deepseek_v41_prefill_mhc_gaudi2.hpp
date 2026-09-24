// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41PrefillMhcGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_prefill_mhc_post_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
class DeepseekV41PrefillMhcCollapseGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_prefill_mhc_collapse_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
class DeepseekV41PrefillMhcRrmsGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_prefill_mhc_rrms_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
class DeepseekV41PrefillMhcPostPrepareGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_prefill_mhc_post_prepare_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
