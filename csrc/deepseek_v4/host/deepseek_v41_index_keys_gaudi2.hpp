// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41IndexKeysGaudi2 {
    bool tiled_;
public:
    explicit DeepseekV41IndexKeysGaudi2(bool tiled=false):tiled_(tiled) {}
    static constexpr const char* name="custom_deepseek_v41_index_keys_gaudi2";
    static constexpr const char* tiled_name="custom_deepseek_v41_index_keys_tiled_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,tpc_lib_api::HabanaKernelInstantiation*);
};
