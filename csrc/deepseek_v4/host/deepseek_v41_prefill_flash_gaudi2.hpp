// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV41PrefillFlashGaudi2 {
    bool mask_;
public:
    explicit DeepseekV41PrefillFlashGaudi2(bool mask = false) : mask_(mask) {}
    static constexpr const char* kv_name = "custom_deepseek_v41_prefill_flash_kv_gaudi2";
    static constexpr const char* mask_name = "custom_deepseek_v41_prefill_flash_mask_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
