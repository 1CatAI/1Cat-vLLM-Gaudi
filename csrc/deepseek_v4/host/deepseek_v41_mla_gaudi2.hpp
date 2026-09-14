// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41MlaGaudi2 {
public:
    explicit DeepseekV41MlaGaudi2(bool gather, bool bf16 = false) : gather_(gather), bf16_(bf16) {}
    static constexpr const char* gather_name = "custom_deepseek_v41_mla_gather_gaudi2";
    static constexpr const char* softmax_name = "custom_deepseek_v41_mla_softmax_gaudi2";
    static constexpr const char* shared_name = "custom_deepseek_v41_mla_shared_kv_gaudi2";
    static constexpr const char* exp_name = "custom_deepseek_v41_mla_exp_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
private:
    bool gather_, bf16_;
};
class DeepseekV41MlaNormalizeGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_mla_normalize_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
