// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41DenseGaudi2 {
public:
    explicit DeepseekV41DenseGaudi2(bool quant) : quant_(quant) {}
    static constexpr const char* quant_name = "custom_deepseek_v41_dense_quant_gaudi2";
    static constexpr const char* scale_name = "custom_deepseek_v41_dense_scale_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
private:
    bool quant_;
};
