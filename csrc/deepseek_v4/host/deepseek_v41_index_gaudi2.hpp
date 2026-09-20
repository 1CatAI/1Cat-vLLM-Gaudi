// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41IndexGaudi2 {
public:
    explicit DeepseekV41IndexGaudi2(unsigned mode): mode_(mode) {}
    static constexpr const char* names[] = {"custom_deepseek_v41_index_scores_gaudi2",
        "custom_deepseek_v41_index_threshold_gaudi2", "custom_deepseek_v41_index_emit_gaudi2",
        "custom_deepseek_v41_index_scores_decoded_gaudi2"};
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,tpc_lib_api::HabanaKernelInstantiation*);
private:
    unsigned mode_;
};
