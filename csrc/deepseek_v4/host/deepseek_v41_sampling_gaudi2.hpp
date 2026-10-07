// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
struct DeepseekV41SamplingParams { int columns,ranks,width; };
class DeepseekV41SamplingGaudi2 {
public:
    explicit DeepseekV41SamplingGaudi2(unsigned mode) : mode_(mode) {}
    static constexpr const char* names[]={"custom_deepseek_v41_sampling_unpack_gaudi2",
        "custom_deepseek_v41_sampling_mask_gaudi2","custom_deepseek_v41_sampling_select_gaudi2"};
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
private:
    unsigned mode_;
};
