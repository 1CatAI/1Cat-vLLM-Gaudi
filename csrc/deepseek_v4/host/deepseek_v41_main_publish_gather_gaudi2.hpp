// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"
class DeepseekV41MainPublishGatherGaudi2 {
    bool vector_;
    bool vector_mask_;
    bool native_codec_;
    bool tensor_mask_;
public:
    explicit DeepseekV41MainPublishGatherGaudi2(bool vector = false, bool vector_mask = false, bool native_codec = false, bool tensor_mask = false)
        : vector_(vector), vector_mask_(vector_mask), native_codec_(native_codec), tensor_mask_(tensor_mask) {}
    static constexpr const char* name = "custom_deepseek_v41_main_publish_gather_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
