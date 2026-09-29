// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41RoutePackGaudi2 {
    int mode_;
public:
    explicit DeepseekV41RoutePackGaudi2(int mode = 0) : mode_(mode) {}
    const char* name() const {
        return mode_ == 2 ? "custom_deepseek_v41_route_order_gaudi2" : mode_ == 1 ? "custom_deepseek_v41_route_reduce_gaudi2" : "custom_deepseek_v41_route_pack_gaudi2";
    }
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
