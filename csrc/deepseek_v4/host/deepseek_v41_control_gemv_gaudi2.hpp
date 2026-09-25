// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41ControlGemvGaudi2 {
    bool batch4_;
    bool prefetch_;
public:
    explicit DeepseekV41ControlGemvGaudi2(bool batch4 = false, bool prefetch = false)
        : batch4_(batch4), prefetch_(prefetch) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
