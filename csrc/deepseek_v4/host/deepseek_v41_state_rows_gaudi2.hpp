// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41StateRowsGaudi2 {
    unsigned mode_;
public:
    explicit DeepseekV41StateRowsGaudi2(unsigned mode):mode_(mode) {}
    static const char* names[6];
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,tpc_lib_api::HabanaKernelInstantiation*);
};
