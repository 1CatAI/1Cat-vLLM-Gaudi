// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41Fp4PackGaudi2 {
public:
    // Mode 16/32 is a pure encoder; mode 0 updates the main and index caches.
    explicit DeepseekV41Fp4PackGaudi2(unsigned mode) : mode_(mode) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
private:
    unsigned mode_;
};
