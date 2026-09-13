// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41Csa2PrepGaudi2 {
public:
    explicit DeepseekV41Csa2PrepGaudi2(bool rope) : rope_(rope) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                                 tpc_lib_api::HabanaKernelInstantiation*);
private:
    bool rope_;
};
