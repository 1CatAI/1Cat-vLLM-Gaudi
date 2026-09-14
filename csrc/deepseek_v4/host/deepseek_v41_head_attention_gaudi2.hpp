// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41HeadAttentionGaudi2 {
    int phase_;
public:
    explicit DeepseekV41HeadAttentionGaudi2(int phase) : phase_(phase) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
