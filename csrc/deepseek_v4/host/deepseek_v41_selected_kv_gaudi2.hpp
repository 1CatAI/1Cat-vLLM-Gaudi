// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV41SelectedKVGaudi2 {
    bool vector_scales_;
public:
    explicit DeepseekV41SelectedKVGaudi2(bool vector_scales = false) : vector_scales_(vector_scales) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                                tpc_lib_api::HabanaKernelInstantiation*);
};
