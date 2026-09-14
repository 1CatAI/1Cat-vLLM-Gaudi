// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41SelectedKVGaudi2 {
public:
    explicit DeepseekV41SelectedKVGaudi2(unsigned ordered = 0, bool valid_only = false,
                                         bool vector = false, bool vector_scales = false)
        : ordered_(ordered), valid_only_(valid_only), vector_(vector), vector_scales_(vector_scales) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                                tpc_lib_api::HabanaKernelInstantiation*);
private:
    unsigned ordered_;
    bool valid_only_;
    bool vector_;
    bool vector_scales_;
};
