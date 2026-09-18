// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41DecodedKVGaudi2 {
public:
    enum Mode { SWA_WRITE, FP4_WRITE, ATTENTION, ATTENTION_BLOCK,
                SWA_PAGED_WRITE, FP4_PAGED_WRITE };
    explicit DeepseekV41DecodedKVGaudi2(Mode mode) : mode_(mode) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
private:
    Mode mode_;
};
