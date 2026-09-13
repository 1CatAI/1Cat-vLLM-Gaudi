// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV4Mxfp4IndexedDequantBF16Gaudi2 {
    bool normal_;
 public:
    explicit DeepseekV4Mxfp4IndexedDequantBF16Gaudi2(bool normal = false) : normal_(normal) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
