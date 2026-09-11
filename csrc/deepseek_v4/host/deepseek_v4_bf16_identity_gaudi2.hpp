// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tpc_kernel_lib_interface.h"

class DeepseekV4BF16IdentityGaudi2 {
    bool v41_;
 public:
    explicit DeepseekV4BF16IdentityGaudi2(bool v41 = false) : v41_(v41) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(
        char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*,
        tpc_lib_api::HabanaKernelInstantiation*);
};
