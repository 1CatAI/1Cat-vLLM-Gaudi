// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 {
    bool normal_;
    int half_;
    bool v41_;
    bool k128_;
    bool pipeline_;

 public:
    explicit DeepseekV4Mxfp4PreparedDequantBF16Gaudi2(
        bool normal = false, int half = -1, bool v41 = false, bool k128 = false, bool pipeline = false)
        : normal_(normal), half_(half), v41_(v41), k128_(k128), pipeline_(pipeline) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
