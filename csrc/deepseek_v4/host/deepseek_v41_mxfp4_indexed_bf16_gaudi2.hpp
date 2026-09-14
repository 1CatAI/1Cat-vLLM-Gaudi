// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tpc_kernel_lib_interface.h"

class DeepseekV41Mxfp4IndexedBF16Gaudi2 {
 public:
    enum Stage { FC1, FC2 };

    DeepseekV41Mxfp4IndexedBF16Gaudi2(Stage stage, bool normal)
        : stage_(stage), normal_(normal) {}

    tpc_lib_api::GlueCodeReturn GetKernelName(
        char kernelName[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*,
        tpc_lib_api::HabanaKernelInstantiation*);

 private:
    Stage stage_;
    bool normal_;
};
