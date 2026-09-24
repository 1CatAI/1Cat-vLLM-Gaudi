// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV41PrefillSparseMlaGaudi2 {
public:
    enum Kind { Gather = 0, Exp = 1, Normalize = 2 };
    explicit DeepseekV41PrefillSparseMlaGaudi2(Kind kind) : kind_(kind) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                                 tpc_lib_api::HabanaKernelInstantiation*);
private:
    Kind kind_;
};
