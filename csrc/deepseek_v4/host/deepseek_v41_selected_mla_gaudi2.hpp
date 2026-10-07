// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV41SelectedMlaGaudi2 {
    bool gather_;
    bool register_row_;
public:
    explicit DeepseekV41SelectedMlaGaudi2(bool gather, bool register_row = false) : gather_(gather), register_row_(register_row) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
