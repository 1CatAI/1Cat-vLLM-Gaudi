// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41DecodeCoordinatesGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_decode_coordinates_gaudi2";
    tpc_lib_api::GlueCodeReturn GetKernelName(char out[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
};
