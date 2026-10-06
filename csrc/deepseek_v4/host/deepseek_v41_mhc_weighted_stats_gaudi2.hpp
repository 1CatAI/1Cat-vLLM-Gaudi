// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"
class DeepseekV41MhcWeightedStatsGaudi2 {
 bool finish_;
public:
 explicit DeepseekV41MhcWeightedStatsGaudi2(bool finish=false):finish_(finish){}
 tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,tpc_lib_api::HabanaKernelInstantiation*);
};
