// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41CandidateCoordinatesGaudi2 {
public:
    explicit DeepseekV41CandidateCoordinatesGaudi2(bool whole=false) : whole_(whole) {}
    static constexpr const char* name = "custom_deepseek_v41_candidate_coordinates_gaudi2";
    static constexpr const char* global_name = "custom_deepseek_v41_candidate_coordinates_global_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
private:
    bool whole_;
};
