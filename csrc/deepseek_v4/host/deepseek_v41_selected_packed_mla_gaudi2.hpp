// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"
class DeepseekV41SelectedPackedMlaGaudi2 {
    bool vector_;
public:
    static constexpr const char* name="custom_deepseek_v41_selected_packed_mla_gather_gaudi2";
    static constexpr const char* vector_name="custom_deepseek_v41_selected_packed_mla_vector_gaudi2";
    explicit DeepseekV41SelectedPackedMlaGaudi2(bool vector = false) : vector_(vector) {}
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,tpc_lib_api::HabanaKernelInstantiation*);
};
