// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
struct Dsv41BatchMlaMetadataParams { int ratio, swa_rows, tile; };
class DeepseekV41BatchMlaMetadataGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_batch_mla_metadata_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
