// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41CompressorBatchGaudi2 {
public:
    static constexpr const char* name = "custom_deepseek_v41_compressor_batch_bf16_gaudi2";
    static constexpr const char* gatherName = "custom_deepseek_v41_compressor_batch_gather_f32_gaudi2";
    explicit DeepseekV41CompressorBatchGaudi2(bool gather = false) : gather_(gather) {}
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*,
                                               tpc_lib_api::HabanaKernelInstantiation*);
private:
    bool gather_;
};
