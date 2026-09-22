// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
class DeepseekV41WoaGaudi2 {
public:
    explicit DeepseekV41WoaGaudi2(bool quant, bool rope = false, bool roundtrip = false,
                                 bool product = false, bool wide = false)
        : quant_(quant), rope_(rope), roundtrip_(roundtrip), product_(product), wide_(wide) {}
    static constexpr const char* quant_name = "custom_deepseek_v41_woa_quant_gaudi2";
    static constexpr const char* rope_quant_name = "custom_deepseek_v41_woa_rope_quant_gaudi2";
    static constexpr const char* product_rope_quant_name =
        "custom_deepseek_v41_mla_product_rope_quant_gaudi2";
    static constexpr const char* scale_name = "custom_deepseek_v41_woa_scale_gaudi2";
    static constexpr const char* scale_roundtrip_name = "custom_deepseek_v41_woa_scale_roundtrip_gaudi2";
    static constexpr const char* scale_roundtrip_wide_name = "custom_deepseek_v41_woa_scale_roundtrip_wide_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
private:
    bool quant_;
    bool rope_;
    bool roundtrip_;
    bool product_;
    bool wide_;
};
