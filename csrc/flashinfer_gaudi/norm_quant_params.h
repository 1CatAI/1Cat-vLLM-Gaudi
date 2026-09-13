// SPDX-License-Identifier: Apache-2.0
#pragma once

inline constexpr const char* kAddRmsNormQuantGuid = "flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2";
struct AddRmsNormQuantParams {
    float epsilon;
    float inverse_width;
    float inverse_range;
};
static_assert(sizeof(AddRmsNormQuantParams) == 12, "Norm-quant scalar ABI must contain three FP32 values");
