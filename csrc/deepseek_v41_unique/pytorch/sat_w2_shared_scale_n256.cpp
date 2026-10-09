// SPDX-License-Identifier: Apache-2.0
// C1 deferred shared scaling, with the existing bounded SAT producer intact.
#define DSV41_N256_W13_K_PIPELINE 1
#define DSV41_N256_SHARED_SCALE_INPUT 1
#define DSV41_N256_SCALE_REDUCE_GUID "custom_deepseek_v41_w2_reduce_shared_scale_n256_gaudi2"
#define DSV41_SPLIT_SCALE_OPERATOR custom_deepseek_v41_expert_n256_moe_w2_shared_scale_n256_fp8_gaudi2
#include "sat_split_scale_planes.cpp"
