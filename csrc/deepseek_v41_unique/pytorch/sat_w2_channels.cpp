// SPDX-License-Identifier: Apache-2.0
// Private C2-C6 producer/consumer layout; all C1 and communication paths unchanged.
#define DSV41_N256_W13_K_PIPELINE 1
#define DSV41_N256_DOWN_CHANNEL_LAYOUT 1
#define DSV41_N256_DOWN_TOKEN_LAYOUT 1
#define DSV41_N256_W2_SPLIT_GUID "custom_deepseek_v41_w2_channel_decode_gaudi2"
#define DSV41_N256_SCALE_REDUCE_GUID "custom_deepseek_v41_w2_channel_reduce_gaudi2"
#define DSV41_SPLIT_SCALE_OPERATOR custom_deepseek_v41_expert_n256_moe_w2_channels_fp8_gaudi2
#include "sat_split_scale_planes.cpp"
