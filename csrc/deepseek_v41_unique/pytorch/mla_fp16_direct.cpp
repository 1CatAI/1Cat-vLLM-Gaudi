// SPDX-License-Identifier: Apache-2.0
// Distinct additive producer format; C1 and ordinary FP32 cache schemas stay intact.
#define DSV41_MAIN_FP16_DIRECT 1
#define DSV41_MAIN_FP16_PV 1
#define DSV41_MAIN_SPLIT_REUSE 1
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_main_fp16_direct_publish_mla_gaudi2"
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_main_fp16_direct_reuse_mla_gaudi2"
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_mla_fp16_publish_gather_gaudi2"
#define DSV41_MAIN_REUSE_GATHER_GUID "custom_deepseek_v41_mla_fp16_reuse_gather_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp"
