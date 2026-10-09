// SPDX-License-Identifier: Apache-2.0
// Additive capability only. Public KV planes and C1 registrations are unchanged.
#define DSV41_MAIN_FP16_PV 1
#define DSV41_MAIN_SPLIT_REUSE 1
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_main_fp16_publish_mla_gaudi2"
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_main_fp16_reuse_mla_gaudi2"
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_main_batch_publish_gather_gaudi2"
#define DSV41_MAIN_REUSE_GATHER_GUID "custom_deepseek_v41_swa_only_reuse_gather_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp"
