// SPDX-License-Identifier: Apache-2.0
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_swa_cached_publish_mla_gaudi2"
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_swa_cached_reuse_mla_gaudi2"
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_swa_cached_publish_gaudi2"
#define DSV41_MAIN_REUSE_GATHER_GUID "custom_deepseek_v41_swa_cached_reuse_gaudi2"
#define DSV41_MAIN_SPLIT_REUSE 1
#define DSV41_MAIN_SWA_CACHED 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp"
