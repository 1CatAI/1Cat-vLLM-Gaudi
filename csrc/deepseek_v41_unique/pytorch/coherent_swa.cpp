// SPDX-License-Identifier: Apache-2.0
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_coherent_swa_publish_mla_gaudi2"
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_coherent_swa_mla_gaudi2"
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_main_batch_publish_gather_gaudi2"
#define DSV41_MAIN_SPLIT_REUSE 1
#define DSV41_COHERENT_SWA 1
#define DSV41_COHERENT_SWA_BF16_PV 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp"
