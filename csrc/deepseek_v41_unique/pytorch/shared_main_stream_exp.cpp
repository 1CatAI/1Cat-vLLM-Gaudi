// SPDX-License-Identifier: Apache-2.0
#define DSV41_MAIN_PUBLISH_SCHEMA "custom_deepseek_v41_main_stream_exp_publish_mla_gaudi2"
#define DSV41_MAIN_REUSE_SCHEMA "custom_deepseek_v41_main_stream_exp_reuse_mla_gaudi2"
#define DSV41_MAIN_PUBLISH_GATHER_GUID "custom_deepseek_v41_main_single_bank_gather_gaudi2"
#define DSV41_MAIN_REUSE_GATHER_GUID "custom_deepseek_v41_swa_keys_reuse_gaudi2"
#define DSV41_MAIN_SPLIT_REUSE 1
#define DSV41_MAIN_SINGLE_BANK 1
#define DSV41_MAIN_STREAM_EXP_PV 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_shared_main_mla_pt2.cpp"
