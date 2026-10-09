// SPDX-License-Identifier: Apache-2.0
// C1 production operator already supports C1-C6 and rank-major peer operands.
#define DSV41_WEIGHTED_STATS_PREPARED 1
#define DSV41_MHC_STATS_SCHEMA "custom_deepseek_v41_dspark_mhc_post_norm_statistics_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_mhc_weighted_stats_pt2.cpp"
