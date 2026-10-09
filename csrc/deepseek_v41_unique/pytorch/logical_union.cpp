// SPDX-License-Identifier: Apache-2.0
// Reuse the common MLA contract and its unchanged QK/softmax/PV pipeline.
#define DSV41_LOGICAL_MLA_SCHEMA "custom_deepseek_v41_logical_mla_union_gaudi2"
#define DSV41_LOGICAL_MLA_UNION 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_logical_mla_pt2.cpp"
