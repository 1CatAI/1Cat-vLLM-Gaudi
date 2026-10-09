// SPDX-License-Identifier: Apache-2.0
// Same compound QK/softmax/PV; only the gather's batch-slicing contract changes.
#define DSV41_LOGICAL_MLA_SCHEMA "custom_deepseek_v41_logical_mla_cohesive_gaudi2"
#define DSV41_LOGICAL_MLA_VECTOR_GUID "custom_deepseek_v41_logical_mla_cohesive_vector_gaudi2"
#include "../../deepseek_v4/pytorch/hpu_dsv41_logical_mla_pt2.cpp"
