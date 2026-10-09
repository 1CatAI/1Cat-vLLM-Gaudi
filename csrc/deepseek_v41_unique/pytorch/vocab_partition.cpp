// SPDX-License-Identifier: Apache-2.0
// Same exact FP32 threshold/emit code, with 8 independent partitions per row.
#define DSV41_PREFILL_TOPK_OP "custom_deepseek_v41_vocab_partition_topk_gaudi2"
#define DSV41_PREFILL_THRESHOLD_GUID "custom_deepseek_v41_vocab_radix_threshold_gaudi2"
#define DSV41_PREFILL_EMIT_GUID "custom_deepseek_v41_vocab_radix_emit_gaudi2"
#define DSV41_PREFILL_MAX_COLUMNS 4096
#define DSV41_PREFILL_MAX_ROWS 48
#define DSV41_PREFILL_MAX_WIDTH 256
#define DSV41_PREFILL_BITMAP 1
#include "../../deepseek_v4/pytorch/hpu_dsv41_prefill_topk_pt2.cpp"
