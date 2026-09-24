// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_flash_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_prefill_flash_kv_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_flash_kv_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_flash_mask_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_flash_mask_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillFlashGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (p->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const unsigned ids_index = mask_ ? 0 : 1;
    const auto& ids = p->inputTensors[ids_index].geometry;
    const auto& lengths = p->inputTensors[ids_index + 1].geometry;
    const auto& y = p->outputTensors[0].geometry;
    if (ids.dims != 2 || ids.dataType != DATA_I32 || ids.maxSizes[0] < 1 || ids.maxSizes[0] > 640 ||
        ids.maxSizes[1] < 1 || ids.maxSizes[1] > 512 || lengths.dims != 1 ||
        lengths.dataType != DATA_I32 || lengths.maxSizes[0] != ids.maxSizes[1])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned columns = ids.maxSizes[0], tokens = ids.maxSizes[1];
    if (mask_) {
        const auto& sink = p->inputTensors[2].geometry;
        if (sink.dims != 1 || sink.dataType != DATA_F32 || sink.maxSizes[0] < 1 || sink.maxSizes[0] > 64 ||
            !p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != sizeof(int))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const int rows = *static_cast<const int*>(p->nodeParams.nodeParams);
        if (rows < 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.dims != 3 || y.dataType != DATA_F32 || y.maxSizes[0] != columns + 1 ||
            y.maxSizes[1] != sink.maxSizes[0] || y.maxSizes[2] != tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 3;
        out->indexSpaceGeometry[0] = (columns + 64) / 64;
        out->indexSpaceGeometry[1] = sink.maxSizes[0];
        out->indexSpaceGeometry[2] = tokens;
        // The extra output column is a sink, with no corresponding ID read.
        // A linear map would put its last work point beyond the input extent.
        out->inputTensorAccessPattern[0].allRequired = true;
        out->inputTensorAccessPattern[1].mapping[0] = {2, 1, 0, 0};
        out->inputTensorAccessPattern[2].mapping[0] = {1, 1, 0, 0};
        out->outputTensorAccessPattern[0].mapping[0] = {0, 64, 0, 63};
        out->outputTensorAccessPattern[0].mapping[1] = {1, 1, 0, 0};
        out->outputTensorAccessPattern[0].mapping[2] = {2, 1, 0, 0};
        out->kernel.paramsNr = 1;
        out->kernel.scalarParams[0] = rows;
    } else {
        const auto& cache = p->inputTensors[0].geometry;
        if (cache.dims != 2 || cache.dataType != DATA_BF16 || cache.maxSizes[0] != 512 || cache.maxSizes[1] < 1)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (y.dims != 3 || y.dataType != DATA_BF16 || y.maxSizes[0] != 512 ||
            y.maxSizes[1] != columns + 1 || y.maxSizes[2] != tokens)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Group 16 adjacent selected rows per work point to amortize sparse
        // address and scheduling costs. Keep a precise output access map so
        // the selected KV can be streamed to its real MME consumer. The final
        // work point also writes the sink row, which has no input ID.
        constexpr unsigned rows_per_work_point = 16;
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = (columns + 1 + rows_per_work_point - 1) / rows_per_work_point;
        out->indexSpaceGeometry[1] = tokens;
        out->inputTensorAccessPattern[0].allRequired = true;
        out->inputTensorAccessPattern[0].sparseAccess = true;
        out->inputTensorAccessPattern[1].allRequired = true;
        out->inputTensorAccessPattern[2].mapping[0] = {1, 1, 0, 0};
        out->outputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 511};
        out->outputTensorAccessPattern[0].mapping[1] = {0, rows_per_work_point, 0, rows_per_work_point - 1};
        out->outputTensorAccessPattern[0].mapping[2] = {1, 1, 0, 0};
        out->kernel.paramsNr = 0;
    }
    const auto* begin = mask_ ? &_binary___deepseek_v41_prefill_flash_mask_gaudi2_o_start
                              : &_binary___deepseek_v41_prefill_flash_kv_gaudi2_o_start;
    const auto* end = mask_ ? &_binary___deepseek_v41_prefill_flash_mask_gaudi2_o_end
                            : &_binary___deepseek_v41_prefill_flash_kv_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
