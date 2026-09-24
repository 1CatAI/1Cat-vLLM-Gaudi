// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_topk_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_prefill_topk_threshold_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_topk_threshold_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_topk_emit_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_topk_emit_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41PrefillTopkGaudi2::GetGcDefinitions(HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out || !p->nodeParams.nodeParams || p->nodeParams.nodeParamsSize != 2 * sizeof(int))
        return GLUE_FAILED;
    const auto* params = static_cast<const int*>(p->nodeParams.nodeParams);
    const int columns = params[0], width = params[1];
    if (columns < 64 || columns > 4096 || columns % 64 || width < 1 || width > columns || width > 2048)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (p->inputTensorNr != (emit_ ? 2u : 1u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != (emit_ ? 2u : 1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& score = p->inputTensors[0].geometry;
    const unsigned batch = score.maxSizes[1];
    auto shape = [batch](const TensorGeometry& g, unsigned n, TensorDataType dtype) {
        return g.dims == 2 && g.maxSizes[0] == n && g.maxSizes[1] == batch && g.dataType == dtype;
    };
    if (!shape(score, columns, DATA_F32) || batch < 1 || batch > 8192)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (emit_) {
        if (!shape(p->inputTensors[1].geometry, 18, DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(p->outputTensors[0].geometry, width, DATA_F32) ||
            !shape(p->outputTensors[1].geometry, width, DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    } else if (!shape(p->outputTensors[0].geometry, 18, DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = emit_ ? 8 : 1;
    out->indexSpaceGeometry[1] = batch;
    // Each request is independent. Emission partitions have dynamic output
    // offsets, bounded by the entire output row; prefix counts make writes disjoint.
    for (unsigned i = 0; i < p->inputTensorNr; ++i) {
        auto& a = out->inputTensorAccessPattern[i];
        a.allRequired = false;
        a.mapping[0] = {0, 0, 0, static_cast<float>(i ? 17 : columns - 1)};
        a.mapping[1] = {1, 1, 0, 0};
    }
    for (unsigned i = 0; i < p->outputTensorNr; ++i) {
        auto& a = out->outputTensorAccessPattern[i];
        a.allRequired = false;
        a.mapping[0] = {0, 0, 0, static_cast<float>(emit_ ? width - 1 : 17)};
        a.mapping[1] = {1, 1, 0, 0};
    }
    out->kernel.paramsNr = 2;
    std::memcpy(out->kernel.scalarParams, params, 2 * sizeof(int));
    auto* start = emit_ ? &_binary___deepseek_v41_prefill_topk_emit_gaudi2_o_start
                        : &_binary___deepseek_v41_prefill_topk_threshold_gaudi2_o_start;
    auto* end = emit_ ? &_binary___deepseek_v41_prefill_topk_emit_gaudi2_o_end
                      : &_binary___deepseek_v41_prefill_topk_threshold_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
