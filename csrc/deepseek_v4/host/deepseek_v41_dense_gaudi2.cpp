// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_dense_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_dense_quant_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_dense_quant_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_dense_scale_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_dense_scale_gaudi2_o_end;
namespace {
void map(tpc_lib_api::TensorAccessPattern& p, int dim, int index, int a, int first, int last) {
    p.mapping[dim].indexSpaceDim = index; p.mapping[dim].a = a;
    p.mapping[dim].start_b = first; p.mapping[dim].end_b = last;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41DenseGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != (quant_ ? 1u : 3u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != (quant_ ? 2u : 1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& a = p->inputTensors[0].geometry;
    const auto& b = p->outputTensors[0].geometry;
    const auto width = a.maxSizes[0], rows = a.maxSizes[1];
    if (rows < 1 || rows > 8192 || a.dims != 2 || b.dims != 2 ||
        b.maxSizes[0] != width || b.maxSizes[1] != rows) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (quant_) {
        const auto& s = p->outputTensors[1].geometry;
        if (a.dataType != DATA_BF16 || b.dataType != DATA_F8_143 || s.dataType != DATA_F32)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if ((width != 1280 && width != 4096) || s.dims != 2 || s.maxSizes[0] != 1 || s.maxSizes[1] != rows)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = rows;
        map(out->inputTensorAccessPattern[0], 0, 0, 0, 0, width - 1);
        map(out->inputTensorAccessPattern[0], 1, 0, 1, 0, 0);
        for (int i = 0; i < 2; ++i) {
            map(out->outputTensorAccessPattern[i], 0, 0, 0, 0, i == 0 ? width - 1 : 0);
            map(out->outputTensorAccessPattern[i], 1, 0, 1, 0, 0);
        }
    } else {
        const auto& sw = p->inputTensors[1].geometry;
        const auto& sx = p->inputTensors[2].geometry;
        if (a.dataType != DATA_F32 || sw.dataType != DATA_F32 || sx.dataType != DATA_F32 || b.dataType != DATA_BF16)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if ((width != 5120 && width != 16384) || sw.dims != 2 || sw.maxSizes[0] != width || sw.maxSizes[1] != 1 ||
            sx.dims != 2 || sx.maxSizes[0] != 1 || sx.maxSizes[1] != rows) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = width / 128; out->indexSpaceGeometry[1] = rows;
        for (int i = 0; i < 3; ++i) {
            map(out->inputTensorAccessPattern[i], 0, 0, i == 2 ? 0 : 128, 0, i == 2 ? 0 : 127);
            map(out->inputTensorAccessPattern[i], 1, 1, i == 1 ? 0 : 1, 0, 0);
        }
        map(out->outputTensorAccessPattern[0], 0, 0, 128, 0, 127);
        map(out->outputTensorAccessPattern[0], 1, 1, 1, 0, 0);
    }
    auto* begin = quant_ ? &_binary___deepseek_v41_dense_quant_gaudi2_o_start : &_binary___deepseek_v41_dense_scale_gaudi2_o_start;
    auto* end = quant_ ? &_binary___deepseek_v41_dense_quant_gaudi2_o_end : &_binary___deepseek_v41_dense_scale_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin; out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
