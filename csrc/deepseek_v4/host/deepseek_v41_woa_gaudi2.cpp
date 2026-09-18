// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_woa_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_woa_quant_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_woa_quant_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_woa_scale_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_woa_scale_gaudi2_o_end;
namespace {
void map(tpc_lib_api::TensorAccessPattern& p, int dim, int index, int a, int first, int last) {
    p.mapping[dim].indexSpaceDim = index;
    p.mapping[dim].a = a;
    p.mapping[dim].start_b = first;
    p.mapping[dim].end_b = last;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41WoaGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != (quant_ ? 1u : 3u)) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != (quant_ ? 2u : 1u)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& a = p->inputTensors[0].geometry;
    const auto& b = p->outputTensors[0].geometry;
    const auto tokens = quant_ ? a.maxSizes[2] : a.maxSizes[1];
    if (tokens < 1 || tokens > 8192 || a.dims != 3 || b.dims != 3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (quant_) {
        const auto& s = p->outputTensors[1].geometry;
        if (a.dataType != DATA_BF16 || b.dataType != DATA_F8_143 || s.dataType != DATA_F32)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (a.maxSizes[0] != 4096 || a.maxSizes[1] != 4 || b.maxSizes[0] != 4096 ||
            b.maxSizes[1] != tokens || b.maxSizes[2] != 4 || s.dims != 3 ||
            s.maxSizes[0] != 1 || s.maxSizes[1] != tokens || s.maxSizes[2] != 4)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = tokens; out->indexSpaceGeometry[1] = 4;
        auto& input = out->inputTensorAccessPattern[0];
        map(input, 0, 0, 0, 0, 4095); map(input, 1, 1, 1, 0, 0); map(input, 2, 0, 1, 0, 0);
        for (int i = 0; i < 2; ++i) {
            auto& output = out->outputTensorAccessPattern[i];
            map(output, 0, 0, 0, 0, i == 0 ? 4095 : 0);
            map(output, 1, 0, 1, 0, 0); map(output, 2, 1, 1, 0, 0);
        }
    } else {
        const auto& sw = p->inputTensors[1].geometry;
        const auto& sx = p->inputTensors[2].geometry;
        if (a.dataType != DATA_F32 || sw.dataType != DATA_F32 || sx.dataType != DATA_F32 || b.dataType != DATA_BF16)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (a.maxSizes[0] != 1024 || a.maxSizes[2] != 4 || b.maxSizes[0] != 1024 || b.maxSizes[1] != 4 ||
            b.maxSizes[2] != tokens || sw.dims != 3 || sx.dims != 3 || sw.maxSizes[0] != 1024 ||
            sw.maxSizes[1] != 1 || sw.maxSizes[2] != 4 || sx.maxSizes[0] != 1 || sx.maxSizes[1] != tokens ||
            sx.maxSizes[2] != 4) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank = 3;
        out->indexSpaceGeometry[0] = 8; out->indexSpaceGeometry[1] = tokens; out->indexSpaceGeometry[2] = 4;
        for (int i = 0; i < 3; ++i) {
            auto& input = out->inputTensorAccessPattern[i];
            map(input, 0, 0, i == 2 ? 0 : 128, 0, i == 2 ? 0 : 127);
            map(input, 1, 1, i == 1 ? 0 : 1, 0, 0); map(input, 2, 2, 1, 0, 0);
        }
        auto& output = out->outputTensorAccessPattern[0];
        map(output, 0, 0, 128, 0, 127); map(output, 1, 2, 1, 0, 0); map(output, 2, 1, 1, 0, 0);
    }
    auto* begin = quant_ ? &_binary___deepseek_v41_woa_quant_gaudi2_o_start : &_binary___deepseek_v41_woa_scale_gaudi2_o_start;
    auto* end = quant_ ? &_binary___deepseek_v41_woa_quant_gaudi2_o_end : &_binary___deepseek_v41_woa_scale_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin; out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
