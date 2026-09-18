// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_q_scale_rope_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_q_scale_rope_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_q_scale_rope_gaudi2_o_end;
namespace {
void map(tpc_lib_api::TensorAccessPattern& p, int dim, int stride, int last) {
    p.mapping[dim].indexSpaceDim = 0; p.mapping[dim].a = stride;
    p.mapping[dim].start_b = 0; p.mapping[dim].end_b = last;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41QScaleRopeGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* p, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (!p || !out) return GLUE_FAILED;
    if (p->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& y = p->outputTensors[0].geometry;
    if (y.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (y.dims != 2 || y.maxSizes[0] != 16384 || y.maxSizes[1] != 1) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    for (int i = 0; i < 5; ++i) {
        const auto& x = p->inputTensors[i].geometry;
        if (x.dataType != (i == 3 ? DATA_I32 : DATA_F32)) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (x.dims != (i == 3 ? 1u : 2u) ||
            x.maxSizes[0] != (i < 2 ? 16384u : i == 4 ? 64u : 1u) ||
            (i == 4 ? (!x.maxSizes[1] || x.maxSizes[1] > 1048576u) :
             (i != 3 && x.maxSizes[1] != 1u))) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (i == 4) {
            // The selected row depends on the runtime position scalar and
            // cannot be expressed as an affine index-space mapping.  The TPC
            // kernel still issues only one 64-value row load.
            out->inputTensorAccessPattern[i].allRequired = true;
            out->inputTensorAccessPattern[i].sparseAccess = true;
            continue;
        }
        map(out->inputTensorAccessPattern[i], 0, i < 2 ? 512 : 0, i < 2 ? 511 : i == 4 ? 63 : 0);
        if (i != 3) map(out->inputTensorAccessPattern[i], 1, 0, 0);
    }
    out->indexSpaceRank = 1; out->indexSpaceGeometry[0] = 32;
    map(out->outputTensorAccessPattern[0], 0, 512, 511);
    map(out->outputTensorAccessPattern[0], 1, 0, 0);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = &_binary___deepseek_v41_q_scale_rope_gaudi2_o_end -
                          &_binary___deepseek_v41_q_scale_rope_gaudi2_o_start;
    out->kernel.paramsNr = 0;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, &_binary___deepseek_v41_q_scale_rope_gaudi2_o_start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
