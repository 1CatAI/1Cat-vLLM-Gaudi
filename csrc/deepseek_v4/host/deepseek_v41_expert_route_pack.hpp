// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"
#include <cstring>
namespace dsv41_route_pack {
inline void map(tpc_lib_api::TensorAccessPattern& p, unsigned dim, unsigned axis,
                int coefficient, int first, int last) {
    p.mapping[dim].indexSpaceDim = axis;
    p.mapping[dim].a = coefficient;
    p.mapping[dim].start_b = first;
    p.mapping[dim].end_b = last;
}
inline tpc_lib_api::GlueCodeReturn instantiate(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
    unsigned route_pack, const unsigned char* start, const unsigned char* end,
    unsigned n_blocks_per_point=1) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const TensorDataType types[] = {DATA_I32, DATA_I16, DATA_I16, DATA_BF16};
    for (unsigned i = 0; i < 4; ++i) {
        if (in->inputTensors[i].geometry.dataType != types[i]) {
            in->inputTensors[i].geometry.dataType = types[i];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    const auto& ids = in->inputTensors[0].geometry;
    const auto& q = in->inputTensors[1].geometry;
    const auto& s = in->inputTensors[2].geometry;
    const auto& lut = in->inputTensors[3].geometry;
    if (ids.dims != 2 || ids.maxSizes[1] != 1 || !ids.maxSizes[0] || ids.maxSizes[0] % 6 ||
        ids.maxSizes[0] > 192 || q.dims != 3 || !q.maxSizes[0] || q.maxSizes[0] % 8192 ||
        q.maxSizes[0] > 327680 || !q.maxSizes[1] || q.maxSizes[1] > 20 ||
        !q.maxSizes[2] || q.maxSizes[2] > 384 || s.dims != 3 ||
        (s.maxSizes[0] * 8 != q.maxSizes[0] && s.maxSizes[0] != q.maxSizes[0] / 16 + 128) || s.maxSizes[1] != q.maxSizes[1] ||
        s.maxSizes[2] != q.maxSizes[2] || lut.dims != 1 || lut.maxSizes[0] != 128)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const bool compact = s.maxSizes[0] == q.maxSizes[0] / 16 + 128;
    const auto blocks = q.maxSizes[1], k = q.maxSizes[0] / 64, batches = ids.maxSizes[0] / route_pack;
    auto& result = in->outputTensors[0].geometry;
    if (result.dataType != DATA_F8_143) {
        result.dataType = DATA_F8_143; return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (result.dims != 3 || result.maxSizes[0] != route_pack * blocks * 256 ||
        result.maxSizes[1] != k || result.maxSizes[2] != batches) {
        result.dims = 3; result.maxSizes[0] = route_pack * blocks * 256;
        result.maxSizes[1] = k; result.maxSizes[2] = batches;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 3;
    if(!n_blocks_per_point || (route_pack*blocks)%n_blocks_per_point)return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceGeometry[0] = route_pack * blocks/n_blocks_per_point;
    out->indexSpaceGeometry[1] = batches;
    out->indexSpaceGeometry[2] = k / 128;
    out->inputTensorAccessPattern[0].allRequired = true;
    for (unsigned i = 1; i < 3; ++i) {
        const int words = i == 1 ? 8192 : compact ? 512 : 1024;
        map(out->inputTensorAccessPattern[i], 0, 2, words, 0, words - 1);
        if (i == 2 && compact)
            map(out->inputTensorAccessPattern[i], 0, 2, 0, 0, s.maxSizes[0] - 1);
        map(out->inputTensorAccessPattern[i], 1, 0, 0, 0, blocks - 1);
        map(out->inputTensorAccessPattern[i], 2, 1, 0, 0, q.maxSizes[2] - 1);
    }
    out->inputTensorAccessPattern[3].allRequired = true;
    map(out->outputTensorAccessPattern[0], 0, 0, 256*n_blocks_per_point, 0, 256*n_blocks_per_point-1);
    map(out->outputTensorAccessPattern[0], 1, 2, 128, 0, 127);
    map(out->outputTensorAccessPattern[0], 2, 1, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}

} // namespace dsv41_route_pack
