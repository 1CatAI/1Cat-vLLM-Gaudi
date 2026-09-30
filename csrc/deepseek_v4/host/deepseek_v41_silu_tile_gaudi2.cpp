// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_silu_tile_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_silu_activate_tile_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_silu_activate_tile_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_silu_quant_tile_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_silu_quant_tile_gaudi2_o_end;
namespace {
using namespace tpc_lib_api;
void map(TensorAccessPattern& p, unsigned dim, unsigned axis, float a, float first, float last) {
    p.mapping[dim] = {axis, a, first, last};
}
GlueCodeReturn binary(HabanaKernelInstantiation* out, unsigned char* begin, unsigned char* end) {
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    out->kernel.paramsNr = 0;
    return GLUE_SUCCESS;
}
bool shape(const TensorGeometry& g, TensorDataType type, uint64_t x, uint64_t y, uint64_t z) {
    return g.dataType == type && g.dims == 3 && g.maxSizes[0] == x && g.maxSizes[1] == y && g.maxSizes[2] == z;
}
}
tpc_lib_api::GlueCodeReturn DeepseekV41SiluActivateTileGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& product = in->inputTensors[0].geometry;
    const auto& ids = in->inputTensors[1].geometry;
    const auto& sx = in->inputTensors[2].geometry;
    const auto& channel = in->inputTensors[3].geometry;
    const auto& router = in->inputTensors[4].geometry;
    constexpr uint64_t width = 640, tiles = 5;
    const auto rows = product.maxSizes[2];
    if (!rows || rows > 36 || !shape(product, DATA_F32, width * 2, 1, rows) ||
        ids.dataType != DATA_I32 || ids.dims != 2 || ids.maxSizes[0] != rows || ids.maxSizes[1] != 1 ||
        sx.dataType != DATA_F32 || sx.dims != 2 || sx.maxSizes[0] != 1 || (sx.maxSizes[1] != 1 && sx.maxSizes[1] != rows) ||
        channel.dataType != DATA_BF16 || channel.dims != 3 || channel.maxSizes[0] != 256 ||
        channel.maxSizes[1] != 5 || !channel.maxSizes[2] || channel.maxSizes[2] > 384 ||
        router.dataType != DATA_F32 || router.dims != 2 || router.maxSizes[0] != rows || router.maxSizes[1] != 1)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!shape(in->outputTensors[0].geometry, DATA_BF16, width, 1, rows) ||
        !shape(in->outputTensors[1].geometry, DATA_F32, tiles, 1, rows)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = tiles; out->indexSpaceGeometry[1] = rows;
    map(out->inputTensorAccessPattern[0], 0, 0, 128, 0, width + 127);
    map(out->inputTensorAccessPattern[0], 1, 0, 0, 0, 0);
    map(out->inputTensorAccessPattern[0], 2, 1, 1, 0, 0);
    for (unsigned i : {1u, 4u}) {
        map(out->inputTensorAccessPattern[i], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[i], 1, 0, 0, 0, 0);
    }
    map(out->inputTensorAccessPattern[2], 0, 0, 0, 0, 0);
    map(out->inputTensorAccessPattern[2], 1, 1, sx.maxSizes[1] == 1 ? 0 : 1, 0, 0);
    out->inputTensorAccessPattern[3].allRequired = true;
    for (unsigned i = 0; i < 2; ++i) {
        map(out->outputTensorAccessPattern[i], 0, 0, i == 0 ? 128 : 1, 0, i == 0 ? 127 : 0);
        map(out->outputTensorAccessPattern[i], 1, 0, 0, 0, 0);
        map(out->outputTensorAccessPattern[i], 2, 1, 1, 0, 0);
    }
    return binary(out, &_binary___deepseek_v41_silu_activate_tile_gaudi2_o_start,
                       &_binary___deepseek_v41_silu_activate_tile_gaudi2_o_end);
}
tpc_lib_api::GlueCodeReturn DeepseekV41SiluQuantTileGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto rows = in->inputTensors[0].geometry.maxSizes[2];
    if (!rows || rows > 36 || !shape(in->inputTensors[0].geometry, DATA_BF16, 640, 1, rows) ||
        !shape(in->inputTensors[1].geometry, DATA_F32, 5, 1, rows)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!shape(in->outputTensors[0].geometry, DATA_F8_143, 640, 1, rows) ||
        !shape(in->outputTensors[1].geometry, DATA_F32, 1, 1, rows)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2; out->indexSpaceGeometry[0] = 5; out->indexSpaceGeometry[1] = rows;
    for (unsigned i = 0; i < 2; ++i) {
        map(out->inputTensorAccessPattern[i], 0, 0, i == 0 ? 128 : 0, 0, i == 0 ? 127 : 4);
        map(out->inputTensorAccessPattern[i], 1, 0, 0, 0, 0);
        map(out->inputTensorAccessPattern[i], 2, 1, 1, 0, 0);
        map(out->outputTensorAccessPattern[i], 0, 0, i == 0 ? 128 : 0, 0, i == 0 ? 127 : 0);
        map(out->outputTensorAccessPattern[i], 1, 0, 0, 0, 0);
        map(out->outputTensorAccessPattern[i], 2, 1, 1, 0, 0);
    }
    return binary(out, &_binary___deepseek_v41_silu_quant_tile_gaudi2_o_start,
                       &_binary___deepseek_v41_silu_quant_tile_gaudi2_o_end);
}
