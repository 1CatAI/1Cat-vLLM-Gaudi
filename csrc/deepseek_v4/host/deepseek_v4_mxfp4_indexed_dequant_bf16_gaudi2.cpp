// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mxfp4_indexed_dequant_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_indexed_dequant_normal_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV4Mxfp4IndexedDequantBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    std::strcpy(name, normal_ ? "custom_deepseek_v4_mxfp4_indexed_dequant_normal_bf16_gaudi2"
                            : "custom_deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2");
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV4Mxfp4IndexedDequantBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 3) {
        in->inputTensorNr = 3;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType types[] = {DATA_I32, DATA_U8, DATA_U8};
    for (unsigned i = 0; i < 3; ++i) {
        if (in->inputTensors[i].geometry.dataType != types[i]) {
            in->inputTensors[i].geometry.dataType = types[i];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    auto& ids = in->inputTensors[0].geometry;
    auto& packed = in->inputTensors[1].geometry;
    auto& scale = in->inputTensors[2].geometry;
    auto& output = in->outputTensors[0].geometry;
    if (output.dataType != DATA_BF16) {
        output.dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    if (ids.dims != 2 || ids.maxSizes[0] != 6 || ids.maxSizes[1] != 1 ||
        packed.dims != 3 || scale.dims != 3 || packed.maxSizes[0] == 0 ||
        packed.maxSizes[0] % 256 || packed.maxSizes[1] == 0 ||
        packed.maxSizes[2] == 0 || packed.maxSizes[0] != scale.maxSizes[0] * 16 ||
        packed.maxSizes[1] != scale.maxSizes[1] || packed.maxSizes[2] != scale.maxSizes[2]) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (output.dims != 3 || output.maxSizes[0] != packed.maxSizes[0] * 2 ||
        output.maxSizes[1] != packed.maxSizes[1] || output.maxSizes[2] != 6) {
        output.dims = 3;
        output.maxSizes[0] = packed.maxSizes[0] * 2;
        output.maxSizes[1] = packed.maxSizes[1];
        output.maxSizes[2] = 6;
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 3;
    out->indexSpaceGeometry[0] = packed.maxSizes[0] / 256;
    out->indexSpaceGeometry[1] = packed.maxSizes[1];
    out->indexSpaceGeometry[2] = 6;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned index, int a, int b, int end) {
        p.mapping[dim].indexSpaceDim = index;
        p.mapping[dim].a = a;
        p.mapping[dim].start_b = b;
        p.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 2, 1, 0, 0);
    map(out->inputTensorAccessPattern[0], 1, 2, 0, 0, 0);
    for (unsigned i = 1; i <= 2; ++i) {
        auto& p = out->inputTensorAccessPattern[i];
        map(p, 0, 0, i == 1 ? 256 : 16, 0, 255);
        map(p, 1, 1, 1, 0, 0);
        // Runtime expert IDs are not affine in the slot. Declare the
        // conservative source range; only the indexed expert is loaded.
        map(p, 2, 2, 0, 0, packed.maxSizes[2] - 1);
    }
    auto& p = out->outputTensorAccessPattern[0];
    map(p, 0, 0, 512, 0, 511);
    map(p, 1, 1, 1, 0, 0);
    map(p, 2, 2, 1, 0, 0);
    out->kernel.paramsNr = 0;
    const auto* start = normal_ ? &_binary___deepseek_v4_mxfp4_indexed_dequant_normal_bf16_gaudi2_o_start
                               : &_binary___deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2_o_start;
    const auto* end = normal_ ? &_binary___deepseek_v4_mxfp4_indexed_dequant_normal_bf16_gaudi2_o_end
                             : &_binary___deepseek_v4_mxfp4_indexed_dequant_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
