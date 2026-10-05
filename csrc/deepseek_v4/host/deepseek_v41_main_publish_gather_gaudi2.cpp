// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_main_publish_gather_gaudi2.hpp"
#include <cstring>
#include <initializer_list>
extern unsigned char _binary___deepseek_v41_main_publish_gather_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_main_publish_gather_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_main_publish_vector_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_main_publish_vector_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_main_publish_vector_mask_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_main_publish_vector_mask_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_main_publish_native_codec_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_main_publish_native_codec_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_main_publish_tensor_mask_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_main_publish_tensor_mask_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41MainPublishGatherGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 6) { in->inputTensorNr = 6; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 4) { in->outputTensorNr = 4; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(int))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int ratio = *static_cast<const int*>(in->nodeParams.nodeParams);
    if (ratio != 1 && ratio != 2) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto tokens = in->inputTensors[2].geometry.maxSizes[1];
    if (tokens != 1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto matches = [](const Tensor& t, unsigned type, unsigned dims, uint64_t d0) {
        return t.geometry.dataType == type && t.geometry.dims == dims && t.geometry.maxSizes[0] == d0;
    };
    if (!matches(in->inputTensors[0], DATA_U8, 2, 528) || in->inputTensors[0].geometry.maxSizes[1] != 256 ||
        !matches(in->inputTensors[1], DATA_U8, 2, 288) || !in->inputTensors[1].geometry.maxSizes[1] ||
        in->inputTensors[1].geometry.maxSizes[1] > 0x7ffffdffULL ||
        !matches(in->inputTensors[2], DATA_I32, 2, 512) ||
        !matches(in->inputTensors[3], DATA_I32, 1, tokens) ||
        !matches(in->inputTensors[4], DATA_I32, 1, in->inputTensors[4].geometry.maxSizes[0]) ||
        !in->inputTensors[4].geometry.maxSizes[0] || in->inputTensors[4].geometry.maxSizes[0] > 8192 ||
        !matches(in->inputTensors[5], DATA_I32, 1, tokens)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    auto map = [](TensorAccessPattern& p, unsigned dim, unsigned axis, int a, int last) {
        p.mapping[dim] = {axis, float(a), 0, float(last)};
    };
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 640;
    out->indexSpaceGeometry[1] = tokens;
    for (unsigned i : {0u, 1u, 4u}) {
        auto& access = out->inputTensorAccessPattern[i];
        std::memset(&access, 0, sizeof(access));
        access.sparseAccess = true;
        for (unsigned dim = 0; dim < in->inputTensors[i].geometry.dims; ++dim)
            map(access, dim, 0, 0, in->inputTensors[i].geometry.maxSizes[dim] - 1);
    }
    // The first 128 output slots use SWA; the remaining slots read selection.
    map(out->inputTensorAccessPattern[2], 0, 0, 0, 511);
    map(out->inputTensorAccessPattern[2], 1, 1, 1, 0);
    map(out->inputTensorAccessPattern[3], 0, 1, 1, 0);
    map(out->inputTensorAccessPattern[5], 0, 1, 1, 0);
    for (unsigned i : {0u, 1u, 3u}) {
        const auto& g = in->outputTensors[i].geometry;
        if (!matches(in->outputTensors[i], i == 1 ? DATA_F32 : DATA_BF16, 3, 512) ||
            g.maxSizes[1] != 640 || g.maxSizes[2] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        map(out->outputTensorAccessPattern[i], 0, 0, 0, 511);
        map(out->outputTensorAccessPattern[i], 1, 0, 1, 0);
        map(out->outputTensorAccessPattern[i], 2, 1, 1, 0);
    }
    if (!matches(in->outputTensors[2], DATA_F32, 2, 640) ||
        in->outputTensors[2].geometry.maxSizes[1] != tokens) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    map(out->outputTensorAccessPattern[2], 0, 0, 1, 0);
    map(out->outputTensorAccessPattern[2], 1, 1, 1, 0);
    if (vector_mask_) out->outputTensorAccessPattern[2].allRequired = true;
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &ratio, sizeof(ratio));
    auto* first = &_binary___deepseek_v41_main_publish_gather_gaudi2_o_start;
    auto* last = &_binary___deepseek_v41_main_publish_gather_gaudi2_o_end;
    if (vector_) {
        first = &_binary___deepseek_v41_main_publish_vector_gaudi2_o_start;
        last = &_binary___deepseek_v41_main_publish_vector_gaudi2_o_end;
    }
    if (vector_mask_) {
        first = &_binary___deepseek_v41_main_publish_vector_mask_gaudi2_o_start;
        last = &_binary___deepseek_v41_main_publish_vector_mask_gaudi2_o_end;
    }
    if (native_codec_) {
        first = &_binary___deepseek_v41_main_publish_native_codec_gaudi2_o_start;
        last = &_binary___deepseek_v41_main_publish_native_codec_gaudi2_o_end;
    }
    if (tensor_mask_) {
        first = &_binary___deepseek_v41_main_publish_tensor_mask_gaudi2_o_start;
        last = &_binary___deepseek_v41_main_publish_tensor_mask_gaudi2_o_end;
    }
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = last - first;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, first, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
