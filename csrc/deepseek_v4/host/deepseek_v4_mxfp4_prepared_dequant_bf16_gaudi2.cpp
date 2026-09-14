// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_end;

extern unsigned char _binary___deepseek_v41_mxfp4_n512_dequant_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_n512_dequant_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2_o_end;

extern unsigned char _binary___deepseek_v41_mxfp4_k128_dequant_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_k128_dequant_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_k128_dequant_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_k128_dequant_normal_bf16_gaudi2_o_end;

extern unsigned char _binary___deepseek_v41_mxfp4_shared_dequant_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_shared_dequant_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_shared_dequant_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_shared_dequant_normal_bf16_gaudi2_o_end;

extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_normal_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_gate_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_gate_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_gate_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_gate_normal_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_up_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_up_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_up_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v4_mxfp4_prepared_dequant_up_normal_bf16_gaudi2_o_end;

tpc_lib_api::GlueCodeReturn DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::GetKernelName(
    char name[tpc_lib_api::MAX_NODE_NAME]) {
    const char* kernel = nullptr;
    if (legacy_ == 2) {
        kernel = "custom_deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2";
    } else if (legacy_ == 1) {
        kernel = normal_ ? "custom_deepseek_v41_mxfp4_prepared_dequant_k128n_bf16_gaudi2"
                         : "custom_deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2";
    } else if (window_) {
        kernel = normal_ ? "custom_deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2"
                         : "custom_deepseek_v41_mxfp4_n512_dequant_bf16_gaudi2";
    } else if (tiled_) {
        kernel = normal_ ? "custom_deepseek_v41_mxfp4_k128_dequant_normal_bf16_gaudi2"
                         : "custom_deepseek_v41_mxfp4_k128_dequant_bf16_gaudi2";
    } else if (shared_) {
        kernel = normal_ ? "custom_deepseek_v41_mxfp4_shared_dequant_normal_bf16_gaudi2"
                         : "custom_deepseek_v41_mxfp4_shared_dequant_bf16_gaudi2";
    } else if (v41_) {
        kernel = normal_ ? "custom_deepseek_v41_mxfp4_prepared_dequant_normal_bf16_gaudi2"
                         : "custom_deepseek_v41_mxfp4_prepared_dequant_bf16_gaudi2";
    } else if (half_ == 0) {
        kernel = normal_ ? "custom_deepseek_v4_mxfp4_prepared_gate_normal_bf16_gaudi2"
                         : "custom_deepseek_v4_mxfp4_prepared_gate_bf16_gaudi2";
    } else if (half_ == 1) {
        kernel = normal_ ? "custom_deepseek_v4_mxfp4_prepared_up_normal_bf16_gaudi2"
                         : "custom_deepseek_v4_mxfp4_prepared_up_bf16_gaudi2";
    } else {
        kernel = normal_ ? "custom_deepseek_v4_mxfp4_prepared_dequant_normal_bf16_gaudi2"
                         : "custom_deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2";
    }
    std::strcpy(name, kernel);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV4Mxfp4PreparedDequantBF16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (tiled_ && (!v41_ || shared_ || half_ >= 0)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (legacy_ == 2 && !normal_) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (window_ && !tiled_) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const unsigned inputs = shared_ ? 5 : 4;
    if (in->inputTensorNr != inputs) {
        in->inputTensorNr = inputs;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType types[] = {DATA_I32, DATA_I16, DATA_BF16, DATA_BF16, DATA_I32};
    for (unsigned index = 0; index < inputs; ++index) {
        if (in->inputTensors[index].geometry.dataType != types[index]) {
            in->inputTensors[index].geometry.dataType = types[index];
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    }
    auto& ids = in->inputTensors[0].geometry;
    auto& q16 = in->inputTensors[1].geometry;
    auto& s16 = in->inputTensors[2].geometry;
    auto& lookup = in->inputTensors[3].geometry;
    auto& output = in->outputTensors[0].geometry;
    if (shared_) {
        const auto& active = in->inputTensors[4].geometry;
        if (!v41_ || active.dims != ids.dims || active.maxSizes[0] != ids.maxSizes[0] ||
            active.maxSizes[1] != ids.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (output.dataType != DATA_BF16) {
        output.dataType = DATA_BF16;
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    }
    const bool validSlots = v41_ ? ids.maxSizes[0] > 0 && ids.maxSizes[0] <= 3072 : ids.maxSizes[0] == 6;
    if (ids.dims != 2 || !validSlots || ids.maxSizes[1] != 1 ||
        q16.dims != 3 || s16.dims != 3 || q16.maxSizes[0] == 0 ||
        q16.maxSizes[0] % (v41_ ? 4096 : 16384) || q16.maxSizes[1] == 0 ||
        q16.maxSizes[1] > (v41_ ? 40 : 32) ||
        q16.maxSizes[2] == 0 || q16.maxSizes[0] != s16.maxSizes[0] * 8 ||
        q16.maxSizes[1] != s16.maxSizes[1] || q16.maxSizes[2] != s16.maxSizes[2] ||
        lookup.dims != 1 || lookup.maxSizes[0] != 128) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (v41_ && (half_ >= 0 || q16.maxSizes[0] > 163840 || q16.maxSizes[2] > 384))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (half_ >= 0 && q16.maxSizes[1] != 16) {
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    unsigned nBlocks = half_ >= 0 ? 8 : q16.maxSizes[1];
    int nBlockOffset = half_ == 1 ? 8 : 0;
    if (window_) {
        if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != 2 * sizeof(int32_t))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        int32_t window[2];
        std::memcpy(window, in->nodeParams.nodeParams, sizeof(window));
        if (window[0] < 0 || window[1] < 1 || window[1] > 4 ||
            static_cast<uint64_t>(window[0] + window[1]) > q16.maxSizes[1])
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        nBlockOffset = window[0];
        nBlocks = window[1];
    }
    if (output.dims != 3 || output.maxSizes[0] != nBlocks * 128 ||
        output.maxSizes[1] != q16.maxSizes[0] / 32 || output.maxSizes[2] != ids.maxSizes[0]) {
        output.dims = 3;
        output.maxSizes[0] = nBlocks * 128;
        output.maxSizes[1] = q16.maxSizes[0] / 32;
        output.maxSizes[2] = ids.maxSizes[0];
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = tiled_ ? 3 : 2;
    out->indexSpaceGeometry[0] = nBlocks;
    out->indexSpaceGeometry[1] = ids.maxSizes[0];
    if (tiled_) out->indexSpaceGeometry[2] = output.maxSizes[1] / 128;
    auto map = [](TensorAccessPattern& pattern, unsigned dim, unsigned indexSpaceDim,
                  int coefficient, int start, int end) {
        pattern.mapping[dim].indexSpaceDim = indexSpaceDim;
        pattern.mapping[dim].a = coefficient;
        pattern.mapping[dim].start_b = start;
        pattern.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[0], 1, 1, 0, 0, 0);
    if (shared_) {
        map(out->inputTensorAccessPattern[4], 0, 1, 1, 0, 0);
        map(out->inputTensorAccessPattern[4], 1, 1, 0, 0, 0);
    }
    auto& qPattern = out->inputTensorAccessPattern[1];
    map(qPattern, 0, tiled_ ? 2 : 0, tiled_ ? 4096 : 0, 0,
        tiled_ ? 4095 : q16.maxSizes[0] - 1);
    map(qPattern, 1, 0, 1, nBlockOffset, nBlockOffset);
    map(qPattern, 2, 1, 0, 0, q16.maxSizes[2] - 1);
    auto& sPattern = out->inputTensorAccessPattern[2];
    map(sPattern, 0, tiled_ ? 2 : 0, tiled_ ? 512 : 0, 0,
        tiled_ ? 511 : s16.maxSizes[0] - 1);
    map(sPattern, 1, 0, 1, nBlockOffset, nBlockOffset);
    map(sPattern, 2, 1, 0, 0, q16.maxSizes[2] - 1);
    map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, 127);
    auto& outputPattern = out->outputTensorAccessPattern[0];
    map(outputPattern, 0, 0, 128, 0, 127);
    map(outputPattern, 1, tiled_ ? 2 : 0, tiled_ ? 128 : 0, 0,
        tiled_ ? 127 : q16.maxSizes[0] / 32 - 1);
    map(outputPattern, 2, 1, 1, 0, 0);
    out->kernel.paramsNr = window_ ? 1 : 0;
    if (window_) out->kernel.scalarParams[0] = nBlockOffset;
    const unsigned char* start = nullptr;
    const unsigned char* end = nullptr;
    if (legacy_ == 2) {
        start = &_binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_end;
    } else if (legacy_ == 1 && normal_) {
        start = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_end;
    } else if (legacy_ == 1) {
        start = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_end;
    } else if (window_ && normal_) {
        start = &_binary___deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_n512_dequant_normal_bf16_gaudi2_o_end;
    } else if (window_) {
        start = &_binary___deepseek_v41_mxfp4_n512_dequant_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_n512_dequant_bf16_gaudi2_o_end;
    } else if (tiled_ && normal_) {
        start = &_binary___deepseek_v41_mxfp4_k128_dequant_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_k128_dequant_normal_bf16_gaudi2_o_end;
    } else if (tiled_) {
        start = &_binary___deepseek_v41_mxfp4_k128_dequant_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_k128_dequant_bf16_gaudi2_o_end;
    } else if (shared_ && normal_) {
        start = &_binary___deepseek_v41_mxfp4_shared_dequant_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_shared_dequant_normal_bf16_gaudi2_o_end;
    } else if (shared_) {
        start = &_binary___deepseek_v41_mxfp4_shared_dequant_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_shared_dequant_bf16_gaudi2_o_end;
    } else if (half_ == 0 && normal_) {
        start = &_binary___deepseek_v4_mxfp4_prepared_dequant_gate_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v4_mxfp4_prepared_dequant_gate_normal_bf16_gaudi2_o_end;
    } else if (half_ == 0) {
        start = &_binary___deepseek_v4_mxfp4_prepared_dequant_gate_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v4_mxfp4_prepared_dequant_gate_bf16_gaudi2_o_end;
    } else if (half_ == 1 && normal_) {
        start = &_binary___deepseek_v4_mxfp4_prepared_dequant_up_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v4_mxfp4_prepared_dequant_up_normal_bf16_gaudi2_o_end;
    } else if (half_ == 1) {
        start = &_binary___deepseek_v4_mxfp4_prepared_dequant_up_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v4_mxfp4_prepared_dequant_up_bf16_gaudi2_o_end;
    } else if (normal_) {
        start = &_binary___deepseek_v4_mxfp4_prepared_dequant_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v4_mxfp4_prepared_dequant_normal_bf16_gaudi2_o_end;
    } else {
        start = &_binary___deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2_o_end;
    }
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - start;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
