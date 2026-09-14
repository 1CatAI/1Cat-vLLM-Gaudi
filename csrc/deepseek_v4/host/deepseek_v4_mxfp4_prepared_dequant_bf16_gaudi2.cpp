// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v4_mxfp4_prepared_dequant_bf16_gaudi2.hpp"
#include <cstring>

extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_end;

extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_end;

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
    if (pipeline_) {
        kernel = "custom_deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2";
    } else if (v41_ && k128_) {
        static_assert(sizeof("custom_deepseek_v41_mxfp4_prepared_dequant_k128n_bf16_gaudi2") <=
                      tpc_lib_api::MAX_NODE_NAME);
        kernel = normal_ ? "custom_deepseek_v41_mxfp4_prepared_dequant_k128n_bf16_gaudi2"
                         : "custom_deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2";
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
    if (in->inputTensorNr != 4) {
        in->inputTensorNr = 4;
        return GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
    if (in->outputTensorNr != 1) {
        in->outputTensorNr = 1;
        return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    }
    const TensorDataType types[] = {DATA_I32, DATA_I16, DATA_BF16, DATA_BF16};
    for (unsigned index = 0; index < 4; ++index) {
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
    const unsigned nBlocks = half_ >= 0 ? 8 : q16.maxSizes[1];
    if (output.dims != 3 || output.maxSizes[0] != nBlocks * 128 ||
        output.maxSizes[1] != q16.maxSizes[0] / 32 || output.maxSizes[2] != ids.maxSizes[0]) {
        output.dims = 3;
        output.maxSizes[0] = nBlocks * 128;
        output.maxSizes[1] = q16.maxSizes[0] / 32;
        output.maxSizes[2] = ids.maxSizes[0];
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    if (k128_ && (!v41_ || half_ >= 0)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (pipeline_ && (!k128_ || !normal_)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    out->indexSpaceRank = k128_ ? 3 : 2;
    out->indexSpaceGeometry[0] = nBlocks;
    out->indexSpaceGeometry[1] = ids.maxSizes[0];
    if (k128_) out->indexSpaceGeometry[2] = q16.maxSizes[0] / 4096;
    auto map = [](TensorAccessPattern& pattern, unsigned dim, unsigned indexSpaceDim,
                  int coefficient, int start, int end) {
        pattern.mapping[dim].indexSpaceDim = indexSpaceDim;
        pattern.mapping[dim].a = coefficient;
        pattern.mapping[dim].start_b = start;
        pattern.mapping[dim].end_b = end;
    };
    map(out->inputTensorAccessPattern[0], 0, 1, 1, 0, 0);
    map(out->inputTensorAccessPattern[0], 1, 1, 0, 0, 0);
    auto& qPattern = out->inputTensorAccessPattern[1];
    map(qPattern, 0, 0, 0, 0, q16.maxSizes[0] - 1);
    const int nBlockOffset = half_ == 1 ? 8 : 0;
    map(qPattern, 1, 0, 1, nBlockOffset, nBlockOffset);
    map(qPattern, 2, 1, 0, 0, q16.maxSizes[2] - 1);
    auto& sPattern = out->inputTensorAccessPattern[2];
    map(sPattern, 0, 0, 0, 0, s16.maxSizes[0] - 1);
    map(sPattern, 1, 0, 1, nBlockOffset, nBlockOffset);
    map(sPattern, 2, 1, 0, 0, q16.maxSizes[2] - 1);
    map(out->inputTensorAccessPattern[3], 0, 0, 0, 0, 127);
    auto& outputPattern = out->outputTensorAccessPattern[0];
    map(outputPattern, 0, 0, 128, 0, 127);
    map(outputPattern, 1, 0, 0, 0, q16.maxSizes[0] / 32 - 1);
    map(outputPattern, 2, 1, 1, 0, 0);
    if (k128_) {
        // Partition only the independent decoder work. The consuming MME
        // retains complete K; no partial products or Split-K reduction.
        map(qPattern, 0, 2, 4096, 0, 4095);
        map(sPattern, 0, 2, 512, 0, 511);
        map(outputPattern, 1, 2, 128, 0, 127);
    }
    out->kernel.paramsNr = 0;
    const unsigned char* start = nullptr;
    const unsigned char* end = nullptr;
    if (pipeline_) {
        start = &_binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_prepared_dequant_pipe_bf16_gaudi2_o_end;
    } else if (k128_ && normal_) {
        start = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_normal_bf16_gaudi2_o_end;
    } else if (k128_) {
        start = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_start;
        end = &_binary___deepseek_v41_mxfp4_prepared_dequant_k128_bf16_gaudi2_o_end;
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
