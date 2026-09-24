// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_prefill_mhc_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_prefill_mhc_post_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_mhc_post_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_mhc_collapse_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_mhc_collapse_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_mhc_rrms_bf16_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_mhc_rrms_bf16_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_prefill_mhc_post_prepare_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_prefill_mhc_post_prepare_gaudi2_o_end;
tpc_lib_api::GlueCodeReturn DeepseekV41PrefillMhcGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 4) { in->inputTensorNr = 4; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& x = in->inputTensors[0].geometry;
    const auto& r = in->inputTensors[1].geometry;
    const auto& p = in->inputTensors[2].geometry;
    const auto& c = in->inputTensors[3].geometry;
    const auto& y = in->outputTensors[0].geometry;
    for (unsigned i=0; i<5; ++i) {
        auto& g = i<4 ? in->inputTensors[i].geometry : in->outputTensors[0].geometry;
        const auto dtype = i==2 || i==3 ? DATA_F32 : DATA_BF16;
        if (g.dataType != dtype) { g.dataType=dtype; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    }
    if (x.dims!=2 || x.maxSizes[0]!=5120 || x.maxSizes[1]<1 || x.maxSizes[1]>8192 ||
        r.dims!=3 || r.maxSizes[0]!=5120 || r.maxSizes[1]!=4 || r.maxSizes[2]!=x.maxSizes[1] ||
        p.dims!=2 || p.maxSizes[0]!=4 || p.maxSizes[1]!=x.maxSizes[1] ||
        c.dims!=3 || c.maxSizes[0]!=4 || c.maxSizes[1]!=4 || c.maxSizes[2]!=x.maxSizes[1])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (y.dims!=3 || y.maxSizes[0]!=5120 || y.maxSizes[1]!=4 || y.maxSizes[2]!=x.maxSizes[1])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=2;
    out->indexSpaceGeometry[0]=40;
    out->indexSpaceGeometry[1]=x.maxSizes[1];
    for (unsigned i=0; i<5; ++i) {
        auto& ap = i<4 ? out->inputTensorAccessPattern[i] : out->outputTensorAccessPattern[0];
        const auto& g = i<4 ? in->inputTensors[i].geometry : in->outputTensors[0].geometry;
        for (unsigned d=0; d<g.dims; ++d) {
            auto& m=ap.mapping[d];
            if (d==g.dims-1) { m.indexSpaceDim=1; m.a=1; m.start_b=m.end_b=0; }
            else if (d==0 && i!=2 && i!=3) {
                m.indexSpaceDim=0; m.a=128; m.start_b=0; m.end_b=127;
            } else { m.indexSpaceDim=0; m.a=0; m.start_b=0; m.end_b=g.maxSizes[d]-1; }
        }
    }
    out->kernel.paramsNr=0;
    const auto* begin=&_binary___deepseek_v41_prefill_mhc_post_gaudi2_o_start;
    const auto* end=&_binary___deepseek_v41_prefill_mhc_post_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-begin;
    if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);
    return GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillMhcCollapseGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    auto& r = in->inputTensors[0].geometry;
    auto& p = in->inputTensors[1].geometry;
    auto& y = in->outputTensors[0].geometry;
    if (r.dataType != DATA_BF16) { r.dataType=DATA_BF16; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    if (p.dataType != DATA_F32) { p.dataType=DATA_F32; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    if (y.dataType != DATA_BF16) { y.dataType=DATA_BF16; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    if (r.dims!=3 || r.maxSizes[0]!=5120 || r.maxSizes[1]!=4 ||
        r.maxSizes[2]<1 || r.maxSizes[2]>8192 ||
        p.dims!=2 || p.maxSizes[0]!=4 || p.maxSizes[1]!=r.maxSizes[2])
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (y.dims!=2 || y.maxSizes[0]!=5120 || y.maxSizes[1]!=r.maxSizes[2])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=2;
    out->indexSpaceGeometry[0]=40;
    out->indexSpaceGeometry[1]=r.maxSizes[2];
    auto map=[](TensorAccessPattern& ap, const TensorGeometry& g, bool feature) {
        for (unsigned d=0; d<g.dims; ++d) {
            auto& m=ap.mapping[d];
            if (d==g.dims-1) { m.indexSpaceDim=1; m.a=1; m.start_b=m.end_b=0; }
            else if (d==0 && feature) { m.indexSpaceDim=0; m.a=128; m.start_b=0; m.end_b=127; }
            else { m.indexSpaceDim=0; m.a=0; m.start_b=0; m.end_b=g.maxSizes[d]-1; }
        }
    };
    map(out->inputTensorAccessPattern[0],r,true);
    map(out->inputTensorAccessPattern[1],p,false);
    map(out->outputTensorAccessPattern[0],y,true);
    out->kernel.paramsNr=0;
    const auto* begin=&_binary___deepseek_v41_prefill_mhc_collapse_gaudi2_o_start;
    const auto* end=&_binary___deepseek_v41_prefill_mhc_collapse_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-begin;
    if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);
    return GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillMhcRrmsGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 1) { in->inputTensorNr = 1; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    auto& r = in->inputTensors[0].geometry;
    auto& y = in->outputTensors[0].geometry;
    if (r.dataType != DATA_BF16) { r.dataType=DATA_BF16; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    if (y.dataType != DATA_F32) { y.dataType=DATA_F32; return GLUE_INCOMPATIBLE_DATA_TYPE; }
    if (r.dims!=3 || r.maxSizes[0]!=5120 || r.maxSizes[1]!=4 ||
        r.maxSizes[2]<1 || r.maxSizes[2]>8192)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (y.dims!=2 || y.maxSizes[0]!=1 || y.maxSizes[1]!=r.maxSizes[2])
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=1;
    out->indexSpaceGeometry[0]=r.maxSizes[2];
    for (unsigned d=0; d<r.dims; ++d) {
        auto& m=out->inputTensorAccessPattern[0].mapping[d];
        if (d==2) { m.indexSpaceDim=0; m.a=1; m.start_b=m.end_b=0; }
        else { m.indexSpaceDim=0; m.a=0; m.start_b=0; m.end_b=r.maxSizes[d]-1; }
    }
    out->outputTensorAccessPattern[0].mapping[0] = {0, 0, 0, 0};
    out->outputTensorAccessPattern[0].mapping[1] = {0, 1, 0, 0};
    out->kernel.paramsNr=0;
    const auto* begin=&_binary___deepseek_v41_prefill_mhc_rrms_bf16_gaudi2_o_start;
    const auto* end=&_binary___deepseek_v41_prefill_mhc_rrms_bf16_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-begin;
    if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);
    return GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn DeepseekV41PrefillMhcPostPrepareGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out) {
    using namespace tpc_lib_api;
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 3) { in->outputTensorNr = 3; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    auto& x = in->inputTensors[0].geometry;
    auto& r = in->inputTensors[1].geometry;
    auto& p = in->inputTensors[2].geometry;
    auto& c = in->inputTensors[3].geometry;
    auto& n = in->inputTensors[4].geometry;
    auto& ro = in->outputTensors[0].geometry;
    auto& co = in->outputTensors[1].geometry;
    auto& rms = in->outputTensors[2].geometry;
    const TensorDataType input_types[5] = {DATA_BF16, DATA_BF16, DATA_F32, DATA_F32, DATA_F32};
    const TensorDataType output_types[3] = {DATA_BF16, DATA_BF16, DATA_F32};
    for (unsigned i=0; i<5; ++i)
        if (in->inputTensors[i].geometry.dataType != input_types[i]) {
            in->inputTensors[i].geometry.dataType=input_types[i]; return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    for (unsigned i=0; i<3; ++i)
        if (in->outputTensors[i].geometry.dataType != output_types[i]) {
            in->outputTensors[i].geometry.dataType=output_types[i]; return GLUE_INCOMPATIBLE_DATA_TYPE;
        }
    const unsigned tokens=x.maxSizes[1];
    if (x.dims!=2 || x.maxSizes[0]!=5120 || tokens<1 || tokens>8192 ||
        r.dims!=3 || r.maxSizes[0]!=5120 || r.maxSizes[1]!=4 || r.maxSizes[2]!=tokens ||
        p.dims!=2 || p.maxSizes[0]!=4 || p.maxSizes[1]!=tokens ||
        c.dims!=3 || c.maxSizes[0]!=4 || c.maxSizes[1]!=4 || c.maxSizes[2]!=tokens ||
        n.dims!=2 || n.maxSizes[0]!=4 || n.maxSizes[1]!=tokens)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (ro.dims!=3 || ro.maxSizes[0]!=5120 || ro.maxSizes[1]!=4 || ro.maxSizes[2]!=tokens ||
        co.dims!=2 || co.maxSizes[0]!=5120 || co.maxSizes[1]!=tokens ||
        rms.dims!=2 || rms.maxSizes[0]!=1 || rms.maxSizes[1]!=tokens)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank=1;
    out->indexSpaceGeometry[0]=tokens;
    auto token_map=[](TensorAccessPattern& ap, const TensorGeometry& g) {
        for (unsigned d=0; d<g.dims; ++d) {
            auto& m=ap.mapping[d];
            if (d==g.dims-1) { m.indexSpaceDim=0; m.a=1; m.start_b=m.end_b=0; }
            else { m.indexSpaceDim=0; m.a=0; m.start_b=0; m.end_b=g.maxSizes[d]-1; }
        }
    };
    for (unsigned i=0; i<5; ++i) token_map(out->inputTensorAccessPattern[i],in->inputTensors[i].geometry);
    for (unsigned i=0; i<3; ++i) token_map(out->outputTensorAccessPattern[i],in->outputTensors[i].geometry);
    out->kernel.paramsNr=0;
    const auto* begin=&_binary___deepseek_v41_prefill_mhc_post_prepare_gaudi2_o_start;
    const auto* end=&_binary___deepseek_v41_prefill_mhc_post_prepare_gaudi2_o_end;
    const unsigned capacity=out->kernel.elfSize;
    out->kernel.elfSize=end-begin;
    if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);
    return GLUE_SUCCESS;
}
