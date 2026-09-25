// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_route_pack_gaudi2.hpp"
#include <cstring>
extern unsigned char _binary___deepseek_v41_route_pack_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_route_pack_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_route_reduce_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_route_reduce_gaudi2_o_end;
extern unsigned char _binary___deepseek_v41_route_order_gaudi2_o_start;
extern unsigned char _binary___deepseek_v41_route_order_gaudi2_o_end;
using namespace tpc_lib_api;
GlueCodeReturn DeepseekV41RoutePackGaudi2::GetGcDefinitions(
    HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out) return GLUE_FAILED;
    const bool reduce_ = mode_ == 1;
    if (mode_ == 2) {
        if (p->inputTensorNr != 1 || p->outputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
        const auto& ids = p->inputTensors[0].geometry;
        const auto& inverse = p->outputTensors[0].geometry;
        const uint64_t r=ids.maxSizes[0];
        if (ids.dataType != DATA_I32 || ids.dims != 2 || ids.maxSizes[1] != 1 ||
            !r || r>384 || r%6 || inverse.dataType != DATA_I32 || inverse.dims != 1 || inverse.maxSizes[0] != r)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out->indexSpaceRank=1;out->indexSpaceGeometry[0]=r;
        out->inputTensorAccessPattern[0].allRequired=true;
        out->outputTensorAccessPattern[0].mapping[0]={0,1,0,0};
        out->kernel.paramsNr=0;
        const unsigned capacity=out->kernel.elfSize;
        out->kernel.elfSize=&_binary___deepseek_v41_route_order_gaudi2_o_end -
                            &_binary___deepseek_v41_route_order_gaudi2_o_start;
        if (capacity<out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
        std::memcpy(out->kernel.kernelElf,&_binary___deepseek_v41_route_order_gaudi2_o_start,out->kernel.elfSize);
        return GLUE_SUCCESS;
    }
    const unsigned inputs = reduce_ ? 2 : 5, outputs = reduce_ ? 1 : 4;
    if (p->inputTensorNr != inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& x = p->inputTensors[0].geometry;
    const auto& ids = p->inputTensors[reduce_ ? 1 : 2].geometry;
    const uint64_t h = x.maxSizes[0], routes = ids.maxSizes[0];
    if (!h || h > 5120 || h % 256 || !routes || routes > 384 || routes % 6 ||
        ids.dataType != DATA_I32 || ids.dims != (reduce_ ? 1u : 2u) ||
        (!reduce_ && ids.maxSizes[1] != 1)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t batch = routes / 6;
    if (reduce_) {
        if (x.dataType != DATA_BF16 || x.dims != 3 || x.maxSizes[1] != 1 || x.maxSizes[2] != routes)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const auto& y = p->outputTensors[0].geometry;
        if (y.dataType != DATA_BF16 || y.dims != 2 || y.maxSizes[0] != h || y.maxSizes[1] != batch)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out->indexSpaceRank = 2;
        out->indexSpaceGeometry[0] = h / 128;
        out->indexSpaceGeometry[1] = batch;
        out->inputTensorAccessPattern[0].mapping[0] = {0,128,0,127};
        out->inputTensorAccessPattern[0].mapping[1] = {0,0,0,0};
        out->inputTensorAccessPattern[0].mapping[2] = {1,0,0,static_cast<int>(routes)-1};
        out->inputTensorAccessPattern[1].mapping[0] = {1,6,0,5};
        out->outputTensorAccessPattern[0].mapping[0] = {0,128,0,127};
        out->outputTensorAccessPattern[0].mapping[1] = {1,1,0,0};
    } else {
        const auto& s = p->inputTensors[1].geometry;
        const auto& r = p->inputTensors[3].geometry;
        const auto& inverse = p->inputTensors[4].geometry;
        if (inverse.dataType != DATA_I32 || inverse.dims != 1 || inverse.maxSizes[0] != routes ||
            x.dataType != DATA_U8 || x.dims != 2 || x.maxSizes[1] != batch ||
            s.dataType != DATA_F32 || s.dims != 2 || s.maxSizes[0] != 1 || s.maxSizes[1] != batch ||
            r.dataType != DATA_F32 || r.dims != 2 || r.maxSizes[0] != routes || r.maxSizes[1] != 1)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const TensorDataType types[] = {DATA_U8,DATA_F32,DATA_I32,DATA_F32,DATA_I32};
        const unsigned dims[] = {3,3,2,2,1};
        const uint64_t sizes[][3] = {{h,1,routes},{1,1,routes},{routes,1,1},{1,routes,1},{routes,1,1}};
        for (unsigned i=0;i<outputs;++i) {
            const auto& y=p->outputTensors[i].geometry;
            if (y.dataType != types[i] || y.dims != dims[i]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            for (unsigned d=0;d<dims[i];++d)
                if (y.maxSizes[d] != sizes[i][d]) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            // Stable rank is data-dependent; prevent a compiler slice from
            // assigning a writable subset that excludes another route rank.
            out->outputTensorAccessPattern[i].allRequired = true;
        }
        out->indexSpaceRank = 1;
        out->indexSpaceGeometry[0] = routes;
        for (unsigned i=0;i<inputs;++i) out->inputTensorAccessPattern[i].allRequired = true;
    }
    out->kernel.paramsNr = 0;
    const auto* begin = reduce_ ? &_binary___deepseek_v41_route_reduce_gaudi2_o_start :
                                 &_binary___deepseek_v41_route_pack_gaudi2_o_start;
    const auto* end = reduce_ ? &_binary___deepseek_v41_route_reduce_gaudi2_o_end :
                               &_binary___deepseek_v41_route_pack_gaudi2_o_end;
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = end-begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf,begin,out->kernel.elfSize);
    return GLUE_SUCCESS;
}
