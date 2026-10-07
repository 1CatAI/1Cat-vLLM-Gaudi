// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tpc_test_core_api.h>
#include <limits>
#include <stdexcept>

namespace gaudi_validation {
// TPC tensor_config encodings are not tpc_lib_api::TensorDataType bit flags.
// FP32 is 7. Integer encodings are I8/I16/I32=0/1/2 and
// U8/U16/U32=3/4/5; a wrong size silently changes GEN_ADDR byte strides.
inline unsigned tensor_config_type(tpc_lib_api::TensorDataType type) {
    using namespace tpc_lib_api;
    switch (type) {
    case DATA_I8: return 0;
    case DATA_U8: return 3;
    case DATA_I16: return 1;
    case DATA_U16: return 4;
    case DATA_I32: return 2;
    case DATA_U32: return 5;
    case DATA_BF16: return 6;
    case DATA_F32: return 7;
    case DATA_F16: return 8;
    case DATA_F8_152: return 9;
    case DATA_F8_143: return 10;
    default: throw std::invalid_argument("Unsupported simulator tensor dtype");
    }
}

inline tpc_tests::TensorDesc2 descriptor(void* data, const tpc_lib_api::Tensor& tensor) {
    const auto& geometry = tensor.geometry;
    if (!data || geometry.dims < 1 || geometry.dims > 5)
        throw std::invalid_argument("Invalid simulator tensor storage or rank");
    tpc_tests::TensorDesc2 result{};
    result.baseAddrUnion.baseAddr = reinterpret_cast<uint64_t>(data);
    result.configuration = tensor_config_type(geometry.dataType) | (0x1f << 8) | ((geometry.dims - 1) << 16);
    uint64_t stride = 1;
    for (unsigned dimension = 0; dimension < 5; ++dimension) {
        const uint64_t size = dimension < geometry.dims ? geometry.maxSizes[dimension] : 1;
        if (!size || size > UINT32_MAX || stride > UINT32_MAX)
            throw std::invalid_argument("Simulator tensor needs unsupported wide strides");
        result.dimDescriptors[dimension].size = size;
        result.dimDescriptors[dimension].stride = stride;
        stride *= size;
    }
    return result;
}
}  // namespace gaudi_validation
