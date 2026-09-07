// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cstring>
#include <vector>
#include "tpc_kernel_lib_interface.h"

extern "C" tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId, uint32_t*, tpc_lib_api::GuidInfo*);

int main() {
    using namespace tpc_lib_api;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, nullptr, nullptr) == GLUE_FAILED);
    uint32_t count = 0;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, nullptr) == GLUE_SUCCESS);
    assert(count > 31);
    std::vector<GuidInfo> guids(count);
    uint32_t capacity = 1;
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &capacity, guids.data()) == GLUE_FAILED);
    assert(capacity == count);
    assert(GetKernelGuids(DEVICE_ID_GAUDI2, &count, guids.data()) == GLUE_SUCCESS);
    uint32_t custom_count = 0;
    for (const auto& guid : guids) {
        if (std::strstr(guid.name, "deepseek_v4")) ++custom_count;
    }
    assert(custom_count == 31);
}
