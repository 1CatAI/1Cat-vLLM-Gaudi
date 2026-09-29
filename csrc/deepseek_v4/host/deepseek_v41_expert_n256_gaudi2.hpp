// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV41ExpertN256Gaudi2 {
 public:
    enum Mode { FP8, FP8Slots, FP8Reuse, FP8Horizontal, BF16, DeadBF16, DeadNormalBF16,
                Scale, SiluQuant, ScaleReduce, NormalBF16 };
    explicit DeepseekV41ExpertN256Gaudi2(Mode mode, bool substitute_prefetch16 = false)
        : mode_(mode), substitute_prefetch16_(substitute_prefetch16) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
 private:
    Mode mode_;
    bool substitute_prefetch16_;
};
