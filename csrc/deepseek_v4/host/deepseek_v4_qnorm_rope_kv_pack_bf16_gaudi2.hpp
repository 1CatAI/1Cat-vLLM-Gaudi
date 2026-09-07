/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV4QnormRopeKvPackBF16Gaudi2 {
public:
    enum Mode { BF16_Q, HYBRID_FP8_BF16_Q };

    explicit DeepseekV4QnormRopeKvPackBF16Gaudi2(Mode mode = BF16_Q)
        : mode_(mode)
    {
    }

    tpc_lib_api::GlueCodeReturn GetKernelName(
        char kernelName[tpc_lib_api::MAX_NODE_NAME]);

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* inDefs,
        tpc_lib_api::HabanaKernelInstantiation* outDefs);

private:
    Mode mode_;
};
