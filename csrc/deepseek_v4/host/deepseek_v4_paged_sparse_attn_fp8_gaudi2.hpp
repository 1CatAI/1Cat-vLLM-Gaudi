/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV4PagedSparseAttnFP8Gaudi2 {
public:
    enum TopkMode {
        GLOBAL_SLOTS,
        LOCAL_BLOCK_TABLE,
        SEQUENTIAL_BLOCK_TABLE,
        PAIR_SEQUENTIAL_BLOCK_TABLE,
        QNORM_SEQUENTIAL_BLOCK_TABLE,
        SWA_ONLY,
    };

    explicit DeepseekV4PagedSparseAttnFP8Gaudi2(
        TopkMode mode = GLOBAL_SLOTS, bool functionalOutput = false)
        : mode_(mode), functionalOutput_(functionalOutput)
    {
    }

    tpc_lib_api::GlueCodeReturn GetKernelName(
        char kernelName[tpc_lib_api::MAX_NODE_NAME]);

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* inDefs,
        tpc_lib_api::HabanaKernelInstantiation* outDefs);

private:
    TopkMode mode_;
    bool functionalOutput_;
};
