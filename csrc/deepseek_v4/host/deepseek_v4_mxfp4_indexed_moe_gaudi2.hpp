/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV4Mxfp4IndexedMoeGaudi2 {
public:
    enum Stage {
        FC1,
        FC2,
    };

    explicit DeepseekV4Mxfp4IndexedMoeGaudi2(Stage stage)
        : stage_(stage)
    {
    }

    tpc_lib_api::GlueCodeReturn GetKernelName(
        char kernelName[tpc_lib_api::MAX_NODE_NAME]);

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* inDefs,
        tpc_lib_api::HabanaKernelInstantiation* outDefs);

private:
    Stage stage_;
};
