/**********************************************************************
Copyright (c) 2026 Habana Labs. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the conditions in the project
license are met.
********************************************************************/

#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class DeepseekV4SavePartialStatesF32Gaudi2 {
public:
    tpc_lib_api::GlueCodeReturn GetKernelName(
        char kernelName[tpc_lib_api::MAX_NODE_NAME]);

    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* inDefs,
        tpc_lib_api::HabanaKernelInstantiation* outDefs);
};
