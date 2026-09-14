// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tpc_kernel_lib_interface.h"

class DeepseekV4Mxfp4PreparedDequantBF16Gaudi2 {
    bool normal_;
    int half_;
    bool v41_;
    bool shared_;
    bool tiled_;
    bool window_;
    unsigned legacy_;

 public:
    enum LegacyMode { LEGACY_K128, LEGACY_PIPELINE };
    explicit DeepseekV4Mxfp4PreparedDequantBF16Gaudi2(bool normal = false, int half = -1, bool v41 = false,
                                                  bool shared = false, bool tiled = false, bool window = false)
        : normal_(normal), half_(half), v41_(v41), shared_(shared), tiled_(tiled), window_(window), legacy_(0) {}
    explicit DeepseekV4Mxfp4PreparedDequantBF16Gaudi2(LegacyMode mode, bool normal = false)
        : normal_(normal), half_(-1), v41_(true), shared_(false), tiled_(true), window_(false),
          legacy_(mode == LEGACY_PIPELINE ? 2 : 1) {}
    tpc_lib_api::GlueCodeReturn GetKernelName(char name[tpc_lib_api::MAX_NODE_NAME]);
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};
