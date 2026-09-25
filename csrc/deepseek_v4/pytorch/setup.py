# SPDX-License-Identifier: Apache-2.0
"""Build against the source/generated headers of the installed Gaudi Bridge."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir


def directory(name):
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        raise RuntimeError(f"{name} must point to the matching Gaudi Bridge headers")
    return Path(value).resolve()


bridge = directory("GAUDI_PYTORCH_BRIDGE_ROOT")
build = directory("GAUDI_PYTORCH_BRIDGE_BUILD_ROOT")
includes = [
    get_include_dir(),
    "/usr/include/habanalabs",
    "/usr/include/habanalabs/hl_logger",
    str(bridge),
    str(bridge / "pytorch_helpers"),
    str(bridge / "python_packages/habana_frameworks/torch/jit/csrc"),
    str(bridge / "pytorch_helpers/habana_helpers/habana_serialization/include"),
    str(build),
    str(build / "_deps"),
    str(build / "_deps/abseil-cpp-src"),
    str(build / "_deps/magic_enum-src/include"),
    str(build / "_deps/fmt-src/include"),
    str(build / "_deps/exprtk-src/include"),
    str(build / "_deps/nlohmann_json-src/include"),
]
if os.environ.get("GAUDI_HL_LOGGER_INCLUDE"):
    includes.append(str(directory("GAUDI_HL_LOGGER_INCLUDE")))

setup(
    name="hpu_dsv4_sparse_attn_pt2",
    ext_modules=[
        CppExtension(
            "hpu_dsv4_sparse_attn_pt2",
            [
                "hpu_dsv4_sparse_attn_pt2.cpp", "hpu_dsv4_mhc_pt2.cpp", "hpu_dsv4_router_pt2.cpp",
                "hpu_dsv4_mxfp4_mme_pt2.cpp", "hpu_dsv4_native_attention_pt2.cpp", "hpu_dsv4_sinkhorn_pt2.cpp",
                "hpu_dsv41_mxfp4_mme_pt2.cpp", "hpu_dsv41_expert_n256_pt2.cpp", "hpu_dsv41_grouped_n256_pt2.cpp",
                "hpu_dsv41_route_pack_pt2.cpp", "hpu_dsv41_prefill_permuted_pt2.cpp",
                "hpu_dsv41_prefill_q_projection_pt2.cpp", "hpu_dsv41_prefill_index_pt2.cpp",
                "hpu_dsv41_prefill_topk_pt2.cpp", "hpu_dsv41_prefill_route_pt2.cpp", "hpu_dsv41_prefill_mhc_pt2.cpp",
                "hpu_dsv41_candidate_gather_pt2.cpp", "hpu_dsv41_prefill_flash_pt2.cpp",
                "hpu_dsv41_prefill_sparse_mla_pt2.cpp", "hpu_dsv41_prefill_bmm_pt2.cpp",
                "hpu_dsv41_prefill_paged_index_pt2.cpp", "hpu_dsv41_prefill_main_decode_pt2.cpp",
                "hpu_dsv41_indexed_moe_pt2.cpp", "hpu_dsv41_quant_roundtrip_pt2.cpp", "hpu_dsv41_selected_kv_pt2.cpp",
                "hpu_dsv41_paged_attention_pt2.cpp", "hpu_dsv41_selected_mla_pt2.cpp", "hpu_dsv41_kv_pack_pt2.cpp",
                "hpu_dsv41_kv_norm_rope_pt2.cpp", "hpu_dsv41_ffn_norm_quant_pt2.cpp", "hpu_dsv41_rope_pt2.cpp",
                "hpu_dsv41_prefix_layout_pt2.cpp", "hpu_dsv41_control_gemv_pt2.cpp", "hpu_dsv41_csa2_prep_pt2.cpp",
                "hpu_dsv41_compressor_pair_pt2.cpp", "hpu_dsv41_compressor_batch_pt2.cpp",
                "hpu_dsv41_compressor_batch_gather_pt2.cpp", "hpu_dsv41_decoded_kv_pt2.cpp",
                "hpu_dsv41_fp4_pack_pt2.cpp", "hpu_dsv41_fp8_operands_pt2.cpp", "hpu_dsv41_dense_fp8_pt2.cpp",
                "hpu_dsv41_q_projection_rope_pt2.cpp", "hpu_dsv41_mla_mme_pt2.cpp", "hpu_dsv41_router_top6_pt2.cpp",
                "hpu_dsv41_router_logits_top6_pt2.cpp", "hpu_dsv41_swa_pack_pt2.cpp", "hpu_dsv41_woa_fp8_pt2.cpp",
                "hpu_dsv41_bf16_linear_f32_pt2.cpp", "hpu_dsv41_mhc_gates_pt2.cpp", "hpu_dsv41_index_pt2.cpp",
                "hpu_dsv41_index_keys_pt2.cpp", "hpu_dsv41_reindex_compact_pt2.cpp", "hpu_dsv41_reindex_tile_pt2.cpp",
                "hpu_dsv41_state_rows_pt2.cpp", "hpu_dsv41_index_reduce_pt2.cpp", "hpu_dsv41_batch_mla_metadata_pt2.cpp"
            ],
            include_dirs=includes,
            library_dirs=[get_lib_dir()],
            libraries=["habana_pytorch2_plugin.upstream", "habana_pytorch_backend.upstream"],
            extra_compile_args=["-O2", "-std=c++17", "-DFMT_HEADER_ONLY=1", "-DGENERIC_HELPERS"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
