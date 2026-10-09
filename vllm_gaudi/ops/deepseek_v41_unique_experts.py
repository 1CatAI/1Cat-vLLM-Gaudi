# SPDX-License-Identifier: Apache-2.0
"""Optional C2-C6 expert grouping with device-selected matrix row counts.

The normal C1 implementation is unchanged. Compiler-owned immutable program
banks are bound before capture; routing counts never leave the device.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def load_unique_expert_operators():
    """Load the additive operators from their independently built manifest."""
    root = Path(os.environ['VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR'])
    manifest = json.loads((root / 'deepseek_v41_unique_build.json').read_text())
    libraries = list(root.glob('hpu_dsv41_unique_pt2*.so'))
    if manifest.get('schema') != 1 or len(libraries) != 1:
        raise RuntimeError('Unique expert build manifest or registration is incomplete')
    if (manifest.get('control_mme_from_base') and
            not hasattr(torch.ops.custom_op, 'custom_deepseek_v41_control_mme_f32_gaudi2')):
        raise RuntimeError('The additive build requires the serving base to register mHC MME first')
    library = libraries[0]
    if hashlib.sha256(library.read_bytes()).hexdigest() != manifest.get('binaries', {}).get(library.name):
        raise RuntimeError('Unique expert registration differs from its build manifest')
    torch.ops.load_library(str(library))
    # Additional schemas may be rebuilt independently of the qualified base.
    # Validate the complete additive set before any new registration is loaded.
    extras = manifest.get("extra_registrations", [])
    paths = []
    for name in extras:
        if Path(name).name != name or not name.endswith(".so"):
            raise RuntimeError("Invalid additive registration name")
        extra = root / name
        if hashlib.sha256(extra.read_bytes()).hexdigest() != manifest.get("binaries", {}).get(name):
            raise RuntimeError("Additive registration differs from its build manifest")
        paths.append(extra)
    for extra in paths:
        torch.ops.load_library(str(extra))
    return library


class UniqueExpertInputs:
    def __init__(self, device):
        required = ('custom_deepseek_v41_unique_control_i32_gaudi2',
                    'custom_deepseek_v41_unique_pack_fp8_gaudi2',
                    'custom_deepseek_v41_unique_program_i32_gaudi2',
                    'custom_deepseek_v41_unique_expert_fp8_gaudi2',
                    'custom_deepseek_v41_unique_restore_bf16_gaudi2')
        if not all(hasattr(torch.ops.custom_op, name) for name in required):
            raise RuntimeError('Unique expert operators must be loaded before preparation')
        self.bank = torch.zeros((6, 65536), dtype=torch.int32, device=device)
        self.spans = torch.zeros((2048, 4), dtype=torch.int32, device=device)
        self.generation = torch.zeros(2, dtype=torch.int32, device=device)

    def tensors(self):
        return self.bank, self.spans

    def forward(self, quantized, scale, ids, routing, q13, p13, q2, p2, lookup, channel13, channel2):
        count = quantized.shape[0]
        if not 2 <= count <= 6:
            raise ValueError('Unique expert verification requires C2-C6')
        ops = torch.ops.custom_op
        metadata, controls = ops.custom_deepseek_v41_unique_control_i32_gaudi2(
            ids.to(torch.int32), self.generation, q13.shape[0])
        packed, input_scales, route_weights = ops.custom_deepseek_v41_unique_pack_fp8_gaudi2(
            quantized, scale, routing, metadata)
        program = ops.custom_deepseek_v41_unique_program_i32_gaudi2(self.bank, self.spans, metadata)
        matrices = ops.custom_deepseek_v41_unique_expert_fp8_gaudi2(
            packed, input_scales, route_weights, metadata, controls,
            q13, p13, q2, p2, lookup, channel13, channel2, program)
        return ops.custom_deepseek_v41_unique_restore_bf16_gaudi2(matrices, metadata, count)
