# SPDX-License-Identifier: Apache-2.0
"""CPU layout conversion preserves all compressed bytes and scale decisions."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, restore_expert
from vllm_gaudi.ops.deepseek_v41_startup_pack import native_q16_packer


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    directory = tmp_path_factory.mktemp("startup-pack")
    path = directory / "pack.so"
    source = Path(__file__).resolve().parents[3] / "csrc/deepseek_v41_startup_pack.cpp"
    subprocess.run(["g++", "-std=c++17", "-O3", "-fPIC", "-shared", str(source), "-o", str(path)], check=True)
    path.with_suffix(".json").write_text(
        json.dumps(dict(schema=1, abi=1, sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    return path


@pytest.mark.parametrize("compact", [False, True])
def test_native_packing_preserves_signed_nibbles_scales_and_inverse(library, compact):
    packer = native_q16_packer(library)
    random = np.random.default_rng(1050)
    for blocks, k in ((2, 128), (4, 256), (8, 512)):
        q = random.integers(-32768, 32768, (blocks, k * 32), dtype=np.int16)
        s = np.full((blocks, k * 4), 127 << 7, dtype=np.uint16)
        reference = prepare_expert(q, s, compact_scales=compact)
        candidate = prepare_expert(q, s, compact_scales=compact, pack_q16=packer)
        assert all(np.array_equal(a, b) for a, b in zip(reference[:3], candidate[:3], strict=True))
        original_q, original_s = restore_expert(*candidate[:2])
        assert np.array_equal(original_q, q)
        assert np.array_equal(original_s, s)


def test_native_packing_rejects_noncontiguous_source_and_bad_manifest(library, tmp_path):
    packer = native_q16_packer(library)
    with pytest.raises(ValueError, match="contiguous"):
        packer(np.zeros((2, 8192), dtype=np.int16)[:, ::2])
    broken = tmp_path / "bad.so"
    broken.write_bytes(library.read_bytes())
    broken.with_suffix(".json").write_text(json.dumps(dict(schema=1, abi=1, sha256="0" * 64)))
    with pytest.raises(ValueError, match="build contract"):
        native_q16_packer(broken)
