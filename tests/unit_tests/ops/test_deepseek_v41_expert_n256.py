# SPDX-License-Identifier: Apache-2.0
"""Reversible compressed storage and FP8 range qualification."""
import numpy as np
import pytest
import json
import struct
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, restore_expert
from vllm_gaudi.ops.deepseek_v41_weights import prepare_q16, prepare_s16


@pytest.mark.parametrize("n,k", [(256, 128), (2304, 5120), (5120, 1152)])
def test_repack_preserves_nibbles_scales_and_storage(n, k):
    rng = np.random.default_rng(410)
    packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    q, logical = prepare_q16(packed)
    codes = rng.integers(113, 119, (n, k // 32), dtype=np.uint8)
    s, _ = prepare_s16(codes, logical)
    nq, ns, channels, record = prepare_expert(q, s)
    rq, rs = restore_expert(nq, ns)
    assert np.array_equal(q, rq) and np.array_equal(s, rs)
    assert nq.nbytes == q.nbytes and ns.nbytes == s.nbytes
    assert channels.shape == (n // 256, 256)
    assert record["temporary_upper_bound_bytes"] < 2 * 2**30


def test_zero_and_scale_boundaries_retain_original_encodings():
    for encoding in (0x00, 0x88, 0x77, 0xFF):
        q, logical = prepare_q16(np.full((256, 64), encoding, dtype=np.uint8))
        for code in ((2, 127, 254) if encoding in (0x00, 0x88) else (7, 127, 250)):
            s, _ = prepare_s16(np.full((256, 4), code, dtype=np.uint8), logical)
            nq, ns, _, _ = prepare_expert(q, s)
            rq, rs = restore_expert(nq, ns)
            assert np.array_equal(q, rq) and np.array_equal(s, rs)
        if encoding not in (0x00, 0x88):
            s, _ = prepare_s16(np.full((256, 4), 2, dtype=np.uint8), logical)
            with pytest.raises(ValueError, match="normal BF16 power"):
                prepare_expert(q, s)
        for code in (0, 1, 255):
            s, _ = prepare_s16(np.full((256, 4), code, dtype=np.uint8), logical)
            with pytest.raises(ValueError, match="finite normal"):
                prepare_expert(q, s)
        with pytest.raises(ValueError, match="exact E8M0"):
            prepare_expert(q, s | 1)


def test_invalid_output_width_and_truncated_planes_fail():
    q, logical = prepare_q16(np.full((128, 64), 0x77, dtype=np.uint8))
    s, _ = prepare_s16(np.full((128, 4), 127, dtype=np.uint8), logical)
    with pytest.raises(ValueError, match="divisible"):
        prepare_expert(q, s)
    with pytest.raises(ValueError, match="dimensions"):
        restore_expert(np.zeros((1, 8192), dtype=np.int16), np.zeros((1, 1023), dtype=np.int16))


def test_bounded_loader_handles_partial_final_batch(tmp_path):
    from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection
    from vllm_gaudi.ops.deepseek_v41_weights import read_header
    rng = np.random.default_rng(17)
    q = rng.integers(-32768, 32768, (17, 2, 4096), dtype=np.int16)
    s = np.full((17, 2, 512), 127 << 7, dtype=np.uint16)
    header = json.dumps({
        "w_q16": {
            "dtype": "I16",
            "shape": list(q.shape),
            "data_offsets": [0, q.nbytes]
        },
        "w_s16": {
            "dtype": "BF16",
            "shape": list(s.shape),
            "data_offsets": [q.nbytes, q.nbytes + s.nbytes]
        }
    }).encode()
    path = tmp_path / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + q.tobytes() + s.tobytes())
    shard = SimpleNamespace(catalog=read_header(path), check_identity=lambda: None)
    nq, ns, channels = load_projection(shard, "w", "cpu")
    assert channels.shape == (17, 1, 256)
    for expert in range(17):
        rq, rs = restore_expert(nq[expert].numpy(), ns[expert].numpy())
        assert np.array_equal(rq, q[expert]) and np.array_equal(rs, s[expert])


@pytest.mark.parametrize("n256,dspark,valid", [(True, False, True), (False, False, False), (True, True, False)])
def test_native_stage_retains_mhc_overlap_with_bf16_n256_boundaries(monkeypatch, n256, dspark, valid):
    from vllm_gaudi.models import deepseek_v41_program as program
    from vllm_gaudi.compilation import deepseek_v41_overlap as overlap
    monkeypatch.setenv("VLLM_HPU_DSV41_TP_MHC_OVERLAP", "1")
    backend = object()
    monkeypatch.setattr(overlap, "make_backend", lambda: backend)
    calls = []
    monkeypatch.setattr(program, "_compile_group", lambda group, **kwargs: calls.append((group, kwargs)))
    from torch import nn
    stage = SimpleNamespace(layers=[nn.Module() for _ in range(20)],
                            fp8_decode=True,
                            expert_n256=n256,
                            dspark=dspark,
                            pp_rank=0,
                            config={"text_config": {
                                "rms_norm_eps": 1e-6
                            }})
    if valid:
        compiled = program.CompiledStage(stage, native=True)
        assert len(compiled.groups) == len(calls) == 5
        assert all(group.fp8_decode and options == {"native": True, "backend": backend} for group, options in calls)
    else:
        with pytest.raises(ValueError, match="BF16 boundaries"):
            program.CompiledStage(stage, native=True)
        assert not calls
