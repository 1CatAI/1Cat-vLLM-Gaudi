# SPDX-License-Identifier: Apache-2.0
"""Parallel preparation preserves the loader's bytes and allocation bound."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_gaudi.ops.deepseek_v41_expert_load import prepare_expert_batch, preparation_workers
from vllm_gaudi.ops.deepseek_v41_expert_n256 import load_projection
from vllm_gaudi.ops.deepseek_v41_weights import prepare_q16


def sources(tmp_path, *, tp_size=4):
    rng = np.random.default_rng(26)
    packed = rng.integers(0, 256, (3, 256, 64), dtype=np.uint8)
    q = np.stack([prepare_q16(value)[0] for value in packed])
    scales = rng.integers(124, 129, (3, 2, 512), dtype=np.uint16) << 7
    prefix = "layers.20.ffn.experts.w13"
    catalog = {}
    for name, array, dtype in [("_q16", q, "I16"), ("_s16", scales, "BF16")]:
        path = tmp_path / name
        path.write_bytes(array.tobytes())
        catalog[prefix + name] = SimpleNamespace(file=path,
                                                 offset=0,
                                                 dtype=dtype,
                                                 shape=array.shape,
                                                 nbytes=array.nbytes)
    (tmp_path / "config.json").write_text('{"text_config": {"moe_intermediate_size": 512}}')
    shard = SimpleNamespace(catalog=catalog,
                            directory=tmp_path,
                            manifest={"tensor_parallel_size": tp_size},
                            tensor_parallel_size=tp_size,
                            specs={},
                            check_identity=lambda: None)
    return shard, prefix


@pytest.mark.parametrize("tp_size", [2, 4])
def test_rank_local_parallel_loader_matches_serial_bytes(tmp_path, monkeypatch, tp_size):
    monkeypatch.delenv("VLLM_HPU_DSV41_N256_PREPARED_DIR", raising=False)
    shard, prefix = sources(tmp_path, tp_size=tp_size)
    monkeypatch.setenv("VLLM_HPU_DSV41_N256_LOAD_WORKERS", "1")
    reference = load_projection(shard, prefix, "cpu")
    monkeypatch.setenv("VLLM_HPU_DSV41_N256_LOAD_WORKERS", "4")
    candidate = load_projection(shard, prefix, "cpu")
    for old, new in zip(reference, candidate, strict=True):
        assert torch.equal(old.view(torch.int16), new.view(torch.int16))
    assert reference[0].dsv41_sat_eligible == candidate[0].dsv41_sat_eligible


def test_memory_bound_limits_concurrent_preparation():
    q = SimpleNamespace(nbytes=64 << 20, shape=(1, 256, 65536))
    scales = SimpleNamespace(nbytes=8 << 20, shape=(1, 256, 8192))
    assert preparation_workers(q, scales, 4, 128 << 20) == 1
    for requested in (0, 5):
        with pytest.raises(ValueError, match="one to four"):
            preparation_workers(q, scales, requested, 0)
    with pytest.raises(ValueError, match="staging"):
        preparation_workers(q, scales, 1, (128 << 20) + 1)


def test_failed_expert_does_not_publish_a_device_batch(tmp_path):
    shard, prefix = sources(tmp_path)
    q, scales = shard.catalog[prefix + "_q16"], shard.catalog[prefix + "_s16"]
    scales.file.write_bytes(scales.file.read_bytes()[:scales.nbytes // 3])
    with pytest.raises(ValueError, match="Truncated"):
        list(prepare_expert_batch(shard, q, scales, 0, 3, compact_scales=True, active_k=128, workers=4))
