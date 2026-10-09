# SPDX-License-Identifier: Apache-2.0
"""Check the integer device preparation contract with a portable CPU oracle."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from vllm_gaudi.ops.deepseek_v41_device_prepare import prepare_expert_device, read_source_batch
from vllm_gaudi.ops.deepseek_v41_expert_n256 import prepare_expert, saturated_decode_eligible


@pytest.mark.parametrize("compact", [False, True])
def test_batched_compressed_bytes_channels_and_padding(compact):
    random = np.random.default_rng(1051)
    q = random.integers(-32768, 32768, (3, 4, 256 * 32), dtype=np.int16)
    q[0] = 0
    q[1, :, 128 * 32:] = 0
    s = np.full((3, 4, 256 * 4), 127 << 7, dtype=np.uint16)
    packed, planes, channels, checks = prepare_expert_device(torch.from_numpy(q),
                                                             torch.from_numpy(s.view(np.int16)),
                                                             compact_scales=compact,
                                                             active_k=128)
    for index in range(3):
        reference = prepare_expert(q[index], s[index], compact_scales=compact)
        assert np.array_equal(packed[index].numpy(), reference[0])
        assert np.array_equal(planes[index].numpy(), reference[1])
        assert np.array_equal(channels[index].numpy().view(np.uint16), reference[2])
        eligible = not np.any(reference[0][:, 128 * 64:]) and saturated_decode_eligible(reference[1], active_k=128)
        assert checks[index].tolist() == [1, int(eligible)]


@pytest.mark.parametrize("invalid", [(127 << 7) | 0x8000, 0, 255 << 7, (127 << 7) | 1])
def test_invalid_scale_certificate_prevents_publication(invalid):
    q = torch.full((1, 2, 128 * 32), 0x7777, dtype=torch.int16)
    s = np.full((1, 2, 128 * 4), invalid, dtype=np.uint16)
    result = prepare_expert_device(q, torch.from_numpy(s.view(np.int16)), compact_scales=True, active_k=128)
    assert result[3][0, 0].item() == 0


def test_source_batch_offset_and_truncation(tmp_path):
    value = np.arange(24 * 2 * 16, dtype=np.int16).reshape(24, 2, 16)
    path = tmp_path / "source"
    path.write_bytes(b"header" + value.tobytes())
    source = SimpleNamespace(file=path, offset=6, shape=value.shape, nbytes=value.nbytes, dtype="I16")
    assert np.array_equal(read_source_batch(source, 7, 23), value[7:23])
    with pytest.raises(ValueError, match="bounded"):
        read_source_batch(source, 0, 17)
    path.write_bytes(b"header" + value[:20].tobytes())
    with pytest.raises(ValueError, match="Truncated"):
        read_source_batch(source, 19, 24)
