# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_gaudi.omni.attention import HPU_SDPA_SOFTMAX_MODE, HPUSDPAImpl, HPUSDPABackend
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    PackedPaddingMetadata,
)


def test_hpu_sdpa_advertises_mask_free_suffix_padding():
    assert HPUSDPABackend.supports_packed_mask_free()
    assert HPUSDPABackend.supports_prefix_kv_slicing
    assert not HPUSDPABackend.supports_multi_doc_packed_varlen()
    assert HPU_SDPA_SOFTMAX_MODE == "None"


def test_hpu_sdpa_rejects_host_tensors():
    impl = HPUSDPAImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    value = torch.randn(1, 8, 2, 4)

    try:
        impl.forward(value, value, value)
    except RuntimeError as exc:
        assert "implicit host fallback is disabled" in str(exc)
    else:
        raise AssertionError("CPU inputs unexpectedly reached HPU attention")


def test_packed_padding_metadata_contract():
    cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
    metadata = AttentionMetadata(packed_padding=PackedPaddingMetadata(
        q_length=5,
        kv_length=5,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
    ))

    assert metadata.packed_padding.q_length == 5
    assert metadata.packed_padding.kv_length == 5
