# SPDX-License-Identifier: Apache-2.0
"""Use the qualified integer D2H producer without enabling TP2 collectives."""
import os
from pathlib import Path


def _control_bridge():
    from vllm_gaudi.distributed.tp2_fused_ar_norm import _load_bridge, _verify_prepared_runtime

    configured = os.environ.get("VLLM_HPU_TP2_FUSED_AR_NORM_BRIDGE")
    if not configured:
        raise RuntimeError("Prepared TP4 token readback requires the configured native bridge artifact")
    path = Path(configured).expanduser().resolve()
    _verify_prepared_runtime(path)
    bridge = _load_bridge(path)
    if not hasattr(bridge, "copy_sampled_tokens_to_host"):
        raise RuntimeError("Rebuild the native bridge with producer-ordered integer readback")
    return bridge


def prepare_token_readback():
    bridge = _control_bridge()
    return bridge.copy_sampled_tokens_to_host


def resolve_device_runtime(tensor_parallel_size):
    """Resolve compute-only helpers without entering the TP2 peer protocol."""
    if tensor_parallel_size == 2:
        from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
        bridge, backend, _ = _resolve_runtime()
        return bridge, backend
    if tensor_parallel_size != 4:
        raise ValueError("Device continuation requires TP2 or TP4")
    import torch
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    if group.world_size != 4:
        raise RuntimeError("Device continuation TP4 communicator is not initialized")
    return _control_bridge(), group.device_group._get_backend(torch.device("hpu"))


def prepare_decode_control(input_views, position_views, position_bank=None):
    """Bind persistent model inputs once; retain normal Bridge dependencies."""
    bridge = _control_bridge()
    if getattr(bridge, "prepared_control_input_version", None) != 1:
        raise RuntimeError("Rebuild the native bridge with prepared TP4 control input version 1")
    frames = {count: bridge.PreparedControlInputs(input_views[count], position_views[count])
              for count in range(1, 7)}
    if position_bank is not None:
        if getattr(bridge, "prepared_position_bank_version", None) != 3:
            raise RuntimeError("Rebuild the native bridge with prepared position bank version 3")
        position_bank.prepare_native_copies(position_views, frames)
    return frames, bridge.copy_sampled_tokens_to_host
