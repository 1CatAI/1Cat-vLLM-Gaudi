# SPDX-License-Identifier: Apache-2.0
"""Generation-owned, bit-preserving C1 pipeline transport storage."""

import torch


class PackedC1Buffers:
    """Join BF16 residual and FP32 pre-mix without a dtype conversion.

    The caller completes the previous token before acquiring another packet.
    HCCL transports int32 words; typed views retain the original bit patterns.
    """

    def __init__(self, device, *, native_copy=False):
        self.native_copy = native_copy
        self.packets = tuple(torch.empty(10244, dtype=torch.int32, device=device) for _ in range(2))
        self.hidden = tuple(packet[:10240].view(torch.bfloat16) for packet in self.packets)
        self.pre = tuple(packet[10240:].view(torch.float32) for packet in self.packets)
        self.values = tuple({"hidden_states": hidden.view(1, 4, 5120), "pre_mix": pre.view(1, 4)}
                            for hidden, pre in zip(self.hidden, self.pre, strict=True))
        self.generation = 0
        self.completed = [0, 0]
        self.owners = [0, 0]
        self.active = None

    def acquire(self):
        if self.active is not None:
            raise RuntimeError("Previous C1 PP consumer has not completed")
        generation = self.generation + 1
        slot = (generation - 1) % 2
        if self.owners[slot] != self.completed[slot]:
            raise RuntimeError("C1 PP packet still belongs to an old consumer")
        self.generation = self.owners[slot] = generation
        self.active = (slot, generation)
        return self.packets[slot], self.values[slot]

    def pack(self, values):
        if self.active is None:
            raise RuntimeError("C1 PP packet was not acquired")
        hidden, pre = values["hidden_states"], values["pre_mix"]
        if hidden.shape != (1, 4, 5120) or hidden.dtype != torch.bfloat16:
            raise ValueError("C1 PP hidden tensor contract changed")
        if pre.shape != (1, 4) or pre.dtype != torch.float32:
            raise ValueError("C1 PP pre-mix tensor contract changed")
        slot, _ = self.active
        if self.native_copy:
            from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime
            bridge, backend, _ = _resolve_runtime()
            bridge.copy_c1_pipeline_tensors(backend, [hidden, pre], [self.hidden[slot], self.pre[slot]])
            return
        # A one-dimensional destination cannot inherit a producer's nontrivial
        # physical permutation. HCCL sees the same linear storage as both views.
        self.hidden[slot].copy_(hidden.reshape(-1))
        self.pre[slot].copy_(pre.reshape(-1))

    def complete(self):
        # The worker calls this only after its existing completion record has
        # reached the host, or after the startup-only device synchronization.
        if self.active is not None:
            slot, generation = self.active
            if generation != self.owners[slot]:
                raise RuntimeError("Stale C1 PP packet generation")
            self.completed[slot] = generation
            self.active = None
