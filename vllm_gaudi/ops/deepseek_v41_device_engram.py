# SPDX-License-Identifier: Apache-2.0
"""Reuse the C1 mapped Engram producer for accepted-prefix C1-C6 inputs."""

import torch

from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
from vllm_gaudi.ops.deepseek_v41_host import device_engram_parameters


class DeviceEngramRounds:
    """Fixed producer views and histories for a single admitted request.

    The existing producer consumes one token and emits one lookback. Chaining
    those same launches keeps all six speculative lookbacks on device; the
    commit count selects the correct one before the next round. Both Engram
    layers use the same compressed history and their own checkpoint/hash
    constants. No peer/replay implementation or producer ABI is changed.
    """

    def __init__(self, host, ids, history):
        if ids.ndim != 1 or not 1 <= ids.numel() <= 6 or ids.device.type != "hpu" or history.shape != (3,):
            raise ValueError("Device Engram rounds require fixed C1-C6 input and three-token history")
        bridge, backend = resolve_device_runtime(host.tensor_parallel_size)
        if getattr(bridge, "device_engram_shared_mapping_version", 0) != 1:
            raise RuntimeError("Device Engram rounds require the shared checkpoint mapping ABI")
        device = ids.device
        self.count = ids.numel()
        self.ids = torch.empty(self.count, dtype=torch.int32, device=device)
        self.histories = torch.empty((self.count + 1, 3), dtype=torch.int32, device=device)
        self.scratch = torch.empty_like(history)
        self.token_views = tuple(self.ids[index:index + 1] for index in range(self.count))
        self.history_views = tuple(self.histories[index] for index in range(self.count + 1))
        self.producers, self.rows, self.row_views = [], [], []
        token_map = torch.from_numpy(host.history.token_map.astype("int32")).to(device)
        for layer in host.layout.layer_ids:
            shard = host.shards[layer]
            heads = shard["head_stop"] - shard["head_start"]
            weight, scale = host.table_sources[layer]
            shared = bool(weight.get("shared_memfd"))
            if shared != bool(scale.get("shared_memfd")):
                raise RuntimeError("Engram weight and scale mapping ownership disagree")
            parameters = torch.tensor(device_engram_parameters(host.layout, layer, host.tp_rank,
                                       host.history.pad_id, host.tensor_parallel_size),
                                      dtype=torch.int32, device=device)
            table_rows = shard["row_stop"] - shard["row_start"]
            producer = bridge.DeviceEngramProducer(
                backend, weight["file"], int(weight["shard_offset"]), table_rows * host.layout.head_dim,
                scale["file"], int(scale["shard_offset"]), table_rows * (host.layout.head_dim // 32),
                table_rows, token_map, parameters, shared_checkpoint=shared, local_heads=heads)
            rows = torch.empty((self.count, heads, host.layout.head_dim), dtype=torch.bfloat16, device=device)
            self.producers.append(producer)
            self.rows.append(rows)
            self.row_views.append(tuple(rows[index:index + 1] for index in range(self.count)))
        self.prepare_inputs = torch.compile(self._copy_inputs, backend="hpu_backend", fullgraph=True, dynamic=False)
        self.owner = None
        self.calls = 0
        self.reference = torch.compile(self._stage_reference, backend="hpu_backend", fullgraph=True, dynamic=False)

    def _stage_reference(self, sources):
        from vllm_gaudi.ops.deepseek_v41_math import unpack_swa

        for source, destination in zip(sources, self.rows, strict=True):
            destination.copy_(source if source.dtype == torch.bfloat16 else unpack_swa(source, 256))
        return tuple(self.rows)

    def stage_reference(self, sources):
        """Warm ordinary prompt/geometry entries with these same fixed roots."""
        if len(sources) != len(self.rows):
            raise ValueError("Device Engram reference lacks a layer input")
        return self.reference(sources)

    def _copy_inputs(self, ids, history):
        self.ids.copy_(ids.to(torch.int32))
        self.histories[0].copy_(history)

    def prepare(self, owner, ids, history):
        if self.owner not in (None, owner):
            raise RuntimeError("Device Engram cursor belongs to another live request")
        self.owner = owner
        self.prepare_inputs(ids, history)
        for index in range(self.count):
            for layer_index, producer in enumerate(self.producers):
                # Only one producer owns the common history output. The other
                # layer computes identical compression into a throwaway sink.
                next_history = self.history_views[index + 1] if layer_index == 0 else self.scratch
                producer.launch(self.token_views[index], self.history_views[index], next_history,
                                self.row_views[layer_index][index])
        self.calls += 1
        return tuple(self.rows)

    def retire(self, owner):
        if self.owner != owner:
            raise RuntimeError("Cannot retire another request's device Engram history")
        self.owner = None

    def close(self):
        for producer in self.producers:
            producer.close()
        # Release communicator references before distributed/device shutdown,
        # rather than deferring their destruction to Python's finalizer order.
        self.producers.clear()
