# SPDX-License-Identifier: Apache-2.0
"""Engram mmap, native row gather and HPU staging with completion ownership."""

from dataclasses import dataclass
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from vllm_gaudi.ops.deepseek_v41_engram import (
    EngramHashLayout, EngramTokenHistory, build_compressed_token_map,
)
from vllm_gaudi.ops.deepseek_v41_weights import file_hash, publish_json


@dataclass(frozen=True)
class EngramTransfer:
    generation: int
    slot: int
    batch: object
    buffers: tuple[torch.Tensor, ...]


class _TransferSlot:
    def __init__(self, max_tokens, heads, width, device):
        from vllm_gaudi.lib.dsv41_host_gather import GatherSlot
        self.gather = GatherSlot(max_tokens * heads, width)
        self.host = torch.empty((max_tokens, heads, width + width // 32), dtype=torch.uint8,
                                device="cpu").pin_memory("hpu")
        if not self.host.is_pinned("hpu"):
            raise RuntimeError("The HPU runtime did not provide pinned Engram DMA staging")
        self.device = torch.empty_like(self.host, device=device)
        self.dma_done, self.consumer_done = torch.hpu.Event(), torch.hpu.Event()
        self.inflight = False
        self.generation = 0
        self.weight_view = torch.from_numpy(self.gather.weights)
        self.scale_view = torch.from_numpy(self.gather.scales)
        # C1 uses fixed views of the existing pinned allocation. NumPy copies
        # preserve the raw weight/scale bytes without invoking tensor dispatch.
        self.decode_host, self.decode_device = self.host[:1], self.device[:1]
        packed = self.decode_host.numpy()
        self.decode_weights, self.decode_scales = packed[:, :, :width], packed[:, :, width:]
        self.decode_gather_weights = self.gather.weights[:heads].reshape(1, heads, width)
        self.decode_gather_scales = self.gather.scales[:heads].reshape(1, heads, width // 32)

    def fill(self, count):
        if count == 1:
            np.copyto(self.decode_weights, self.decode_gather_weights)
            np.copyto(self.decode_scales, self.decode_gather_scales)
            return self.decode_host, self.decode_device
        heads, width = self.host.shape[1], self.weight_view.shape[1]
        rows = count * heads
        host = self.host[:count]
        host[:, :, :width].copy_(self.weight_view[:rows].reshape(count, heads, width))
        host[:, :, width:].copy_(self.scale_view[:rows].reshape(count, heads, width // 32))
        return host, self.device[:count]

    def reuse(self):
        if self.inflight:
            self.consumer_done.synchronize()
            self.dma_done.synchronize()
        self.inflight = False


class EngramHost:
    def __init__(self, directory, tp_rank, device, *, max_tokens=512, ring_size=3,
                 tokenizer=None, checkpoint_audit=None, force_lock=False):
        from vllm_gaudi.lib import dsv41_host_gather as native
        if native.abi_version != 1 or torch.device(device).type != "hpu":
            raise RuntimeError("V4.1 Engram requires its native host gather and an HPU DMA runtime")
        if max_tokens not in range(1, 513) or ring_size < 2:
            raise ValueError("Invalid bounded Engram staging geometry")
        self.directory, self.tp_rank = Path(directory), tp_rank
        manifest = json.loads((self.directory / "manifest.json").read_text())
        record = manifest["engram_host_shards"][str(tp_rank)]
        path = self.directory / record["file"]
        if file_hash(path) != record["sha256"]:
            raise RuntimeError("Engram host shard manifest changed")
        host = json.loads(path.read_text())
        if (host["tp_rank"] != tp_rank or host["pp_owner"] != 0 or host["sharding"] != "complete_hash_heads"
                or host["model_revision"] != manifest["model_revision"] or not host["shared_read_only"]):
            raise RuntimeError("Engram host shard ownership or revision mismatch")
        config = json.loads((self.directory / "config.json").read_text())["text_config"]
        self.layout = EngramHashLayout.from_config(config)
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True, trust_remote_code=False)
        token_map, compressed = build_compressed_token_map(tokenizer)
        if compressed != self.layout.vocab_size or len(token_map) != config["vocab_size"]:
            raise RuntimeError("Engram tokenizer compression differs from the checkpoint")
        self.history = EngramTokenHistory(self.layout, token_map)
        self.tables, self.shards, self.slots = {}, {}, {}
        self.stream = torch.hpu.Stream()
        self.generation, self.pending, self.closed = 0, None, False
        self.max_tokens, self.ring_size = max_tokens, ring_size
        self.audit = {"gathers": 0, "major_faults": 0, "dma_bytes": 0, "generations": 0}
        for layer in self.layout.layer_ids:
            weight = host["tables"][f"layers.{layer}.engram.embed.weight"]
            scale = host["tables"][f"layers.{layer}.engram.embed.scale"]
            shard = self.layout.head_shard(layer, tp_rank)
            for item in (weight, scale):
                if any(item.get(key) != value for key, value in shard.items()):
                    raise RuntimeError("Engram offsets do not match the frozen hash-head layout")
                self._verify_source(item, checkpoint_audit)
            self.shards[layer] = shard
            self.tables[layer] = native.HostRows(weight["file"], weight["shard_offset"],
                scale["file"], scale["shard_offset"], shard["row_start"], shard["row_stop"],
                self.layout.head_dim, True, force_lock)
            self.slots[layer] = [_TransferSlot(max_tokens, shard["head_stop"] - shard["head_start"],
                                              self.layout.head_dim, device) for _ in range(ring_size)]

    def _verify_source(self, item, checkpoint_audit):
        path = Path(item["file"])
        stat = path.stat()
        identity = {"file": str(path), "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
                    "bytes": stat.st_size, "sha256": item["source_sha256"]}
        cache_path = self.directory / (path.name + ".host-identity.json")
        with (self.directory / (path.name + ".host-identity.lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache_path.exists() and json.loads(cache_path.read_text()) == identity:
                return
            audited = False
            if checkpoint_audit is not None:
                record_path = Path(checkpoint_audit) / "files" / (path.name + ".json")
                record = json.loads(record_path.read_text())
                audited = all(record.get(key) == identity[key] for key in ("inode", "mtime_ns", "bytes", "sha256"))
            if not audited and file_hash(path) != item["source_sha256"]:
                raise RuntimeError("Engram source file differs from the frozen checkpoint")
            publish_json(cache_path, identity)

    def reset(self, request_id):
        if self.pending is not None:
            raise RuntimeError("Cannot replace Engram history before completing or rejecting the pending verify")
        for slots in self.slots.values():
            for slot in slots:
                slot.reuse()
        self.history.reset(request_id)

    def prepare(self, request_id, token_ids, image_mask=None):
        if self.closed or self.pending is not None:
            raise RuntimeError("Engram is closed or its preceding input transaction is still pending")
        count = len(token_ids)
        if count == 0 or count > self.max_tokens:
            raise ValueError("Engram input exceeds fixed staging capacity")
        batch = self.history.prepare(request_id, token_ids, image_mask)
        self.generation += 1
        ring = (self.generation - 1) % self.ring_size
        buffers = []
        # Submission of layer 14 follows completion of layer 1's host lookup;
        # its worker then runs independently of the first layer's DMA.
        def submit(index):
            layer = self.layout.layer_ids[index]
            slot, shard = self.slots[layer][ring], self.shards[layer]
            slot.reuse()
            ids = np.ascontiguousarray(batch.hash_ids[:, index, shard["head_start"]:shard["head_stop"]])
            slot.generation = slot.gather.submit(self.tables[layer], ids)
        submit(0)
        consumer_stream = torch.hpu.current_stream()
        # A C1 layer uploads only a few KiB and is immediately consumed on
        # the caller's stream. Queue order replaces a cross-stream handoff.
        transfer_stream = consumer_stream if count == 1 else self.stream
        with nullcontext() if count == 1 else torch.hpu.stream(transfer_stream):
            for index, layer in enumerate(self.layout.layer_ids):
                slot = self.slots[layer][ring]
                slot.gather.wait(slot.generation)
                if index + 1 < len(self.layout.layer_ids):
                    submit(index + 1)
                host, device = slot.fill(count)
                self.audit["major_faults"] += slot.gather.major_faults
                self.audit["gathers"] += 1
                slot.gather.release(slot.generation)
                device.copy_(host, non_blocking=True)
                slot.dma_done.record(transfer_stream)
                buffers.append(device)
                self.audit["dma_bytes"] += host.numel()
        # Both layer DMAs are ordered on this stream. The last event covers
        # every input; retain individual events for each slot's reuse guard.
        if count != 1:
            consumer_stream.wait_event(slot.dma_done)
        ticket = EngramTransfer(self.generation, ring, batch, tuple(buffers))
        self.pending = ticket
        self.audit["generations"] += 1
        return ticket

    def complete(self, ticket, committed_inputs):
        if self.pending is not ticket or ticket.generation != self.generation:
            raise RuntimeError("Stale Engram transfer/verify completion")
        consumer_stream = torch.hpu.current_stream()
        for layer in self.layout.layer_ids:
            slot = self.slots[layer][ticket.slot]
            slot.consumer_done.record(consumer_stream)
            slot.inflight = True
        self.history.commit(ticket.batch, committed_inputs)
        self.pending = None

    def close(self):
        if self.closed:
            return
        # Exceptions after state writes never invoke another model path. Drain
        # this owner's transfers/consumers before releasing mmap and staging.
        torch.hpu.synchronize()
        if self.pending is not None:
            self.history.discard(self.pending.batch)
            self.pending = None
        self.slots.clear()
        self.tables.clear()
        self.closed = True
