# SPDX-License-Identifier: Apache-2.0
"""Engram mmap, native row gather and HPU staging with completion ownership."""

from dataclasses import dataclass
from functools import cache
import fcntl
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

from vllm_gaudi import envs as gaudi_envs
from vllm_gaudi.ops.deepseek_v41_diagnostics import trace_phase
from vllm_gaudi.ops.deepseek_v41_engram import (
    EngramHashLayout,
    EngramTokenHistory,
    build_compressed_token_map,
)
from vllm_gaudi.ops.deepseek_v41_weights import file_hash, publish_json

envs = gaudi_envs


def _signed_i32_bits(value):
    value = int(value) & 0xffffffff
    return value if value < 0x80000000 else value - 0x100000000


def resident_table_source(item, binding):
    """Validate a service-owned shared backing against its checkpoint extent."""
    source = dict(file=item["file"], offset=item["shard_offset"], length=item["shard_bytes"])
    if binding.get("source") != source or binding.get("backing") != "shared_memfd":
        raise RuntimeError("Engram resident binding differs from the checkpoint extent")
    stat = Path(source["file"]).stat()
    identity = dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if binding.get("source_identity") != identity:
        raise RuntimeError("Engram source changed after resident preparation")
    descriptor = os.open(binding["file"], os.O_RDONLY | os.O_CLOEXEC)
    try:
        target = os.fstat(descriptor)
        seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
        required = fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        if ((target.st_dev, target.st_ino, target.st_size) != (binding["device"], binding["inode"], source["length"])
                or seals & required != required or binding["offset"] != 0 or binding["length"] != source["length"]):
            raise RuntimeError("Engram resident backing identity or extent changed")
    finally:
        os.close(descriptor)
    return dict(item, file=binding["file"], shard_offset=0, shared_memfd=True)


def device_engram_parameters(layout, layer, tp_rank, pad_id):
    """Pack the split-u64 hash constants consumed by the device producer."""
    layer_index = layout.layer_ids.index(layer)
    shard = layout.head_shard(layer, tp_rank)
    result = [int(pad_id), shard["head_start"], shard["row_start"]]
    for multiplier in layout.multipliers[layer_index]:
        bits = int(multiplier) & 0xffffffffffffffff
        result.extend((_signed_i32_bits(bits), _signed_i32_bits(bits >> 32)))
    first, last = shard["head_start"], shard["head_stop"]
    primes = [int(value) for value in layout.primes[layer_index, first:last]]
    result.extend(primes)
    result.extend(int(value) for value in layout.offsets[layer_index, first:last])
    result.extend(pow(2, 64, prime) for prime in primes)
    if len(result) != 47:
        raise RuntimeError("Device Engram parameter ABI requires exactly 47 I32 values")
    return result


@cache
def _configured_host_native(directory):
    root = Path(directory).resolve()
    libraries = list(root.glob("dsv41_host_gather*.so"))
    manifest = json.loads((root / "deepseek_v41_build.json").read_text())
    if (len(libraries) != 1 or file_hash(libraries[0]) != manifest["binaries"].get(libraries[0].name)):
        raise RuntimeError("Engram host binary differs from its configured build manifest")
    spec = importlib.util.spec_from_file_location("dsv41_host_gather", libraries[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.abi_version != manifest["host_gather_abi_version"]
            or module.c1_abi_version != manifest["host_c1_abi_version"]):
        raise RuntimeError("Engram host ABI differs from its configured build manifest")
    return module


def host_native():
    directory = gaudi_envs.VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR
    if directory:
        return _configured_host_native(directory)
    from vllm_gaudi.lib import dsv41_host_gather
    return dsv41_host_gather


@dataclass(frozen=True)
class EngramTransfer:
    generation: int
    slot: int
    batch: object
    buffers: tuple[torch.Tensor, ...]
    packet: bool = False


class _PrefillRows:
    """Queue each layer's DMA dependency at its first ordinary consumer."""

    def __init__(self, owner, ticket):
        self.owner, self.ticket = owner, ticket

    def __len__(self):
        return len(self.ticket.buffers)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.owner.wait_layer(self.ticket, index)


class _C1Packet:
    """One ownership event covers upload and both Engram consumers."""

    def __init__(self, heads, width, device, *, destination=None):
        stride = width + width // 32
        self.host = torch.empty(sum(heads) * stride, dtype=torch.uint8, device="cpu").pin_memory("hpu")
        if not self.host.is_pinned("hpu"):
            raise RuntimeError("The HPU runtime did not pin Engram packet staging")
        if destination is not None and (destination.shape != self.host.shape or destination.dtype != torch.uint8
                                        or not destination.is_contiguous() or destination.storage_offset() != 0
                                        or destination.device.type != torch.device(device).type):
            raise ValueError("Direct Engram destination must be one contiguous byte packet on the target device")
        self.device = torch.empty_like(self.host, device=device) if destination is None else destination
        host_views, device_views, start = [], [], 0
        for count in heads:
            stop = start + count * stride
            host_views.append(self.host[start:stop].view(1, count, stride))
            device_views.append(self.device[start:stop].view(1, count, stride))
            start = stop
        self.host_buffers = tuple(host_views)
        self.targets = [value.numpy() for value in host_views]
        self.buffers = tuple(device_views)
        self.consumer_done = torch.hpu.Event()
        self.inflight = False
        self.generation = self.pending_generation = 0

    @trace_phase
    def reuse(self):
        if self.pending_generation:
            raise RuntimeError("Cannot reuse a pending Engram packet")
        if self.inflight:
            self.consumer_done.synchronize()
        self.inflight = False

    def upload(self, generation, *, late_only=False):
        if self.inflight or self.pending_generation or generation <= self.generation:
            raise RuntimeError("Stale or occupied Engram packet generation")
        self.pending_generation = generation
        if late_only:
            self.buffers[1].copy_(self.host_buffers[1], non_blocking=True)
        else:
            self.device.copy_(self.host, non_blocking=True)

    def complete(self, generation, stream):
        if self.pending_generation != generation:
            raise RuntimeError("Stale Engram packet completion")
        self.consumer_done.record(stream)
        self.inflight = True
        self.generation, self.pending_generation = generation, 0


class _TransferSlot:

    def __init__(self, max_tokens, heads, width, device):
        native = host_native()
        self.gather = native.GatherSlot(max_tokens * heads, width)
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
        self.decode_host, self.decode_device = self.host[:1], self.device[:1]
        packed = self.decode_host.numpy()
        self.decode_weights = packed[:, :, :width]
        self.decode_scales = packed[:, :, width:]
        self.decode_gather_weights = self.gather.weights[:heads].reshape(1, heads, width)
        self.decode_gather_scales = self.gather.scales[:heads].reshape(1, heads, width // 32)
        self.direct = envs.VLLM_HPU_DSV41_FUSED_STAGE_IO
        if self.direct:
            if getattr(native, "packed_output_version", None) != 1:
                raise RuntimeError("Fused Engram staging requires native packed-output capability version 1")
            self.gather.bind_packed_output(self.host.reshape(-1, width + width // 32).numpy())

    @trace_phase
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


class _TransferBatch:
    """One allocation and completion owner for the two independent gathers."""

    def __init__(self, slots, device):
        if not all(slot.direct for slot in slots):
            raise RuntimeError("Batched Engram DMA requires direct native packed staging")
        self.host = torch.zeros(sum(slot.host.numel() for slot in slots), dtype=torch.uint8,
                                device="cpu").pin_memory("hpu")
        if not self.host.is_pinned("hpu"):
            raise RuntimeError("The HPU runtime did not pin batched Engram staging")
        self.device = torch.empty_like(self.host, device=device)
        self.dma_done, self.consumer_done = torch.hpu.Event(), torch.hpu.Event()
        self.inflight, self.generation = False, 0
        offset = 0
        for slot in slots:
            shape, size = slot.host.shape, slot.host.numel()
            slot.host = self.host[offset:offset + size].view(shape)
            slot.device = self.device[offset:offset + size].view(shape)
            slot.gather.bind_packed_output(slot.host.reshape(-1, shape[-1]).numpy())
            offset += size

    def reuse(self):
        if self.inflight:
            self.consumer_done.synchronize()
            self.dma_done.synchronize()
        self.inflight = False

    def stage(self, stream, generation):
        if self.inflight or self.generation != generation:
            raise RuntimeError("Stale or already consumed Engram DMA batch")
        with torch.hpu.stream(stream):
            self.device.copy_(self.host, non_blocking=True)
            self.dma_done.record(stream)
        torch.hpu.current_stream().wait_event(self.dma_done)

    def complete(self, generation):
        if self.inflight or self.generation != generation:
            raise RuntimeError("Stale or repeated Engram batch completion")
        self.consumer_done.record(torch.hpu.current_stream())
        self.inflight = True


class EngramHost:

    def __init__(self,
                 directory,
                 tp_rank,
                 device,
                 *,
                 max_tokens=512,
                 ring_size=3,
                 tokenizer=None,
                 checkpoint_audit=None,
                 force_lock=False,
                 resident_tables=None):
        native = host_native()
        if native.abi_version != 1 or torch.device(device).type != "hpu":
            raise RuntimeError("V4.1 Engram requires its native host gather and an HPU DMA runtime")
        self.native_c1 = None
        self.device_c1 = None
        self.device_rows = None
        self.device_history = None
        self.device_history_parity = 0
        self.device_request = None
        self.device_position = None
        self.device_pending = None
        self.c1_packets = None
        if (gaudi_envs.VLLM_HPU_DSV41_ENGRAM_C1_PACKET and not gaudi_envs.VLLM_HPU_DSV41_ENGRAM_NATIVE_C1):
            raise RuntimeError("Engram C1 packets require native C1 preparation")
        if (gaudi_envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT
                and not (gaudi_envs.VLLM_HPU_DSV41_ENGRAM_C1_PACKET and gaudi_envs.VLLM_HPU_DSV41_NATIVE_INPUT_GRAPH
                         and gaudi_envs.VLLM_HPU_DSV41_GRAPH_REPLAY)):
            raise RuntimeError("Direct Engram inputs require C1 packets and native input graph replay")
        self.c1_abi = getattr(native, "c1_abi_version", None)
        if gaudi_envs.VLLM_HPU_DSV41_ENGRAM_NATIVE_C1 and self.c1_abi not in (1, 2):
            raise RuntimeError("The enabled Engram C1 path requires its matching native preparation ABI")
        if gaudi_envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM and self.c1_abi != 2:
            raise RuntimeError("Device Engram requires native host C1 ABI 2")
        if max_tokens not in range(1, 8193) or ring_size < 2:
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
        self.histories = {}
        self.tables, self.shards, self.slots = {}, {}, {}
        self.table_sources = {}
        self.table_page_counts = {}
        self.stream = torch.hpu.Stream()
        self.generation, self.pending, self.closed = 0, None, False
        self.ready_ticket = None
        self.max_tokens, self.ring_size = max_tokens, ring_size
        self.audit = {"gathers": 0, "major_faults": 0, "dma_bytes": 0, "generations": 0}
        self.profile_records = None
        for layer in self.layout.layer_ids:
            weight = host["tables"][f"layers.{layer}.engram.embed.weight"]
            scale = host["tables"][f"layers.{layer}.engram.embed.scale"]
            shard = self.layout.head_shard(layer, tp_rank)
            for item in (weight, scale):
                if any(item.get(key) != value for key, value in shard.items()):
                    raise RuntimeError("Engram offsets do not match the frozen hash-head layout")
                self._verify_source(item, checkpoint_audit)
            bindings = (resident_tables or {}).get(str(tp_rank), {}).get(str(layer))
            if bindings is not None:
                if len(bindings) != 2:
                    raise RuntimeError("Engram resident binding must contain weight and scale")
                weight, scale = (resident_table_source(item, binding)
                                 for item, binding in zip((weight, scale), bindings, strict=True))
            self.table_sources[layer] = weight, scale
            self.shards[layer] = shard
            page = os.sysconf("SC_PAGESIZE")
            rows = shard["row_stop"] - shard["row_start"]
            self.table_page_counts[layer] = sum(
                (rows * width + item["shard_offset"] % page + page - 1) // page
                for item, width in ((weight, self.layout.head_dim), (scale, self.layout.head_dim // 32)))
            self.tables[layer] = native.HostRows(weight["file"], weight["shard_offset"], scale["file"],
                                                 scale["shard_offset"], shard["row_start"], shard["row_stop"],
                                                 self.layout.head_dim, True, force_lock)
            self.slots[layer] = [
                _TransferSlot(max_tokens, shard["head_stop"] - shard["head_start"], self.layout.head_dim, device)
                for _ in range(ring_size)
            ]
        if gaudi_envs.VLLM_HPU_DSV41_ENGRAM_NATIVE_C1:
            layers = self.layout.layer_ids
            if gaudi_envs.VLLM_HPU_DSV41_ENGRAM_C1_PACKET:
                heads = [self.shards[layer]["head_stop"] - self.shards[layer]["head_start"] for layer in layers]
                first = _C1Packet(heads, self.layout.head_dim, device)
                destination = first.device if gaudi_envs.VLLM_HPU_DSV41_ENGRAM_DIRECT_INPUT else None
                self.c1_packets = [first] + [
                    _C1Packet(heads, self.layout.head_dim, device, destination=destination)
                    for _ in range(ring_size - 1)
                ]
                targets = [packet.targets for packet in self.c1_packets]
                self.audit["c1_packet_bytes"] = sum(packet.host.numel() for packet in self.c1_packets)
                self.audit["direct_input"] = destination is not None
            else:
                targets = [[self.slots[layer][slot].decode_host.numpy() for layer in layers]
                           for slot in range(ring_size)]
            self.native_c1 = native.NativeC1Prepare(
                self.history.token_map, self.layout.multipliers, self.layout.primes, self.layout.offsets,
                np.array([self.shards[layer]["head_start"] for layer in layers], dtype=np.int64),
                np.array([self.shards[layer]["head_stop"] for layer in layers], dtype=np.int64), self.history.pad_id,
                [self.tables[layer] for layer in layers], targets)
        if gaudi_envs.VLLM_HPU_DSV41_V2_DEVICE_ENGRAM:
            self._initialize_device_c1(host, device)
        self.batches = None
        if envs.VLLM_HPU_DSV41_BATCHED_INPUT_STAGING and max_tokens == 6:
            if not envs.VLLM_HPU_DSV41_FUSED_STAGE_IO:
                raise RuntimeError("Batched Engram staging requires fused C6 stage input preparation")
            self.batches = [
                _TransferBatch([self.slots[layer][ring] for layer in self.layout.layer_ids], device)
                for ring in range(ring_size)
            ]

    def residency(self):
        return {
            str(layer): {
                "mapped_pages": self.table_page_counts[layer],
                "resident_pages": table.resident_pages()
            }
            for layer, table in self.tables.items()
        }

    def _initialize_device_c1(self, host, device):
        from vllm_gaudi.distributed.tp2_fused_ar_norm import _resolve_runtime

        layer = self.layout.layer_ids[0]
        if layer != 1:
            raise RuntimeError("Device Engram currently owns the layer-1 table only")
        shard = self.shards[layer]
        local_heads = shard["head_stop"] - shard["head_start"]
        if local_heads != 12 or self.layout.head_dim != 256:
            raise RuntimeError("Device Engram requires the qualified 12x256 TP-local geometry")
        weight, scale = self.table_sources[layer]
        rows = shard["row_stop"] - shard["row_start"]
        token_map = torch.from_numpy(self.history.token_map.astype(np.int32)).to(device)
        parameters = torch.tensor(device_engram_parameters(self.layout, layer, self.tp_rank, self.history.pad_id),
                                  dtype=torch.int32,
                                  device=device)
        self.device_rows = torch.empty((1, local_heads, self.layout.head_dim), dtype=torch.bfloat16, device=device)
        self.device_history = (
            torch.empty(3, dtype=torch.int32, device=device),
            torch.empty(3, dtype=torch.int32, device=device),
        )
        bridge, backend, _ = _resolve_runtime()
        if not hasattr(bridge, "DeviceEngramProducer"):
            raise RuntimeError("The TP2 bridge lacks the device Engram producer ABI")
        shared = bool(weight.get("shared_memfd"))
        if shared != bool(scale.get("shared_memfd")):
            raise RuntimeError("Device Engram weight and scale backing disagree")
        if shared and getattr(bridge, "device_engram_shared_mapping_version", 0) != 1:
            raise RuntimeError("The TP2 bridge cannot share the resident Engram backing")
        mapping_options = {"shared_checkpoint": True} if shared else {}
        self.device_c1 = bridge.DeviceEngramProducer(
            backend,
            weight["file"],
            int(weight["shard_offset"]),
            rows * self.layout.head_dim,
            scale["file"],
            int(scale["shard_offset"]),
            rows * (self.layout.head_dim // 32),
            rows,
            token_map,
            parameters,
            **mapping_options,
        )
        self.audit["device_c1_mapped_bytes"] = self.device_c1.mapped_bytes()
        self.audit["device_c1_workspace_bytes"] = self.device_c1.workspace_bytes()

    def prepare_device_c1(self, request_id, raw_token):
        if self.device_c1 is None or self.closed or self.pending is not None or self.device_pending is not None:
            raise RuntimeError("Device Engram preparation started outside an idle layer-1 transaction")
        if raw_token.device.type != "hpu" or raw_token.dtype != torch.int32 or raw_token.numel() != 1:
            raise ValueError("Device Engram requires one raw I32 HPU token")
        if self.device_request != request_id or self.device_position != self.history.position:
            suffix = np.full(3, -1, dtype=np.int32)
            committed = self.history.history[-3:][::-1]
            suffix[:len(committed)] = committed
            self.device_history[0].copy_(torch.from_numpy(suffix), non_blocking=True)
            self.device_history_parity = 0
            self.device_request = request_id
            self.device_position = self.history.position
            self.audit["device_c1_history_resets"] = self.audit.get("device_c1_history_resets", 0) + 1
        next_parity = self.device_history_parity ^ 1
        self.device_c1.launch(raw_token.reshape(-1), self.device_history[self.device_history_parity],
                              self.device_history[next_parity], self.device_rows)
        self.device_history_parity = next_parity
        self.device_position += 1
        self.device_pending = request_id
        self.audit["device_c1_launches"] = self.audit.get("device_c1_launches", 0) + 1
        return self.device_rows

    def stage_device_c1_reference(self, request_id, packed_rows):
        if self.device_c1 is None or self.device_pending is not None:
            raise RuntimeError("Device Engram warmup staging overlaps another layer-1 generation")
        from vllm_gaudi.ops.deepseek_v41_math import unpack_swa

        self.device_rows.copy_(unpack_swa(packed_rows, self.layout.head_dim))
        self.device_pending = request_id
        return self.device_rows

    def consume_device_c1(self, request_id):
        if self.device_c1 is None or self.device_pending != request_id:
            raise RuntimeError("No matching device Engram layer-1 generation is ready")
        self.device_pending = None
        return self.device_rows

    def _verify_source(self, item, checkpoint_audit):
        path = Path(item["file"])
        stat = path.stat()
        identity = {
            "file": str(path),
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "bytes": stat.st_size,
            "sha256": item["source_sha256"]
        }
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
        for packet in self.c1_packets or ():
            packet.reuse()
        for batch in getattr(self, "batches", None) or ():
            batch.reuse()
        self.history.reset(request_id)
        self.device_request = None
        self.device_position = None
        self.device_pending = None

    def activate(self, request_id, *, reset=False):
        if self.pending is not None:
            raise RuntimeError("Cannot switch Engram requests before completing verify")
        if request_id not in self.histories:
            history = EngramTokenHistory(self.layout, self.history.token_map)
            history.reset(request_id)
            self.histories[request_id] = history
        self.history = self.histories[request_id]
        if reset:
            self.reset(request_id)

    def release_request(self, request_id):
        self.histories.pop(request_id, None)

    def prepare(self, request_id, token_ids, image_mask=None, *, defer_wait=False, device_layer1=False):
        if self.closed or self.pending is not None:
            raise RuntimeError("Engram is closed or its preceding input transaction is still pending")
        count = len(token_ids)
        if count == 0 or count > self.max_tokens:
            raise ValueError("Engram input exceeds fixed staging capacity")
        if device_layer1 and (count != 1 or self.device_c1 is None or self.device_pending != request_id):
            raise RuntimeError("Layer-14-only preparation requires the matching device layer-1 generation")
        if count == 1 and self.native_c1 is not None:
            ticket = self._prepare_native_c1(request_id, token_ids, image_mask, device_layer1=device_layer1)
            if not defer_wait:
                self.wait(ticket)
            return ticket
        batch = self.history.prepare(request_id, token_ids, image_mask)
        self.generation += 1
        ring = (self.generation - 1) % self.ring_size
        if getattr(self, "batches", None) is not None:
            batch_owner = self.batches[ring]
            batch_owner.reuse()
            batch_owner.generation = self.generation
        buffers = tuple(self.slots[layer][ring].device[:count] for layer in self.layout.layer_ids)
        ticket = EngramTransfer(self.generation, ring, batch, buffers)
        self.pending = ticket
        self.ready_ticket = None
        self.ready_layers = 0
        self._submit(ticket, 0)
        if envs.VLLM_HPU_DSV41_FUSED_STAGE_IO:
            for index in range(1, len(self.layout.layer_ids)):
                self._submit(ticket, index)
        self.audit["generations"] += 1
        if not defer_wait:
            self.wait(ticket)
        return ticket

    def _prepare_native_c1(self, request_id, token_ids, image_mask, *, device_layer1=False):
        if image_mask is not None and len(image_mask) != 1:
            raise ValueError("Image span mask does not match C1 input")
        image = bool(image_mask[0]) if image_mask is not None else False
        self.generation += 1
        ring = (self.generation - 1) % self.ring_size
        packet = self.c1_packets[ring] if self.c1_packets is not None else None
        if packet is not None:
            packet.reuse()
        else:
            for layer in self.layout.layer_ids:
                self.slots[layer][ring].reuse()
        arguments = (request_id, int(token_ids[0]), image, self.native_c1, self.generation, ring)
        if self.c1_abi == 2:
            batch, faults = self.history.prepare_c1(*arguments, device_layer1)
        else:
            if device_layer1:
                raise RuntimeError("Layer-14-only preparation requires native C1 ABI 2")
            batch, faults = self.history.prepare_c1(*arguments)
        if packet is not None:
            packet.upload(self.generation, late_only=device_layer1)
            buffers = packet.buffers
            self.audit["dma_bytes"] += packet.host_buffers[1].numel() if device_layer1 else packet.host.numel()
            self.audit["c1_packets"] = self.audit.get("c1_packets", 0) + 1
        else:
            stream = torch.hpu.current_stream()
            buffers = []
            for layer in self.layout.layer_ids:
                slot = self.slots[layer][ring]
                slot.decode_device.copy_(slot.decode_host, non_blocking=True)
                slot.dma_done.record(stream)
                buffers.append(slot.decode_device)
                self.audit["dma_bytes"] += slot.decode_host.numel()
        self.audit["major_faults"] += faults
        self.audit["gathers"] += 1 if device_layer1 else len(self.layout.layer_ids)
        self.audit["generations"] += 1
        self.audit["native_c1"] = self.audit.get("native_c1", 0) + 1
        ticket = EngramTransfer(self.generation, ring, batch, tuple(buffers), packet is not None)
        self.pending = ticket
        self.ready_ticket = None
        return ticket

    def _submit(self, ticket, index):
        layer = self.layout.layer_ids[index]
        slot, shard = self.slots[layer][ticket.slot], self.shards[layer]
        slot.reuse()
        ids = np.ascontiguousarray(ticket.batch.hash_ids[:, index, shard["head_start"]:shard["head_stop"]])
        slot.generation = slot.gather.submit(self.tables[layer], ids)

    def wait(self, ticket):
        """Bind both DMA completions after independent embedding submission.

        No graph may read the ticket until this queues its stream waits. The
        generation continues to own the slot until verify commits or closes it.
        """
        if self.pending is not ticket or ticket.generation != self.generation or self.ready_ticket is ticket:
            raise RuntimeError("Stale or already completed Engram preparation")
        if (ticket.packet or (ticket.batch.hash_ids.shape[0] == 1 and self.native_c1 is not None)):
            self.ready_ticket = ticket
            return ticket.buffers
        self._stage_layers(ticket, len(self.layout.layer_ids))
        return ticket.buffers

    def prefill_rows(self, ticket):
        """Defer prefill waits while retaining the existing transaction owner."""
        if (self.pending is not ticket or ticket.generation != self.generation or self.ready_ticket is ticket
                or ticket.batch.hash_ids.shape[0] <= 6):
            raise RuntimeError("Deferred Engram rows require a pending prefill transaction")
        return _PrefillRows(self, ticket)

    def wait_layer(self, ticket, index):
        if self.pending is not ticket or ticket.generation != self.generation:
            raise RuntimeError("Stale Engram layer consumption")
        if not 0 <= index < len(self.layout.layer_ids):
            raise IndexError(index)
        self._stage_layers(ticket, index + 1)
        return ticket.buffers[index]

    def _stage_layers(self, ticket, stop):
        count, ring = ticket.batch.hash_ids.shape[0], ticket.slot
        batch_owner = self.batches[ring] if getattr(self, "batches", None) is not None else None
        if batch_owner is not None:
            stop = len(self.layout.layer_ids)
        # Submission of layer 14 follows completion of layer 1's host lookup;
        # its worker then runs independently of the first layer's DMA.
        for index in range(self.ready_layers, stop):
            layer = self.layout.layer_ids[index]
            slot = self.slots[layer][ring]
            slot.gather.wait(slot.generation)
            if not slot.direct and index + 1 < len(self.layout.layer_ids):
                self._submit(ticket, index + 1)
            heads = slot.host.shape[1]
            rows, width = count * heads, self.layout.head_dim
            if not slot.direct:
                slot.host[:count, :, :width].copy_(slot.weight_view[:rows].reshape(count, heads, width))
                slot.host[:count, :, width:].copy_(slot.scale_view[:rows].reshape(count, heads, width // 32))
            self.audit["major_faults"] += slot.gather.major_faults
            self.audit["gathers"] += 1
            if self.profile_records is not None:
                generation, started, finished, row_count, tid = slot.gather.timing
                if generation != slot.generation or finished < started or len(self.profile_records) >= 65536:
                    raise RuntimeError("Invalid or overflowing Engram profiling record")
                self.profile_records.append({
                    "generation": self.generation,
                    "slot_generation": generation,
                    "layer": layer,
                    "ring": ring,
                    "start_unix_ns": started,
                    "end_unix_ns": finished,
                    "rows": row_count,
                    "worker_tid": tid,
                    "major_faults": slot.gather.major_faults
                })
            slot.gather.release(slot.generation)
            if batch_owner is None:
                with torch.hpu.stream(self.stream):
                    slot.device[:count].copy_(slot.host[:count], non_blocking=True)
                    slot.dma_done.record(self.stream)
                torch.hpu.current_stream().wait_event(slot.dma_done)
                self.audit["dma_bytes"] += count * heads * (width + width // 32)
            self.ready_layers = index + 1
        if batch_owner is not None and self.ready_ticket is not ticket:
            batch_owner.stage(self.stream, ticket.generation)
            # Fixed C6 capacity also includes the unused tail for C1-C5.
            self.audit["dma_bytes"] += batch_owner.host.numel()
        if self.ready_layers == len(self.layout.layer_ids):
            self.ready_ticket = ticket

    def set_profiling(self, enabled):
        if self.pending is not None:
            raise RuntimeError("Engram profiling cannot change during a pending request transaction")
        for slots in self.slots.values():
            for slot in slots:
                slot.gather.set_profiling(enabled)
        self.profile_records = [] if enabled else None

    def export_profile(self, path):
        if self.profile_records is None:
            raise RuntimeError("No Engram host profile is active")
        Path(path).write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "tp_rank": self.tp_rank,
                    "clock": "CLOCK_REALTIME equivalent std::chrono::system_clock",
                    "units": "ns",
                    "scope": "native row-copy loop only; excludes worker queue wait, staging copies and HPU DMA",
                    "records": self.profile_records
                },
                indent=2) + "\n")

    def complete(self, ticket, committed_inputs):
        if self.pending is not ticket or ticket.generation != self.generation or self.ready_ticket is not ticket:
            raise RuntimeError("Stale Engram transfer/verify completion")
        if ticket.packet:
            self.c1_packets[ticket.slot].complete(ticket.generation, torch.hpu.current_stream())
        elif (getattr(self, "batches", None) is not None
              and not (self.native_c1 is not None and ticket.batch.hash_ids.shape[0] == 1)):
            self.batches[ticket.slot].complete(ticket.generation)
        else:
            for layer in self.layout.layer_ids:
                slot = self.slots[layer][ticket.slot]
                slot.consumer_done.record(torch.hpu.current_stream())
                slot.inflight = True
        self.history.commit(ticket.batch, committed_inputs)
        if self.c1_abi == 2 and self.native_c1 is not None and len(ticket.batch.compressed_ids) == 1:
            self.native_c1.complete(ticket.batch.request_id, ticket.batch.generation, ticket.generation)
        self.pending = None
        self.ready_ticket = None

    def complete_device(self, ticket, committed_inputs):
        """Complete a device verify transaction after one scalar handoff.

        The PP commit path validates and stages ``committed_inputs`` before
        calling this method.  Engram history remains host-owned, but it never
        performs a second device read or a full commit-buffer ``tolist``.
        """
        if isinstance(committed_inputs, torch.Tensor):
            raise RuntimeError("Engram complete_device requires the already-consumed scalar")
        self.complete(ticket, int(committed_inputs))

    def close(self):
        if self.closed:
            return
        # Exceptions after state writes never invoke another model path. Drain
        # this owner's transfers/consumers before releasing mmap and staging.
        torch.hpu.synchronize()
        if self.pending is not None:
            if self.c1_abi == 2 and self.native_c1 is not None and len(self.pending.batch.compressed_ids) == 1:
                self.native_c1.complete(self.pending.batch.request_id, self.pending.batch.generation,
                                        self.pending.generation)
            self.history.discard(self.pending.batch)
            self.pending = None
        self.ready_ticket = None
        self.native_c1 = None
        if self.device_c1 is not None:
            self.device_c1.close()
        self.device_c1 = None
        self.device_rows = None
        self.device_history = None
        self.slots.clear()
        if self.c1_packets is not None:
            self.c1_packets.clear()
        if getattr(self, "batches", None) is not None:
            self.batches.clear()
        self.tables.clear()
        self.closed = True
