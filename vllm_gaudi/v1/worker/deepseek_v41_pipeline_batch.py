# SPDX-License-Identifier: Apache-2.0
"""Two PP microbatches with separate scratch and persistent transport owners."""
from dataclasses import dataclass

import torch

from vllm_gaudi.ops.deepseek_v41_batch_input import RequestInputFrame
import torch.distributed as dist

from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput


def split_requests(requests):
    if not 32 <= len(requests) <= 64 or len({r.req_id for r in requests}) != len(requests):
        raise ValueError("Two ordinary microbatches require 32..64 distinct request owners")
    middle = (len(requests) + 1) // 2
    return requests[:middle], requests[middle:]


@dataclass
class Lane:
    host: torch.Tensor
    metadata: torch.Tensor
    pages: torch.Tensor
    packet: torch.Tensor
    hidden: torch.Tensor
    pre: torch.Tensor
    token: torch.Tensor
    token_host: torch.Tensor
    producer: object
    transferred: object
    inputs: RequestInputFrame
    generation: int = 0


class TwoMicrobatchPipeline:
    """One scheduler transaction owns both lanes through their final consumer.

    The stage compute stream is ordered. PP transfers use a separate stream,
    joining each producer/consumer individually. Both sends/receives precede
    the reverse token broadcasts, preserving identical HCCL order on peers.
    This does not increase the executor queue depth or duplicate model state.
    """

    def __init__(self, execution):
        self.execution = execution
        self.runner, self.model, self.bank = execution.runner, execution.model, execution.bank
        self.stream = torch.hpu.Stream()
        self.done, self.sampled = torch.hpu.Event(), torch.hpu.Event()
        self.frames = {}
        self.work = []
        self.generation = 0
        for bucket in (16, 32):
            lanes = []
            for _ in range(2):
                host = torch.zeros(3, bucket, dtype=torch.int32).pin_memory("hpu")
                metadata = torch.zeros_like(host, device=execution.device)
                pages = torch.zeros(bucket, self.bank.pages.shape[1], dtype=torch.int32, device=execution.device)
                packet = torch.empty(bucket * 10244, dtype=torch.int32, device=execution.device)
                hidden = packet[:bucket * 10240].view(torch.bfloat16).view(bucket, 4, 5120)
                pre = packet[bucket * 10240:].view(torch.float32).view(bucket, 4)
                token = torch.zeros(bucket, 1, dtype=torch.int32, device=execution.device)
                token_host = torch.zeros_like(token, device="cpu").pin_memory("hpu")
                lanes.append(
                    Lane(host, metadata, pages, packet, hidden, pre, token, token_host, torch.hpu.Event(),
                         torch.hpu.Event(), RequestInputFrame(self.bank, host, metadata, pages)))
            self.frames[bucket] = tuple(lanes)

    def require_ready(self, bucket):
        for replay in self.model.batch_replay_lanes:
            replay.require_ready(bucket)

    def execute(self, requests):
        halves = split_requests(requests)
        execution, runner, bank, model = self.execution, self.runner, self.bank, self.model
        if execution.last_done is not None:
            execution.last_done.synchronize()
        if self.work:
            raise RuntimeError("PP microbatch transport still has an unconsumed generation")
        owners, spans = [], []
        for request in requests:
            start = request.num_computed_tokens
            if start >= request.token_count or start >= model.program.length or len(request.block_ids) != 1:
                raise RuntimeError("PP microbatch lacks a committed input or scheduler pages")
            owner = bank.acquire(request.req_id)
            bank.publish_pages(owner, request.block_ids[0], runner.state.blocks)
            token = request.token_at(start)
            owners.append(owner)
            spans.append((request.req_id, start, [token], [token in (129264, 129265)]))
        generation = bank.begin(owners)
        execution.generation += 1
        self.generation += 1
        bucket = 1 << (len(halves[0]) - 1).bit_length()
        frames = self.frames[bucket]
        offset = 0
        for half, frame in zip(halves, frames, strict=True):
            frame.generation = self.generation
            frame.inputs.prepare(half,
                                 owners[offset:offset + len(half)],
                                 input_ids=(span[2][0] for span in spans[offset:offset + len(half)]))
            offset += len(half)
        pp, compute = runner.pp.group, torch.hpu.current_stream()
        if model.pp_rank == 1:
            with torch.hpu.stream(self.stream):
                for frame in frames:
                    self.work.append(dist.irecv(frame.packet, src=pp.ranks[0], group=pp.device_group))
                    frame.transferred.record()
            runner.pp.receives = getattr(runner.pp, "receives", 0) + 2
        offset = 0
        for lane, (half, frame) in enumerate(zip(halves, frames, strict=True)):
            if frame.generation != self.generation:
                raise RuntimeError("Stale PP microbatch generation")
            ids, positions, slots = frame.inputs.inputs
            arguments = (ids, positions, slots, frame.pages, spans[offset:offset + len(half)])
            if model.pp_rank == 0:
                value = model.forward_request_batch(*arguments, lane=lane)
                frame.hidden.copy_(value["hidden_states"])
                frame.pre.copy_(value["pre_mix"])
                frame.producer.record(compute)
                # Ordinary decode commits one known input. The host staging
                # slot records its actual stage consumer before another ticket
                # is prepared; it cannot be overwritten until that event ends.
                model.complete_request_batch([1] * len(half))
                with torch.hpu.stream(self.stream):
                    self.stream.wait_event(frame.producer)
                    self.work.append(dist.isend(frame.packet, dst=pp.ranks[1], group=pp.device_group))
                    frame.transferred.record()
                runner.pp.sends = getattr(runner.pp, "sends", 0) + 1
            else:
                compute.wait_event(frame.transferred)
                value = model.forward_request_batch(*arguments,
                                                    IntermediateTensors({
                                                        "hidden_states": frame.hidden,
                                                        "pre_mix": frame.pre
                                                    }),
                                                    lane=lane)
                frame.token.copy_(execution.samplers[bucket](value))
                model.complete_request_batch([1] * len(half))
            offset += len(half)
        self.sampled.record(compute)
        with torch.hpu.stream(self.stream):
            self.stream.wait_event(self.sampled)
            for frame in frames:
                pp.broadcast(frame.token, src=1)
                frame.token_host.copy_(frame.token, non_blocking=True)
            self.done.record()
        self.done.synchronize()
        # HCCL handles are retained until the final device consumer. These
        # waits release ownership after completion, not between microbatches.
        for work in self.work:
            work.wait()
        self.work.clear()
        outputs = [
            row for half, frame in zip(halves, frames, strict=True) for row in frame.token_host[:len(half)].tolist()
        ]
        bank.finish(generation, self.done)
        execution.last_done = self.done
        runner.pp.commits = getattr(runner.pp, "commits", 0) + 1
        for request, output in zip(requests, outputs, strict=True):
            request.output.extend(output)
        audit = runner.audit
        for name in ("decode_steps", "target_steps", "batch_decode_steps", "pipeline_decode_steps"):
            audit[name] = audit.get(name, 0) + 1
        audit["target_tokens"] += len(requests)
        audit["batch_padding_rows"] = audit.get("batch_padding_rows", 0) + 2 * bucket - len(requests)
        counts = audit.setdefault("batch_bucket_calls", {})
        counts[bucket] = counts.get(bucket, 0) + 2
        audit["batch_pp_payload_bytes"] = audit.get("batch_pp_payload_bytes", 0) + sum(
            frame.packet.numel() * frame.packet.element_size() for frame in frames)
        ids = [request.req_id for request in requests]
        return (ModelRunnerOutput(
            req_ids=ids, req_id_to_index={
                name: i
                for i, name in enumerate(ids)
            }, sampled_token_ids=outputs) if model.pp_rank == 1 else None)
