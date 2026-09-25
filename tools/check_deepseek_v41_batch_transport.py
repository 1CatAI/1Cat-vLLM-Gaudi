# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Exercise the production batch runner's TP->PP->TP->sample dependency chain."""
import json
import faulthandler
import os
from pathlib import Path
from types import SimpleNamespace

rank = int(os.environ["LOCAL_RANK"])
os.environ["HLS_MODULE_ID"] = os.environ["HABANA_VISIBLE_MODULES"].split(",")[rank]
stage_topology = os.environ.get("DSV41_TRANSPORT_STAGE_TOPOLOGY") == "1"
pipeline = os.environ.get("VLLM_HPU_DSV41_PP_MICROBATCHES") == "2"
if os.environ.get("DSV41_MICRO_RANK_CPUS"):
    os.sched_setaffinity(0, json.loads(os.environ["DSV41_MICRO_RANK_CPUS"])[rank])
else:
    main_cpu = int(os.environ["VLLM_HPU_DSV4_WORKER_CPUS"].split(",")[rank])
    helpers = os.environ["VLLM_HPU_DSV4_WORKER_HELPER_CPUS"].split(";")[rank]
    os.sched_setaffinity(0, [main_cpu, *(int(cpu) for cpu in helpers.split(",") if cpu)])

import torch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (init_distributed_environment, initialize_model_parallel, get_pp_group,
                              destroy_distributed_environment, destroy_model_parallel)
from vllm.sequence import IntermediateTensors
from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
from vllm_gaudi.ops.deepseek_v41_batch_state import BatchStageState
from vllm_gaudi.ops.tp2_model_adapter import DecoderTopology
from vllm_gaudi.ops.tp2_prepared_plan import (collect_prepared_group_replays, record_native_decoder_outputs,
                                              replay_native_decoder, prepared_group_stats,
                                              shutdown_prepared_group_plans)
from vllm_gaudi.v1.worker.deepseek_v41_batch_runner import BatchExecution
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState


class Model:

    def __init__(self):
        self.pp_rank, self.engram_host = rank // 2, None
        self.program = SimpleNamespace(length=1048576,
                                       sample_greedy_token=lambda x: x[:, :1].int(),
                                       layers=[],
                                       shared=SimpleNamespace(block_table=torch.zeros(8192,
                                                                                      dtype=torch.int32,
                                                                                      device="hpu"),
                                                              topk={}))
        self.batch_state = BatchStageState(self.program, 64)
        self.entries = {}
        self.completed = 0
        self.control_weight = torch.zeros(24, 20480, dtype=torch.float32, device="hpu") if stage_topology else None

    def complete_request_batch(self, counts):
        assert all(count == 1 for count in counts)
        self.completed += len(counts)

    def forward_request_batch(self, ids, positions, slots, pages, spans, intermediate_tensors=None, *, lane=0):
        b = ids.numel()
        if self.pp_rank == 0:
            hidden = ids.to(torch.bfloat16)[:, None, None].expand(b, 4, 5120).contiguous()
            pre = positions.float()[:, None].expand(b, 4).contiguous()
        else:
            hidden, pre = intermediate_tensors["hidden_states"], intermediate_tensors["pre_mix"]
        key = (lane, b)
        if key not in self.entries:
            from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
            reduce, _ = stage_collectives(rank % 2, True)
            owner = torch.nn.Identity()
            fixed = tuple(v.clone() for v in (hidden, pre, positions))
            counts = (11, 10, 10, 10, 8 + self.pp_rank) if stage_topology else (2, )

            def make_program(index, count):

                def program(x, previous, pos):
                    current = x[:, 0, :] if index == 0 else x
                    for i in range(count):
                        current = reduce(current + int(index == 0 and i < 2)) * 0.5
                        if stage_topology and i == 0:
                            residual = previous.repeat(1, 5120).contiguous()
                            independent = torch.ops.custom_op.custom_deepseek_v41_control_gemv_f32_gaudi2(
                                residual, self.control_weight)
                            current = current + independent[:, :1].to(current.dtype)
                    # Consumption of the received pre-mix affects sampling.
                    current = current + (previous[:, :1] - pos[:, None].float()).to(current.dtype)
                    return current, previous, pos

                # Each group must own its compiled wrapper and prepared plan.
                import types
                program = types.FunctionType(program.__code__.replace(co_name=f"transport_{index}_{count}"),
                                             program.__globals__,
                                             closure=program.__closure__)
                from vllm_gaudi.compilation.deepseek_v41_overlap import make_backend
                backend = make_backend() if stage_topology else "hpu_backend"
                return torch.compile(program, backend=backend, fullgraph=True, dynamic=False)

            compiled = [make_program(i, n) for i, n in enumerate(counts)]
            topology = DecoderTopology("deepseek_v41_batch_transport", (4, ) * 5, 2, False,
                                       sum(counts) -
                                       40) if stage_topology else DecoderTopology("deepseek_v41_batch_transport",
                                                                                  (1, ), 2, False)
            self.entries[key] = owner, fixed, compiled, topology
        owner, fixed, compiled, topology = self.entries[key]
        metadata = SimpleNamespace(is_prompt=False)
        roots = dict(hidden_states=hidden, pre_mix=pre, positions=positions, metadata=metadata)
        result = replay_native_decoder(owner, **roots)
        if result is None:
            for dest, source in zip(fixed, (hidden, pre, positions)):
                dest.copy_(source)
            with collect_prepared_group_replays(owner=owner,
                                                adapter=topology,
                                                snapshot=lambda: SimpleNamespace(restore=lambda: None),
                                                **dict(roots,
                                                       hidden_states=fixed[0],
                                                       pre_mix=fixed[1],
                                                       positions=fixed[2])) as context:
                values = fixed
                for index, group in enumerate(compiled):
                    context["group_index"] = index
                    values = group(*values)
                result = values[0], values[1], None
                record_native_decoder_outputs(*result)
        if self.pp_rank == 0:
            return IntermediateTensors({
                "hidden_states": result[0][:, None, :].expand(b, 4, 5120),
                "pre_mix": result[1]
            })
        return result[0]


@torch.inference_mode()
def main():
    torch.hpu.set_device(rank)
    init_distributed_environment(world_size=4,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    initialize_tp2_fused_ar_norm_runtime()
    if stage_topology:
        torch.ops.load_library(os.environ["VLLM_HPU_DSV4_TPC_OP_LIBRARY"])
    model = Model()
    runner = SimpleNamespace(model=model,
                             device=torch.device("hpu"),
                             state=SimpleNamespace(blocks=8193),
                             pp=SimpleNamespace(group=get_pp_group()),
                             audit=dict(decode_steps=0, target_steps=0, target_tokens=0))
    batch = BatchExecution(runner, 64)
    records = []
    # Include partial buckets and out-of-order slots, then reuse every bucket.
    sizes = ((32, 64, 33, 63, 32, 8) if pipeline else (1, 8, 64, 7, 1) if stage_topology else
             (1, 2, 3, 8, 15, 32, 63, 64, 7, 1))
    if os.environ.get("DSV41_TRANSPORT_SIZES"):
        sizes = tuple(int(value) for value in os.environ["DSV41_TRANSPORT_SIZES"].split(","))
    faulthandler.enable()
    for generation, count in enumerate(sizes):
        requests = [
            RequestState(f"generation{generation}-request{i}", [i + 1], [], None, ([i + 1], )) for i in range(count)
        ]
        requests.reverse()
        for step in range(5):
            print(f"TRANSPORT rank={rank} count={count} generation={generation} step={step} begin", flush=True)
            faulthandler.dump_traceback_later(120, repeat=False)
            expected = [[request.tokens[-1] + 4] for request in requests]
            output = batch.execute(requests)
            faulthandler.cancel_dump_traceback_later()
            print(f"TRANSPORT rank={rank} count={count} generation={generation} step={step} complete", flush=True)
            actual = [[request.tokens[-1]] for request in requests]
            assert actual == expected, (rank, generation, step, actual, expected)
            if model.pp_rank == 1:
                assert output.sampled_token_ids == expected
            for request in requests:
                request.num_computed_tokens += 1
        for request in requests:
            model.batch_state.release(request.req_id)
        records.append(dict(generation=generation, count=count, steps=5, exact=True))
    evidence = Path(os.environ["DSV41_RUN_EVIDENCE"])
    (evidence / f"rank{rank}.json").write_text(
        json.dumps(dict(records=records,
                        audit=runner.audit,
                        completed_inputs=model.completed,
                        native=prepared_group_stats()),
                   indent=2))
    shutdown_prepared_group_plans()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    try:
        with set_current_vllm_config(
                VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2))):
            main()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)  # torchrun terminates peers waiting on a failed PP producer.
