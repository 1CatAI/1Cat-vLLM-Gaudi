# SPDX-License-Identifier: Apache-2.0
"""Qualify TP4 compiled embedding/residual ingress through normalization and sampling."""
import argparse
import json
import os
from pathlib import Path
import time
from types import MethodType

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('prepared', type=Path)
args = parser.parse_args()
rank = int(os.environ['LOCAL_RANK'])
os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
import torch
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm_gaudi.compilation.deepseek_v41_tp4 import make_backend
from vllm_gaudi.models.deepseek_v41_program import PreparedStage, PreparedInput, _weight_tree, load_weight_tree
from vllm_gaudi.ops.deepseek_v41_math import final_collapse_rms_norm
from vllm_gaudi.ops.deepseek_v41_replay import stage_collectives
from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu

torch.hpu.set_device(rank)
bind_worker_cpu(rank)
torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://', local_rank=rank, backend='hccl')
config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=1))
root = Path(os.environ['DSV41_RUN_EVIDENCE'])
report = dict(rank=rank, status='running', cases=[])
try:
    with set_current_vllm_config(config), torch.inference_mode():
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        reduce, gather = stage_collectives(rank, False)
        shard = PreparedV41Shard(args.prepared, 0, rank)
        specs = {name: spec for name, spec in shard.specs.items()
                 if name in ('embed.weight', 'norm.weight', 'head.weight')}
        stage = torch.nn.Module()
        stage.weights = _weight_tree(specs)
        load_weight_tree(shard, stage.weights, 'hpu', specs)
        stage.is_last_stage, stage.tp_rank, stage.all_gather = True, rank, gather
        stage.bf16_head = True
        stage._head_projection = MethodType(PreparedStage._head_projection, stage)
        stage.sample_greedy = MethodType(PreparedStage.sample_greedy, stage)
        eps = json.loads((args.prepared / 'config.json').read_text())['text_config']['rms_norm_eps']
        ingress = PreparedInput(stage.weights.embed, rank, reduce)
        fast_ingress = torch.compile(ingress, backend=make_backend(), fullgraph=True, dynamic=False)

        def consumer(hidden, pre):
            value = final_collapse_rms_norm(hidden, pre, stage.weights.norm.weight, eps)
            return stage.sample_greedy(value[-1:])
        fast_consumer = torch.compile(consumer, backend=make_backend(), fullgraph=True, dynamic=False)
        width = stage.weights.embed.weight.shape[0]
        for values in ([7], [0, width - 1, width, width * 2, width * 3, 129265], [width * 4 - 1], [42]):
            ids = torch.tensor(values, device='hpu')
            expected, expected_pre = ingress(ids)
            expected_host, expected_pre_host = expected.cpu(), expected_pre.cpu()
            expected_token = consumer(expected, expected_pre).cpu()
            actual, actual_pre = fast_ingress(ids)
            assert torch.equal(actual.cpu(), expected_host)
            assert torch.equal(actual_pre.cpu(), expected_pre_host)
            actual_token = fast_consumer(actual, actual_pre).cpu()
            assert torch.equal(actual_token, expected_token)
            timings = {}
            for name, input_fn, output_fn in [('reference', ingress, consumer),
                                               ('candidate', fast_ingress, fast_consumer)]:
                samples = []
                for _ in range(3):
                    torch.hpu.synchronize()
                    started = time.perf_counter()
                    hidden, pre = input_fn(ids)
                    token = output_fn(hidden, pre).cpu()
                    samples.append((time.perf_counter() - started) * 1000)
                    assert torch.equal(token, expected_token)
                timings[name] = samples
            report['cases'].append(dict(input_ids=values, selected=actual_token.tolist(), exact=True,
                                        host_sync_ms=timings))
            (root / f'io-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        report['status'] = 'passed'
except Exception as error:
    import traceback
    traceback.print_exc()
    report.update(status='failed', error=repr(error))
finally:
    (root / f'io-rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'passed':
        os._exit(1)
