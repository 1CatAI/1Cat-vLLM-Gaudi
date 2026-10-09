# SPDX-License-Identifier: Apache-2.0
"""Untimed compiled registration/layout gate over real projection inputs."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--device-epoch', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment, prepare_native_libraries
    prepare_environment(args.prepared, tensor_parallel_size=4, pipeline_parallel_size=1)
    prepare_native_libraries()
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = str(root / f'recipes/rank{rank}') + ',false,4096'
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.entrypoints.deepseek_v41 import load_native_operators
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    bind_worker_cpu(rank)
    torch.set_grad_enabled(False)
    load_native_operators()
    torch.ops.load_library(str(args.library.resolve()))
    os.environ['VLLM_HPU_DSV41_BOUNDED_RECEIVE_PARENT_KERNEL'] = os.environ['GC_KERNEL_PATH']
    os.environ['GC_KERNEL_PATH'] = str(args.database.resolve())
    torch.hpu.set_device(rank)
    bind_worker_helpers(rank)
    config_scope = set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4)))
    config_scope.__enter__()
    report = dict(status='running', rank=rank, checks=[], timed=False, gain_credit_ms=0)
    output = root / f'future-peer-rank{rank}.json'

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    def raw(name, dtype, shape):
        data = bytearray((args.fixtures / f'rank{rank}' / name).read_bytes())
        return torch.frombuffer(data, dtype=dtype).reshape(shape).clone().to('hpu')

    try:
        shard = PreparedV41Shard(args.prepared, 0, rank)
        weight = raw('producer-weight.bin', torch.bfloat16, (5120, 2048))
        gate = raw('consumer-weight.bin', torch.float32, (384, 5120))
        norm = shard.tensor('layers.20.ffn_norm.weight', 'hpu')
        operation = torch.ops.custom_op.custom_deepseek_v41_future_peer_sum_gaudi2
        for rows in (2, 6):
            for group in (2, 4):
                owner_rank = rank % group
                capacity = 128
                while 4*capacity*group*rows*5120 <= 48 << 20:
                    capacity *= 2
                peers = torch.zeros((2, capacity, group, rows, 5120), dtype=torch.bfloat16, device='hpu')
                flags = torch.ones((1 << 24,), dtype=torch.int32, device='hpu')
                if args.device_epoch:
                    flags.zero_()
                epoch = flags if args.device_epoch else torch.ones((1,), dtype=torch.int32, device='hpu')
                bank = 0 if args.device_epoch else 1

                def body(x, residual, post, comb, pre, peer_values, flag_values, epoch_values, *,
                         candidate, owner_rank=owner_rank, group=group, rows=rows, bank=bank):
                    if candidate and args.device_epoch:
                        x = torch.ops.custom_op.custom_deepseek_v41_future_epoch_gaudi2(x, flag_values)
                    local = torch.nn.functional.linear(x, weight)
                    if candidate:
                        value, packet, status = operation(local.contiguous(), peer_values, flag_values, epoch_values,
                                                           0, owner_rank, 65536, True)
                    else:
                        value = (local if owner_rank == 0 else peer_values[bank, 0, 0]).float()
                        for peer_rank in range(1, group):
                            packet_rank = local if owner_rank == peer_rank else peer_values[bank, 0, peer_rank]
                            value = value + packet_rank.float()
                        value, packet = value.bfloat16(), local
                        status = torch.ones((rows, 40), dtype=torch.int32, device='hpu')
                    residual_out, collapsed = torch.ops.custom_op.custom_deepseek_v41_mhc_post_collapse_gaudi2(
                        value.contiguous(), residual, post, comb, pre)
                    normalized, quantized, scale = torch.ops.custom_op.custom_deepseek_v41_ffn_norm_quant_gaudi2(
                        collapsed, norm, 1e-20)
                    logits = torch.nn.functional.linear(normalized.float(), gate)
                    return residual_out, normalized, quantized, scale, logits, packet, status

                reference = torch.compile(lambda *v, function=body: function(*v, candidate=False),
                                          backend='hpu_backend', fullgraph=True, dynamic=False)
                candidate = torch.compile(lambda *v, function=body: function(*v, candidate=True),
                                          backend='hpu_backend', fullgraph=True, dynamic=False)
                for case in range(3):
                    x = raw(f'input-{case}.bin', torch.bfloat16, (6, 2048))[:rows].contiguous()
                    packets = raw(f'peers-{case}.bin', torch.bfloat16, (4, 6, 5120))[:group, :rows].contiguous()
                    peers[bank, 0].copy_(packets)
                    if args.device_epoch:
                        flags[0] = 2*(case+1)
                    fields = [raw(f'{key}-{case}.bin', dtype, shape)[:rows].contiguous()
                              for key, dtype, shape in (
                                  ('residual', torch.bfloat16, (6, 4, 5120)),
                                  ('post', torch.float32, (6, 4)),
                                  ('comb', torch.float32, (6, 4, 4)),
                                  ('pre', torch.float32, (6, 4)))]
                    expected = [v.cpu() for v in reference(x, *fields, peers, flags, epoch)]
                    actual = [v.cpu() for v in candidate(x, *fields, peers, flags, epoch)]
                    exact = [torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                             for a, b in zip(expected[:-1], actual[:-1], strict=True)]
                    ready = bool((actual[-1] > 0).all())
                    observed_epoch = int(flags[-1].cpu()) if args.device_epoch else None
                    if args.device_epoch and observed_epoch != 2*(case+1):
                        raise AssertionError('Transport generation did not mutate owned flags in-place')
                    report['checks'].append(dict(rows=rows, group=group, case=case,
                                                 exact_fields=exact, ready=ready, capacity=capacity,
                                                 observed_device_epoch=observed_epoch))
                    save()
                    if not all(exact) or not ready:
                        raise AssertionError('Future registration changed the production projection/consumer boundary')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        save()
        config_scope.__exit__(None, None, None)


if __name__ == '__main__':
    main()
