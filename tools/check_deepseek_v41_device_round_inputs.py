# SPDX-License-Identifier: Apache-2.0
"""Check mapped C6 Engram, accepted history and queued native output readback."""
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    rank = int(os.environ['LOCAL_RANK'])
    os.environ['HLS_MODULE_ID'] = os.environ['HABANA_VISIBLE_MODULES'].split(',')[rank]
    os.environ['PT_HPU_RECIPE_CACHE_CONFIG'] = os.environ.get('PT_HPU_RECIPE_CACHE_CONFIG', '').replace(
        '{rank}', str(rank))
    os.environ['VLLM_HPU_DSV41_GRAPH_REPLAY'] = '1'
    import numpy as np
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment, initialize_model_parallel,
                                  destroy_model_parallel, destroy_distributed_environment)
    from vllm_gaudi.distributed.tp2_fused_ar_norm import initialize_tp2_fused_ar_norm_runtime
    from vllm_gaudi.ops.deepseek_v4_config import bind_worker_cpu, bind_worker_helpers
    from vllm_gaudi.ops.deepseek_v41_completion import resolve_device_runtime
    from vllm_gaudi.ops.deepseek_v41_device_engram import DeviceEngramRounds
    from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory
    from vllm_gaudi.ops.deepseek_v41_math import unpack_swa
    from vllm_gaudi.ops.deepseek_v41_round_inputs import DeviceRoundInputs
    from vllm_gaudi.ops.deepseek_v41_verify import VerifyRing
    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    torch.hpu.set_device(rank)
    bind_worker_cpu(rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=4))
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(world_size=4, rank=rank, distributed_init_method='env://',
                                     local_rank=rank, backend='hccl')
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
        initialize_tp2_fused_ar_norm_runtime()
        bind_worker_helpers(rank)
        bridge, _ = resolve_device_runtime(4)
        if os.environ.get('GRAPH_VISUALIZATION') == '1':
            from vllm_gaudi.ops.deepseek_v41_native_trace import configure_post_graph_directory
            configure_post_graph_directory(root / 'graphs' / f'rank{rank}')
        layout = EngramHashLayout.from_config(dict(engram_layer_ids=[1, 14], engram_num_embeddings=[10000, 10000],
            engram_max_ngram_size=4, engram_n_heads=8, engram_compressed_vocab_size=8, engram_vocab_size=5,
            engram_pad_token_id=2, engram_head_dim=256))
        host = SimpleNamespace(layout=layout, tensor_parallel_size=4, tp_rank=rank, shards={}, table_sources={})
        host.history = EngramTokenHistory(layout, np.arange(128) % 8)
        host.history.reset('fixture')
        prefix = host.history.prepare('fixture', [5, 6, 7])
        host.history.commit(prefix, 3)
        tables = {}
        for layer in layout.layer_ids:
            shard = layout.head_shard(layer, rank, 4)
            host.shards[layer] = shard
            rows = shard['row_stop'] - shard['row_start']
            generator = torch.Generator().manual_seed(300 + layer + rank)
            weight = (torch.rand(rows, 256, generator=generator) * 4 - 2).to(torch.float8_e4m3fn).view(torch.uint8)
            scale = torch.randint(124, 130, (rows, 8), generator=generator, dtype=torch.uint8)
            tables[layer] = torch.cat((weight, scale), dim=-1)
            sources = []
            for name, data in [('weight', weight), ('scale', scale)]:
                path = root / f'{name}-l{layer}-r{rank}.bin'
                path.write_bytes(data.numpy().tobytes())
                sources.append(dict(file=str(path), shard_offset=0))
            host.table_sources[layer] = tuple(sources)
        cursor = DeviceRoundInputs(torch.empty(6, dtype=torch.int64, device='hpu'),
                                   torch.empty(6, dtype=torch.int32, device='hpu'))
        cursor.seed('fixture', [10, 11, 12, 13, 14, 15], 3, 1000, 4096, 1, [7, 6, 5])
        engram = DeviceEngramRounds(host, cursor.ids, cursor.history)
        ring = VerifyRing('hpu', last_rank=True, native_readback=bridge.copy_integer_record_to_host)
        consume = torch.compile(lambda a, b: (a.flatten(1) + b.flatten(1)).sum(-1),
                                backend='hpu_backend', fullgraph=True, dynamic=False)
        results = []
        tokens = [10, 11, 12, 13, 14, 15]
        try:
            for iteration, count in enumerate([1, 6, 2, 5, 3, 4]):
                rows = engram.prepare('fixture', cursor.ids, cursor.history)
                consumed = consume(*rows)
                batch = host.history.prepare('fixture', tokens)
                for layer, actual in zip(layout.layer_ids, rows, strict=True):
                    shard = host.shards[layer]
                    first, last = shard['head_start'], shard['head_stop']
                    hashes = batch.hash_ids[:, layout.layer_ids.index(layer), first:last] - shard['row_start']
                    expected = unpack_swa(tables[layer][torch.from_numpy(hashes.astype('int64'))], 256)
                    if not torch.equal(actual.cpu(), expected):
                        raise AssertionError(f'layer{layer} mapped C6 differs, accepted test {count}')
                # Include the consumer before validating/advancing the cursor.
                assert torch.isfinite(consumed.cpu()).all()
                next_anchor = 40 + iteration
                outputs = [next_anchor - count + 1 + index for index in range(count)] + [-1] * (6 - count)
                drafts = list(range(next_anchor + 1, next_anchor + 6))
                ticket = ring.acquire(6)
                ticket.record.copy_(torch.tensor([ticket.generation, count, count, 5, *outputs, *drafts, 0],
                                                  dtype=torch.int64, device='hpu'))
                ring.stage(ticket)
                cursor.next(ticket.record, engram.histories)
                # The next producer is deliberately before scheduler readback.
                next_rows = engram.prepare('fixture', cursor.ids, cursor.history)
                consume(*next_rows)
                committed, output, proposed = ring.consume(ticket)
                ring.release(ticket)
                assert (committed, output, proposed) == (count, outputs[:count], drafts)
                host.history.commit(batch, count)
                expected_tail = np.full(3, -1, dtype='int32')
                tail = host.history.history[-3:][::-1]
                expected_tail[:len(tail)] = tail
                assert torch.equal(cursor.history.cpu(), torch.from_numpy(expected_tail))
                assert cursor.ids.cpu().tolist() == [next_anchor, *drafts]
                assert int(cursor.positions.cpu()[0]) == host.history.position
                results.append(dict(committed=count, both_engram_layers_exact=True, cursor_exact=True,
                                    next_consumer_before_host_read=True))
                tokens = [next_anchor, *drafts]
            document = dict(status='passed', rank=rank, results=results, performance_qualification=False)
        finally:
            ring.close()
            engram.close()
            destroy_model_parallel()
            destroy_distributed_environment()
        (root / f'round-input-rank{rank}.json').write_text(json.dumps(document, indent=2) + '\n')


if __name__ == '__main__':
    main()
