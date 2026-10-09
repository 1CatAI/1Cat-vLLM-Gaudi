# SPDX-License-Identifier: Apache-2.0
"""CPU-only real weights/fixture seeds for an untimed future-input capability."""
import argparse
import json
from pathlib import Path

import torch

from vllm_gaudi.ops.deepseek_v41_shard_loader import PreparedV41Shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=range(4), default=0)
    parser.add_argument('--peer-fixtures', type=Path)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--production-post', action='store_true')
    args = parser.parse_args()
    if args.production_post and (args.peer_fixtures is None or args.layer != 20):
        parser.error('The archived post fixtures cover layer20 and require --peer-fixtures')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    shard = PreparedV41Shard(args.prepared, 0, args.rank)
    prefix = f'layers.{args.layer}'
    producer = shard.dense(prefix+'.attn.wo_b.weight', 'cpu')
    consumer = (shard.tensor(prefix+'.ffn.gate.weight', 'cpu') if args.production_post
                else shard.dense(prefix+'.attn.wq_a.weight', 'cpu'))
    expected_consumer = (384, 5120) if args.production_post else (1280, 5120)
    if tuple(producer.shape) != (5120, 2048) or tuple(consumer.shape) != expected_consumer:
        raise ValueError(f'Checkpoint dimensions differ: {producer.shape}, {consumer.shape}')

    def save(name, value):
        value.contiguous().view(torch.uint8).numpy().tofile(args.output / name)

    save('producer-weight.bin', producer)
    save('consumer-weight.bin', consumer.float() if args.production_post else consumer)
    if args.production_post:
        save('norm-weight.bin', shard.tensor(prefix+'.ffn_norm.weight', 'cpu'))
        from vllm_gaudi.ops.deepseek_v41_math import hc_pre
        control = [shard.tensor(prefix+'.hc_attn_'+name, 'cpu').float() for name in ('fn', 'scale', 'base')]
    cases = []
    for c in range(3):
        path = args.fixtures / f'rank{args.rank}' / f'c6-{c}.pt'
        data = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        # A capability seed from actual hidden rows. This is not the captured
        # WO_B activation and is not used to qualify production performance.
        if args.peer_fixtures is None:
            x = data['hidden'].reshape(6, -1)[:, :2048].to(torch.bfloat16).contiguous()
        else:
            packet_path = args.peer_fixtures / f'peer-inputs-rank{args.rank}-case{c}.pt'
            packet = torch.load(packet_path, weights_only=True, map_location='cpu', mmap=True)[0]
            x = packet['projection_input'].bfloat16().contiguous()
            if x.shape != (6, 2048) or not data['request_context_qualified'] or not data['groups_qualified']:
                raise ValueError('Production projection fixture contract differs')
        if args.production_post:
            residual, previous = (data['groups'][4][k].contiguous() for k in ('residual', 'pre'))
            _, pre, post, comb = hc_pre(residual, previous, *control)
            for key, value in (('residual', residual), ('pre', pre), ('post', post), ('comb', comb)):
                save(f'{key}-{c}.bin', value.contiguous())
        peers = []
        for rank in range(4):
            peer_shard = PreparedV41Shard(args.prepared, 0, rank)
            weight = peer_shard.dense(prefix+'.attn.wo_b.weight', 'cpu')
            if args.peer_fixtures:
                other = torch.load(args.peer_fixtures / f'peer-inputs-rank{rank}-case{c}.pt',
                                   weights_only=True, map_location='cpu', mmap=True)[0]
                peer_input = other['projection_input'].float()
            else:
                peer_input = x.float()
            peers.append(torch.nn.functional.linear(peer_input, weight.float()).to(torch.bfloat16))
        save(f'input-{c}.bin', x)
        save(f'peers-{c}.bin', torch.stack(peers))
        cases.append({'fixture': str(path), 'input_shape': list(x.shape)})
    (args.output / 'fixture.json').write_text(json.dumps({
        'cases': cases, 'actual_checkpoint_weights': True,
        'production_wo_b_input_capture': args.peer_fixtures is not None, 'timed': False,
        'layer': args.layer, 'production_post': args.production_post,
        'post_gates': 'Frozen CPU official FP32 hc_pre over real residual; both arms use identical gates',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
