# SPDX-License-Identifier: Apache-2.0
"""Small reproduction of the serving decoded-SWA startup contract."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

from vllm_gaudi.entrypoints.deepseek_v41 import prepare_environment
prepare_environment(Path('/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2'),
                    tensor_parallel_size=4, pipeline_parallel_size=1)
import torch
import habana_frameworks.torch  # noqa: F401 - register the HPU backend
from vllm_gaudi.ops.deepseek_v41_prefill_regions import prefill_swa_workspace
from vllm_gaudi.ops.deepseek_v41_math import unpack_swa

torch.ops.load_library(os.environ['VLLM_HPU_DSV4_TPC_OP_LIBRARY'])
root = Path(os.environ['DSV41_RUN_EVIDENCE'])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokens', default='1024,2048')
parser.add_argument('--decoded-only', action='store_true')
parser.add_argument('--native-kv', action='store_true')
parser.add_argument('--prepared-pass', action='store_true')
parser.add_argument('--position-bank', action='store_true')
parser.add_argument('--mme-kv', action='store_true')
parser.add_argument('--engine-config', action='store_true')
parser.add_argument('--reuse-state', action='store_true')
parser.add_argument('--random-values', action='store_true',
                    help='Check decoded state against canonical packed bytes at varied finite BF16 scales')
args = parser.parse_args()
context = nullcontext()
if args.engine_config:
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    config = EngineArgs(model='/opt/optane/prepared/DeepSeek-V4.1-Flash-dba1be0-tp4-pp1-q16v2',
                        dtype='bfloat16', tensor_parallel_size=4, pipeline_parallel_size=1,
                        max_model_len=1048576, max_num_batched_tokens=8192, max_num_seqs=32,
                        block_size=128, load_format='dsv41_prepared', enable_prefix_caching=False,
                        async_scheduling=True).create_engine_config()
    context = set_current_vllm_config(config)
if args.prepared_pass:
    from vllm_gaudi.ops.tp2_prepared_plan import register_tp2_prepared_group_pass
    register_tp2_prepared_group_pass()
report = dict(status='running', cases=[],
              compiler_environment={k: v for k, v in os.environ.items() if k.startswith('FUSER_')})
try:
    with context, torch.inference_mode():
        owner = torch.nn.Module()
        if args.reuse_state:
            owner.register_buffer('swa', torch.zeros(256, 528, dtype=torch.uint8, device='hpu'))
            owner.register_buffer('decoded', torch.zeros(20480, 512, dtype=torch.bfloat16, device='hpu'))
            owner.register_buffer('positions', torch.arange(8192, dtype=torch.int32, device='hpu'))
            owner.register_buffer('offsets', torch.arange(128, dtype=torch.int32, device='hpu'))
        for tokens in map(int, args.tokens.split(',')):
            kv = torch.ones(tokens, 512, dtype=torch.bfloat16, device='hpu') * .125
            positions = torch.arange(max(tokens, 8192) if args.position_bank else tokens,
                                     dtype=torch.int32, device='hpu')[:tokens]
            if args.reuse_state:
                positions = owner.positions[:tokens]
            if args.mme_kv:
                hidden = torch.full((tokens, 5120), .03125, dtype=torch.bfloat16, device='hpu')
                projection = torch.full((1792, 5120), .125, dtype=torch.bfloat16, device='hpu')
                kv = torch.nn.functional.linear(hidden, projection)[:, 1280:].contiguous()
            if args.native_kv:
                weight = torch.ones(512, dtype=torch.bfloat16, device='hpu')
                phase = torch.zeros(tokens, 64, dtype=torch.float32, device='hpu')
                kv = torch.ops.custom_op.custom_deepseek_v41_attention_norm_bf16_gaudi2(kv, weight, 1e-6)
                kv = torch.ops.custom_op.custom_deepseek_v41_prefill_rope_bf16_gaudi2(
                    kv.reshape(tokens, 1, 512), positions, phase).reshape(tokens, 512)
            if args.random_values:
                generator = torch.Generator(device='cpu').manual_seed(1701 + tokens)
                values = torch.randn(tokens, 512, generator=generator)
                scales = torch.tensor([2.0**exponent for exponent in (-20, -10, -3, 0, 3, 10, 20, 40)])
                values *= scales[torch.arange(tokens).remainder(scales.numel())].unsqueeze(-1)
                values[0, :32] = 0
                values[0, 1] = -0.0
                kv = values.to(torch.bfloat16).to('hpu')
            offsets = torch.arange(128, dtype=torch.int32, device='hpu')
            for decoded in ((True,) if args.decoded_only else (False, True)):
                if args.reuse_state:
                    swa = owner.swa.zero_()
                    state = owner.decoded.zero_() if decoded else None
                    positions, offsets = owner.positions[:tokens], owner.offsets
                else:
                    swa = torch.zeros(256, 528, dtype=torch.uint8, device='hpu')
                    state = torch.zeros(20480, 512, dtype=torch.bfloat16, device='hpu') if decoded else None
                actual = prefill_swa_workspace(kv, positions, swa, offsets, state, 12288)
                torch.hpu.synchronize()
                expected = kv.cpu()
                cache, indices = (value.cpu() for value in actual)
                assert cache.shape == (127 + tokens, 512) and torch.isfinite(cache).all()
                assert torch.equal(cache[-tokens:], expected)
                if decoded:
                    canonical = unpack_swa(swa.cpu())
                    observed = state[12288:12544].cpu()
                    mismatch = observed != canonical
                    if mismatch.any():
                        coordinates = mismatch.nonzero()[:12]
                        report['codec_mismatch'] = dict(count=int(mismatch.sum()),
                            coordinates=coordinates.tolist(),
                            observed=[float(observed[tuple(index)]) for index in coordinates.tolist()],
                            canonical=[float(canonical[tuple(index)]) for index in coordinates.tolist()])
                    # The accepted HPU activation codec canonicalizes zero;
                    # CPU unpack preserves its sign. Require exact nonzero
                    # values and retain the harmless signed-zero difference.
                    report['signed_zero_differences'] = report.get('signed_zero_differences', 0) + int(
                        ((observed == 0) & (canonical == 0) &
                         (observed.view(torch.int16) != canonical.view(torch.int16))).sum())
                    assert torch.equal(observed, canonical), \
                        'Decoded ring differs from canonical packed SWA bytes'
                report['cases'].append(dict(tokens=tokens, decoded=decoded, status='passed'))
                (root/'workspace.json').write_text(json.dumps(report,indent=2)+'\n')
        report['status'] = 'passed'
except BaseException as error:
    report.update(status='failed',error=repr(error))
    raise
finally:
    (root/'workspace.json').write_text(json.dumps(report,indent=2)+'\n')
