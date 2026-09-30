# SPDX-License-Identifier: Apache-2.0
"""Small mapped-host capability and numerical gate; no performance claim."""
import json
import fcntl
import os
from pathlib import Path


def main():
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    import numpy as np
    import torch
    import torch.distributed as dist
    import habana_frameworks.torch.core  # noqa: F401
    from vllm_gaudi.ops.deepseek_v41_completion import _control_bridge
    from vllm_gaudi.ops.deepseek_v41_engram import EngramHashLayout, EngramTokenHistory
    from vllm_gaudi.ops.deepseek_v41_host import device_engram_parameters

    root = Path(os.environ['DSV41_RUN_EVIDENCE'])
    report = dict(status='running', scope='mapped-host producer, six heads per TP4 shard', cases=[])
    producer = None
    backing_fds = []
    try:
        print('Acquiring one HPU for mapped-host capability', flush=True)
        torch.hpu.set_device(0)
        print('Initializing local lifetime communicator', flush=True)
        dist.init_process_group('hccl', store=dist.HashStore(), rank=0, world_size=1)
        ready = torch.ones(1, dtype=torch.bfloat16, device='hpu')
        dist.all_reduce(ready)
        ready.cpu()
        backend = dist.group.WORLD._get_backend(torch.device('hpu'))
        print('Loading version-locked producer', flush=True)
        bridge = _control_bridge()
        assert bridge.device_engram_tp4_version == 3
        layout = EngramHashLayout.from_config(dict(
            engram_layer_ids=[1], engram_num_embeddings=[8192], engram_max_ngram_size=4,
            engram_n_heads=8, engram_compressed_vocab_size=97, engram_vocab_size=101,
            engram_pad_token_id=0, engram_head_dim=256))
        token_map = np.arange(129280, dtype=np.int32) % 97
        device_map = torch.tensor(token_map, device='hpu', dtype=torch.int32)
        raw = torch.empty(1, device='hpu', dtype=torch.int32)
        history = torch.full((3,), -1, device='hpu', dtype=torch.int32)
        next_history = torch.empty_like(history)
        decoded = torch.empty(1, 6, 256, device='hpu', dtype=torch.bfloat16)
        for rank in range(4):
            shard = layout.head_shard(1, rank, 4)
            rows = shard['row_stop'] - shard['row_start']
            rng = np.random.default_rng(731 + rank)
            weights = rng.integers(0, 120, size=(rows, 256), dtype=np.uint8)
            scales = rng.integers(120, 132, size=(rows, 8), dtype=np.uint8)
            weight_path, scale_path = root / f'w{rank}.bin', root / f's{rank}.bin'
            weights.tofile(weight_path)
            scales.tofile(scale_path)
            sources = []
            for label, array in (("weight", weights), ("scale", scales)):
                fd = os.memfd_create(f"tp4-probe-{rank}-{label}", os.MFD_ALLOW_SEALING)
                backing_fds.append(fd)
                payload = array.tobytes()
                assert os.write(fd, payload) == len(payload)
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                sources.append(f"/proc/{os.getpid()}/fd/{fd}")
            parameters = torch.tensor(device_engram_parameters(layout, 1, rank, 0, 4),
                                      dtype=torch.int32, device='hpu')
            producer = bridge.DeviceEngramProducer(
                backend, sources[0], 0, weights.nbytes, sources[1], 0, scales.nbytes,
                rows, device_map, parameters, True, 6)
            print(f'Shard {rank}: producer compiled', flush=True)
            reference = EngramTokenHistory(layout, token_map)
            reference.reset('probe')
            history.fill_(-1)
            for token in (7, 96, 1023, 129264, 3, 29, 82, 129265, 91, 11, 47):
                raw.copy_(torch.tensor([token], dtype=torch.int32))
                batch = reference.prepare('probe', [token], image_mask=[token in (129264, 129265)])
                indices = batch.hash_ids[0, 0, rank * 6:rank * 6 + 6] - shard['row_start']
                values = torch.from_numpy(weights[indices].copy()).view(torch.float8_e4m3fn).float()
                scale = torch.exp2(torch.from_numpy(scales[indices].astype(np.int32)) - 127).repeat_interleave(32, 1)
                expected = (values * scale).to(torch.bfloat16).view(1, 6, 256)
                producer.launch(raw, history, next_history, decoded)
                assert torch.equal(decoded.cpu(), expected), (rank, token, 'decoded rows')
                reference.commit(batch, 1)
                expected_history = list(reversed(reference.history.tolist()))
                expected_history += [-1] * (3 - len(expected_history))
                assert next_history.cpu().tolist() == expected_history, (rank, token, 'history')
                history, next_history = next_history, history
            report['cases'].append(dict(rank=rank, tokens=11, rows=rows, exact=True,
                                        mapped_bytes=producer.mapped_bytes(), launches=producer.launch_count(),
                                        shared_checkpoint=True, workspace_bytes=producer.workspace_bytes()))
            producer.close()
            producer = None
            for fd in backing_fds:
                os.close(fd)
            backing_fds.clear()
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        if producer is not None:
            producer.close()
        for fd in backing_fds:
            os.close(fd)
        if dist.is_initialized():
            dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
