# SPDX-License-Identifier: Apache-2.0
"""Request-slot writes must not mutate padding, neighbours or stale slots."""
from types import FunctionType

from test_native_moe import HPU, torch

pytestmark = HPU


def test_state_rows_mutation_read_and_replay():
    write = torch.ops.custom_op.custom_deepseek_v41_state_rows_write_gaudi2
    read = torch.ops.custom_op.custom_deepseek_v41_state_rows_read_gaudi2

    def chain(cache, values, rows, read_rows):
        done = write(cache, values, rows)
        # The next compressor operation reads a pair of history rows. Retain
        # the producer completion even when that row was written previously.
        return read(cache, read_rows, done)

    for dtype, widths in ((torch.uint8, (68, 288, 528)), (torch.float32, (512, )), (torch.bfloat16, (512, ))):
        for batch in (1, 2, 8, 64):
            for width in widths:
                host = torch.arange(79 * width).reshape(79, width).remainder(127).to(dtype)
                cache = host.to("hpu")
                ids = torch.randperm(77)[:batch].to(torch.int32)
                ids[::3] = -1
                if batch > 1:
                    ids[-1] = 79  # malformed row is also a no-write
                rows = ids.to("hpu")
                read_ids = ids.clamp(-1, 78)
                read_rows = read_ids.to("hpu")
                values = torch.randint(0, 127, (batch, width)).to(dtype).to("hpu")
                entry = FunctionType(chain.__code__.replace(co_name=f"state_rows_{dtype}_{batch}_{width}"),
                                     chain.__globals__,
                                     closure=chain.__closure__)
                compiled = torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
                for iteration in range(3):
                    incoming = torch.randint(0, 127, (batch, width)).to(dtype)
                    values.copy_(incoming)
                    for i, row in enumerate(ids.tolist()):
                        if 0 <= row < 79:
                            host[row] = incoming[i]
                    expected = torch.stack([
                        host[max(0, int(row))] if 0 <= int(ids[i]) < 79 else torch.zeros(width, dtype=dtype)
                        for i, row in enumerate(read_ids)
                    ])
                    result = compiled(cache, values, rows, read_rows)
                    torch.hpu.synchronize()
                    assert torch.equal(result.cpu(), expected), (dtype, batch, width, iteration)
                    assert torch.equal(cache.cpu(), host), (dtype, batch, width, iteration, "cache")
