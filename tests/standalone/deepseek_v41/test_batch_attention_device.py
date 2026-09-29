# SPDX-License-Identifier: Apache-2.0
from types import FunctionType

from test_native_moe import HPU, torch

from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_packed_mla, write_state_rows
from vllm_gaudi.ops.deepseek_v41_math import pack_fp4, pack_swa, unpack_fp4, unpack_swa


@HPU
def test_batched_packed_mla_mutations_pages_padding_match_independent_attention():
    torch.manual_seed(1598)
    op = torch.ops.custom_op.custom_deepseek_v41_selected_mla_mme_gaudi2
    for batch, ratio in ((2, 0), (8, 1), (16, 2), (64, 1)):
        capacity, heads = 64, 32
        # Share physical main storage, but no request shares a writable page.
        swa_host = pack_swa(torch.randn(capacity * 256, 512).bfloat16())
        swa = swa_host.to("hpu")
        slots_host = torch.randperm(capacity)[:batch].int()
        slots_host[::5] = -1
        positions_host = torch.tensor([(0, 127, 255, 256, 511, 512, 999)[i % 7] for i in range(batch)],
                                      dtype=torch.int32)
        positions_host[slots_host < 0] = -1
        pages_host = torch.stack([torch.arange(8) + 1 + int(max(0, slot)) * 8 for slot in slots_host]).int()
        # Pool has one reserved null page plus capacity*8 request pages.
        main_host = pack_fp4(torch.randn((capacity * 8 + 1) * 128, 512).bfloat16(), 16)
        main = main_host.to("hpu")
        selected_host = torch.arange(512).int().expand(batch, -1).clone()
        selected_host[:, 33::9] = -1
        selected_host[slots_host < 0] = -1
        positions, slots, pages, selected = [
            x.to("hpu") for x in (positions_host, slots_host, pages_host, selected_host)
        ]
        query = torch.randn(batch, heads, 512).bfloat16().to("hpu")
        values = torch.randn(batch, 512).bfloat16().to("hpu")
        sink = torch.randn(heads).float().to("hpu")
        scale = torch.tensor([512**-0.5], device="hpu")
        done = torch.zeros(batch, device="hpu", dtype=torch.int32)

        def chain(q, cache, main_cache, kv, ids, page, pos, slot, bias, scaling, main_ready, ratio=ratio):
            packed = pack_swa(kv)
            rows = torch.where(slot >= 0, slot * 256 + pos.remainder(256), -1).int()
            write_done = write_state_rows(cache, packed, rows)
            return batch_packed_mla(q,
                                    cache,
                                    main_cache,
                                    ids,
                                    page,
                                    pos,
                                    slot,
                                    bias,
                                    scaling,
                                    write_done,
                                    main_ready,
                                    ratio=ratio)

        entry = FunctionType(chain.__code__.replace(co_name=f"batch_mla_{batch}_{ratio}"),
                             chain.__globals__,
                             argdefs=chain.__defaults__,
                             closure=chain.__closure__)
        compiled = torch.compile(entry, backend="hpu_backend", fullgraph=True, dynamic=False)
        for change in range(2):
            if change:
                values.copy_(torch.randn(batch, 512).bfloat16())
                query.copy_(torch.randn(batch, heads, 512).bfloat16())
            actual = compiled(query, swa, main, values, selected, pages, positions, slots, sink, scale, done)
            torch.hpu.synchronize()
            packed = pack_swa(values.cpu())
            for i in range(batch):
                slot, pos = int(slots_host[i]), int(positions_host[i])
                if slot < 0:
                    assert not actual[i].cpu().any(), (batch, ratio, i, "padding")
                    continue
                swa_host[slot * 256 + pos % 256] = packed[i]
                window = torch.arange(pos - 127, pos + 1)
                sw = window.remainder(256) + slot * 256
                cache = unpack_swa(swa_host[sw])
                ids = torch.where(window >= 0, torch.arange(128), -1).int()
                if ratio:
                    logical = selected_host[i]
                    physical = pages_host[i][logical.clamp_min(0) // (128 // ratio)] * (128 // ratio)
                    physical += logical.clamp_min(0).remainder(128 // ratio)
                    cache = torch.cat((cache, unpack_fp4(main_host[physical])), 0)
                    ids = torch.cat(
                        (ids, torch.where((logical >= 0) & (logical < (pos + 1) // ratio),
                                          torch.arange(512) + 128, -1)), 0)
                expected = op(query[i:i + 1], cache.to("hpu"),
                              ids.int().reshape(1, -1).to("hpu"), sink, scale,
                              torch.tensor([ids.numel()], device="hpu", dtype=torch.int32))
                assert torch.equal(actual[i:i + 1].cpu(), expected.cpu()), (batch, ratio, i, change)
            assert torch.equal(swa.cpu(), swa_host), (batch, ratio, "state")
