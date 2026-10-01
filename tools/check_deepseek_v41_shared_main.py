# SPDX-License-Identifier: Apache-2.0
"""Untimed C1 native contract gate; called before the real continuation fixture."""
import torch


def check_shared_main_contract():
    ops = torch.ops.custom_op
    checks = []
    generator = torch.Generator().manual_seed(41029)
    for ratio in (1, 2):
        def parent(q0, q1, swa0, swa1, main, selected, positions, pages, sink, scale, lengths, ratio=ratio):
            a = ops.custom_deepseek_v41_logical_mla_gaudi2(
                q0, swa0, main, selected, positions, pages, sink, scale, lengths, ratio)
            b = ops.custom_deepseek_v41_logical_mla_gaudi2(
                q1, swa1, main, selected, positions, pages, sink, scale, lengths, ratio)
            return a, b

        def candidate(q0, q1, swa0, swa1, main, selected, positions, pages, sink, scale, lengths, ratio=ratio):
            a, rows, mask = ops.custom_deepseek_v41_main_publish_mla_gaudi2(
                q0, swa0, main, selected, positions, pages, sink, scale, lengths, ratio)
            b = ops.custom_deepseek_v41_main_reuse_mla_gaudi2(
                q1, swa1, rows, mask, positions, sink, scale, lengths)
            return a, b

        functions = [torch.compile(fn, backend='hpu_backend', fullgraph=True, dynamic=False)
                     for fn in (parent, candidate)]
        for case in ('duplicates', 'negative_page', 'outside_pool', 'outside_pages', 'all_invalid',
                     'early_window', 'zero_length', 'partial_length'):
            query = [torch.randn(1, 16, 512, generator=generator).to(torch.bfloat16) / 32 for _ in range(2)]
            swa = [torch.randint(0, 127, (256, 528), generator=generator, dtype=torch.uint8) for _ in range(2)]
            for ring in swa:
                ring[:, 512:] = 127
            main = torch.randint(0, 256, (1024, 288), generator=generator, dtype=torch.uint8)
            main[:, 256:] = torch.randint(0, 119, (1024, 32), generator=generator, dtype=torch.uint8)
            selected = torch.randint(0, 512, (1, 512), generator=generator, dtype=torch.int32)
            selected[:, ::7] = -1
            pages = torch.full((8192,), -1, dtype=torch.int32)
            pages[:8] = torch.arange(8, dtype=torch.int32).flip(0)
            if case == 'negative_page':
                pages.fill_(-1)
            elif case == 'outside_pool':
                pages.fill_(1000)
            elif case == 'outside_pages':
                selected.fill_(8192 * (128 // ratio))
            elif case == 'all_invalid':
                selected.fill_(-1)
            positions = torch.tensor([0 if case == 'early_window' else 16384], dtype=torch.int32)
            lengths = torch.tensor([0 if case == 'zero_length' else 639 if case == 'partial_length' else 640],
                                   dtype=torch.int32)
            args = tuple(x.to('hpu') for x in (*query, *swa, main, selected, positions, pages,
                                              torch.zeros(16), torch.tensor([512**-0.5]), lengths))
            expected = [x.cpu().view(torch.int16) for x in functions[0](*args)]
            observed = [x.cpu().view(torch.int16) for x in functions[1](*args)]
            assert all(torch.equal(a, b) for a, b in zip(expected, observed, strict=True)), (ratio, case)
            checks.append(dict(ratio=ratio, case=case, exact_bits=True))
    return checks
