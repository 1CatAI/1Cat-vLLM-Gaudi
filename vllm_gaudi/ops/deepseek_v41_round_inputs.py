# SPDX-License-Identifier: Apache-2.0
"""Device-owned C6 input progression, including speculative history rollback."""

import torch

from vllm_gaudi.ops.deepseek_v41_verify import DRAFT_START, OUTPUT_START, STATUS


def advance_round(record, control, histories, ids, positions, next_control, history):
    """Publish the next anchor/C5 without reading acceptance on the host.

    ``histories[k]`` is the compressed lookback after committing exactly k
    target inputs. The target writes six speculative rows, but the next
    round must start from the accepted prefix, rather than from row six.
    Control has the same seven metadata and five proposals as verification.
    This function deliberately leaves EOS handling to the ordinary output
    consumer; one extra queued round may be discarded there.
    """
    committed = record[1:2].clamp(1, 6)
    output_count = record[2:3].clamp(1, 6)
    anchor = record[OUTPUT_START:DRAFT_START].gather(0, output_count - 1)
    proposals = record[DRAFT_START:STATUS]
    ids.copy_(torch.cat((anchor, proposals)).to(ids.dtype))
    start = control[4:5] + committed
    positions.copy_(start.to(torch.int32) + torch.arange(6, dtype=torch.int32, device=record.device))
    remaining = control[3:4] - record[2:3]
    next_control.copy_(torch.cat((control[:1] + 1, torch.full_like(committed, 6),
                                 torch.full_like(committed, 5), remaining, start,
                                 control[5:7], proposals)))
    history.copy_(histories.gather(0, committed.reshape(1, 1).expand(1, 3)).reshape(3))
    return ids, positions, next_control, history


class DeviceRoundInputs:
    """Two immutable control frames, with one request-owned device cursor."""

    def __init__(self, ids, positions, *, backend="hpu_backend", frames=2):
        if ids.shape != (6,) or positions.shape != (6,):
            raise ValueError("Device speculative inputs require the native C6 bucket")
        if frames not in (2, 4):
            raise ValueError("Device speculative controls need two or four retained frames")
        self.ids, self.positions = ids, positions
        self.controls = [torch.empty(12, dtype=torch.int64, device=ids.device) for _ in range(frames)]
        self.history = torch.empty(3, dtype=torch.int32, device=ids.device)
        self.parity = 0
        self.owner = None
        self.advance = (torch.compile(advance_round, backend=backend, fullgraph=True, dynamic=False)
                        if ids.device.type == "hpu" else advance_round)

    @property
    def control(self):
        return self.controls[self.parity]

    def seed(self, owner, tokens, start, remaining, limit, generation, lookback):
        if self.owner is not None:
            raise RuntimeError("Device speculative cursor must retire its previous request before reseeding")
        if len(tokens) != 6 or len(lookback) != 3 or not 0 <= start <= limit - 6:
            raise ValueError("Device speculative seed exceeds its request/context geometry")
        self.owner = owner
        self.parity = 0
        # Request admission is the only upload of the cursor and lookback.
        # Subsequent rounds use the accepted device count and record only.
        self.ids.copy_(torch.tensor(tokens, dtype=self.ids.dtype))
        self.positions.copy_(torch.arange(start, start + 6, dtype=torch.int32))
        self.control.copy_(torch.tensor([generation, 6, 5, remaining, start, limit, 1, *tokens[1:]],
                                       dtype=torch.int64))
        self.history.copy_(torch.tensor(lookback, dtype=torch.int32))

    def next(self, record, histories, *, published=False):
        if self.owner is None:
            raise RuntimeError("Device speculative cursor has no request owner")
        next_parity = (self.parity + 1) % len(self.controls)
        if not published:
            self.advance(record, self.control, histories, self.ids, self.positions,
                         self.controls[next_parity], self.history)
        self.parity = next_parity

    def retire(self, owner):
        if self.owner != owner:
            raise RuntimeError("Cannot retire another request's speculative cursor")
        self.owner = None


class RoundInputPublication(torch.nn.Module):
    """Capture next-round coordinates and Engram staging in the control plan.

    The selected parity owns its successor control allocation. Request RNG
    and proposal buffers remain request-owned; this module never substitutes
    persistent graph buffers for them. The existing repair entry deliberately
    does not use publication, because its host generation may skip tickets.
    """

    def __init__(self, cursor, engram, parity):
        super().__init__()
        if not engram.batched or not 0 <= parity < len(cursor.controls):
            raise ValueError("Native round publication needs batched Engram and a retained control parity")
        for name, value in (
            ("ids", cursor.ids), ("positions", cursor.positions),
            ("next_control", cursor.controls[(parity + 1) % len(cursor.controls)]),
            ("history", cursor.history), ("histories", engram.histories),
            ("engram_ids", engram.ids), ("engram_history", engram.initial_history),
        ):
            self.register_buffer(name, value, persistent=False)
        self.parity = parity

    def forward(self, record, control):
        advance_round(record, control, self.histories, self.ids, self.positions,
                      self.next_control, self.history)
        self.engram_ids.copy_(self.ids.to(torch.int32))
        self.engram_history.copy_(self.history)
