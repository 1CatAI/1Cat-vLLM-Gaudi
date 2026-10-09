# SPDX-License-Identifier: Apache-2.0
"""Bounded C6 write journal for asynchronous official-sampling repair.

The next Target can overwrite the current prefix's rings and shared caches.
Keep only the six physical rows it can touch. Restoring the draft protocol
alone is insufficient: its following Target has already changed those rows.
No host position read or physical-prefix assumption is used here.
"""

import torch


class C6WriteJournal(torch.nn.Module):
    """Fixed-size device snapshots of the shared Target's mutable write set."""

    def __init__(self, program):
        super().__init__()
        from vllm_gaudi import envs
        from vllm_gaudi.ops.deepseek_v41_replay import stage_state_tensors

        self.batch_direct = envs.VLLM_HPU_DSV41_DSPARK_JOURNAL_BATCH_DIRECT
        self.batch_copy = envs.VLLM_HPU_DSV41_DSPARK_JOURNAL_BATCH
        self.native_copy = envs.VLLM_HPU_DSV41_DSPARK_JOURNAL_COPY
        self.cache_coordinates = envs.VLLM_HPU_DSV41_DSPARK_JOURNAL_COORDINATES
        active = {id(value) for value in stage_state_tensors(program)}
        self.specs, self.names = [], []
        seen = set()
        self.pages = program.shared.block_table

        def add(name, value, mode, parameter):
            if value is None or id(value) not in active or id(value) in seen:
                return
            if value.ndim < 1 or value.shape[0] < 6:
                raise ValueError(f"C6 repair requires six addressable state rows: {name}")
            index = len(self.specs)
            self.register_buffer(f"target_{index}", value, persistent=False)
            self.register_buffer(
                f"indices_{index}",
                torch.empty(6, dtype=torch.int32 if self.native_copy else torch.int64, device=value.device),
                persistent=False
            )
            self.register_buffer(
                f"saved_{index}",
                torch.empty((6, *value.shape[1:]), dtype=value.dtype, device=value.device),
                persistent=False,
            )
            self.specs.append((mode, parameter))
            self.names.append(name)
            seen.add(id(value))

        for layer in program.layers:
            attention = layer.attention
            if getattr(attention, "dspark_swa_cache", False):
                add(f"layers.{layer.layer}.attention.swa_decoded.swa", attention.swa_decoded.swa, "ring", 256)
            for name in ("swa", "kv_history", "score_history"):
                value = getattr(attention, name, None)
                if value is not None:
                    add(f"layers.{layer.layer}.attention.{name}", value, "ring", value.shape[0])
        for source, cache in program.shared.sources.items():
            if 128 % cache.ratio:
                raise ValueError("Compressed pages must divide the physical page size")
            for name in ("main", "index"):
                add(f"shared.sources.{source}.{name}", getattr(cache, name), "paged", cache.ratio)
            for name in ("decoded_main", "decoded_index_hot", "index_mirror", "main_mirror"):
                add(f"shared.sources.{source}.{name}", getattr(cache, name, None), "logical", cache.ratio)
        for source, selection in program.shared.topk.items():
            add(f"shared.topk.{source}.indices", selection.indices, "prefix", 0)
        add("shared.candidate_pool", program.shared.candidate_pool, "prefix", 0)
        # Page bindings are immutable over the one-round lookahead. An
        # unaccounted mutable allocation must stop qualification, not silently
        # escape rollback. Inactive mirrors are excluded by the stage itself.
        missing = active - seen - {id(self.pages)}
        if missing:
            names = [name for name, value in program.named_buffers() if id(value) in missing]
            raise ValueError(f"Unaccounted C6 repair state: {names}")
        self.bytes = sum(
            getattr(self, f"saved_{i}").numel() * getattr(self, f"saved_{i}").element_size()
            for i in range(len(self.specs))
        )

        if self.batch_copy or self.batch_direct:
            self.prepare_batch_copy()

    def prepare_batch_copy(self):
        """Bind immutable coordinate modes once, grouping equal-width snapshots."""
        if getattr(self, "batch_groups", None) is not None:
            return
        if not self.native_copy:
            raise ValueError("Batched repair requires the original byte-preserving native journal")
        order = sorted(range(len(self.specs)),
                       key=lambda i: getattr(self, f"target_{i}")[0].numel()
                       * getattr(self, f"target_{i}").element_size())
        self.batch_groups = []
        capacity = 4 if self.batch_direct else 8
        for start in range(0, len(order), capacity):
            group = tuple(order[start:start + capacity])
            records = []
            for index in group:
                mode, parameter = self.specs[index]
                if mode != "prefix" and (parameter <= 0 or parameter & (parameter - 1)
                                         or mode == "paged" and parameter > 128):
                    raise ValueError("Batched journal requires qualified power-of-two coordinates")
                records.append(({"ring": 0, "paged": 1, "logical": 2, "prefix": 3}[mode],
                                parameter.bit_length() - 1 if mode != "prefix" else 0))
            records.extend([(-1, 0)] * (capacity - len(group)))
            number = len(self.batch_groups)
            self.register_buffer(f"batch_modes_{number}",
                                 torch.tensor(records, dtype=torch.int32, device=self.pages.device),
                                 persistent=False)
            self.batch_groups.append(group)

    def _batch_forward(self, positions):
        if getattr(self, "batch_groups", None) is None:
            raise RuntimeError("Prepare journal coordinate constants before capture")
        logical = positions.to(torch.int32).contiguous()
        for number, group in enumerate(self.batch_groups):
            if self.batch_direct:
                sources = [getattr(self, f"target_{index}").reshape(
                    getattr(self, f"target_{index}").shape[0], -1) for index in group]
                sources.extend([sources[0]] * (4 - len(group)))
                outputs = torch.ops.custom_op.custom_deepseek_v41_journal_batch_direct_gaudi2(
                    *sources, logical, self.pages, getattr(self, f"batch_modes_{number}"))
                for slot, index in enumerate(group):
                    saved = getattr(self, f"saved_{index}")
                    saved.copy_(outputs[slot].reshape(saved.shape))
                    getattr(self, f"indices_{index}").copy_(outputs[slot + 4])
            else:
                sources = [getattr(self, f"target_{index}").view(torch.uint8).reshape(
                    getattr(self, f"target_{index}").shape[0], -1) for index in group]
                sources.extend([sources[0]] * (8 - len(group)))
                packed, rows = torch.ops.custom_op.custom_deepseek_v41_journal_batch_gaudi2(
                    *sources, logical, self.pages, getattr(self, f"batch_modes_{number}"))
                for slot, index in enumerate(group):
                    saved = getattr(self, f"saved_{index}")
                    columns = saved[0].numel() * saved.element_size()
                    saved.copy_(packed[slot, :, :columns].contiguous().view(saved.dtype).reshape(saved.shape))
                    getattr(self, f"indices_{index}").copy_(rows[slot])
        return positions.clone()

    def forward(self, positions):
        """Capture after this Target, immediately before the lookahead Target."""
        if positions.shape != (6,):
            raise ValueError("C6 repair needs the exact six next Target positions")
        if self.batch_copy or self.batch_direct:
            return self._batch_forward(positions)
        logical_i32 = positions.to(torch.int32).contiguous() if self.native_copy else None
        logical = positions.long()
        coordinates = {}
        for index, (mode, parameter) in enumerate(self.specs):
            target = getattr(self, f"target_{index}")
            if self.native_copy and target.element_size() in (1, 2, 4):
                mode_id = {"ring": 0, "paged": 1, "logical": 2, "prefix": 3}[mode]
                saved, rows = torch.ops.custom_op.custom_deepseek_v41_journal_copy_gaudi2(
                    target, logical_i32, self.pages, mode_id, parameter)
                getattr(self, f"indices_{index}").copy_(rows)
                getattr(self, f"saved_{index}").copy_(saved)
                continue
            key = (mode, parameter, target.shape[0])
            if self.cache_coordinates and key in coordinates:
                rows = coordinates[key]
            elif mode == "ring":
                rows = logical.remainder(parameter)
            elif mode == "paged":
                physical_page = self.pages.gather(0, logical.div(128, rounding_mode="floor")).long()
                rows = physical_page * (128 // parameter) + logical.remainder(128).div(parameter, rounding_mode="floor")
            elif mode == "logical":
                rows = logical.div(parameter, rounding_mode="floor")
            else:
                rows = torch.arange(6, dtype=torch.int64, device=positions.device)
            # Bounded mirrors can remain registered outside their active
            # search bucket. Their producers use the same capacity predicate.
            # A masked gather is safe, and restores row zero to its own value.
            if not self.cache_coordinates or key not in coordinates:
                rows = torch.where((rows >= 0) & (rows < target.shape[0]), rows, 0)
                if self.cache_coordinates:
                    coordinates[key] = rows
            getattr(self, f"indices_{index}").copy_(rows)
            getattr(self, f"saved_{index}").copy_(target.index_select(0, rows))
        return positions.clone()

    def restore(self):
        for index in range(len(self.specs)):
            # Repeated compressed indices all save the same original bytes;
            # no summation or ordering-sensitive accumulation is involved.
            getattr(self, f"target_{index}").index_copy_(
                0, getattr(self, f"indices_{index}").long(), getattr(self, f"saved_{index}")
            )


class C6LookaheadJournal(torch.nn.Module):
    """Protect the union of two possible following C6 writes before either runs.

    A second round starts one to six rows after the first. Twelve consecutive
    logical positions cover both; snapshots happen at the same prior state,
    so repeated ring/compressed addresses restore identical original bytes.
    """

    def __init__(self, program):
        super().__init__()
        self.first = C6WriteJournal(program)
        self.second = C6WriteJournal(program)
        self.bytes = self.first.bytes + self.second.bytes

    def forward(self, positions):
        self.first(positions)
        self.second(positions + 6)
        return positions.clone()

    def restore(self):
        self.first.restore()
        self.second.restore()


class SampledRoundRepairFrame(torch.nn.Module):
    """One retained official draw, draft state and following Target journal.

    Each cursor parity owns its own frame. Its native protocol captures the
    copies along with verification and drafting; no host event/pinned control
    transfer is inserted on the covered path.
    """

    def __init__(self, program, cursor, engram, prototype, parity, *, lookahead=1):
        super().__init__()
        if len(prototype) != 10 or not 0 <= parity < len(cursor.controls) or lookahead not in (1, 2):
            raise ValueError("Sampled C6 repair requires ten protocol inputs and a fixed cursor parity")
        from vllm_gaudi import envs

        if envs.VLLM_HPU_DSV41_DSPARK_COVERAGE_AUDIT:
            self.register_buffer(
                "coverage_flags", torch.zeros(2, dtype=torch.bool, device=prototype[0].device), persistent=False
            )
        self.journal = C6WriteJournal(program) if lookahead == 1 else C6LookaheadJournal(program)
        self.parity = parity
        for index, value in enumerate(prototype):
            self.register_buffer(f"payload_{index}", torch.empty_like(value), persistent=False)
        for index, layer in enumerate(program.draft.layers):
            self.register_buffer(f"draft_source_{index}", layer.attention.swa, persistent=False)
            self.register_buffer(f"draft_saved_{index}", torch.empty_like(layer.attention.swa), persistent=False)
        self.draft_layers = len(program.draft.layers)
        for index, value in enumerate(
            (cursor.ids, cursor.positions, cursor.controls[parity], cursor.history, engram.histories)
        ):
            self.register_buffer(f"cursor_source_{index}", value, persistent=False)
            self.register_buffer(f"cursor_saved_{index}", torch.empty_like(value), persistent=False)

    @property
    def payload(self):
        return tuple(getattr(self, f"payload_{index}") for index in range(10))

    def protocol_states(self, *, exact_repair):
        """Return allocations actually owned by the selected native plan.

        Exact repair receives the saved payload through its dynamic inputs.
        It also preserves the bounded plan's coverage certificate for the
        host audit, rather than writing a new certificate. Neither allocation
        is part of the exact repair's captured tensor program.
        """
        values = []
        for name, value in self.named_buffers():
            if exact_repair and (name.startswith("payload_") or name == "coverage_flags"):
                continue
            prefix, _, leaf = name.rpartition(".")
            if leaf.startswith("batch_modes_"):
                owner = self.get_submodule(prefix)
                if exact_repair or not (owner.batch_copy or owner.batch_direct):
                    continue
            values.append(value)
        return tuple(values)

    def capture_inputs(self, *values):
        for index, value in enumerate(values):
            getattr(self, f"payload_{index}").copy_(value)
        for index in range(self.draft_layers):
            getattr(self, f"draft_saved_{index}").copy_(getattr(self, f"draft_source_{index}"))
        for index in range(5):
            getattr(self, f"cursor_saved_{index}").copy_(getattr(self, f"cursor_source_{index}"))

    def restore_protocol(self):
        for index in range(self.draft_layers):
            getattr(self, f"draft_source_{index}").copy_(getattr(self, f"draft_saved_{index}"))
        for index in range(5):
            getattr(self, f"cursor_source_{index}").copy_(getattr(self, f"cursor_saved_{index}"))
