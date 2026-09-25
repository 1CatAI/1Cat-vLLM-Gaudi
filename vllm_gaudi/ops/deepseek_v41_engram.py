# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Engram hash layout; tokenizer normalization and primality reuse vLLM #56214.

Source: e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba,
vllm/models/deepseek_v4_1/common/engram.py. These three functions are copied
unchanged. Hash arithmetic below adapts its Triton kernel to NumPy host input.
"""

from contextlib import ExitStack
from dataclasses import dataclass
import threading

import numpy as np


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (avoids a sympy import)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize
    alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. The compressed size matters beyond bounds checking:
    every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # A private-use char, so a token that is exactly one space survives
    # Strip() instead of collapsing to the empty string and merging with
    # unrelated tokens.
    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])

    # The raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces).
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # A partial UTF-8 byte token: nothing to normalize, so key it
            # by its raw form.
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


@dataclass(frozen=True)
class EngramHashLayout:
    layer_ids: tuple[int, ...]
    rows: tuple[int, ...]
    primes: np.ndarray
    offsets: np.ndarray
    multipliers: np.ndarray
    vocab_size: int
    pad_token_id: int
    head_dim: int
    max_ngram: int
    heads: int

    @classmethod
    def from_config(cls, config: dict):
        layers = tuple(config["engram_layer_ids"])
        rows = tuple(config["engram_num_embeddings"])
        max_ngram, heads = config["engram_max_ngram_size"], config["engram_n_heads"]
        vocab_size = config["engram_compressed_vocab_size"]
        primes, seen = [], set()
        for _ in layers:
            values = []
            for _ in range(max_ngram - 1):
                current = config["engram_vocab_size"] - 1
                for _ in range(heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    values.append(current)
            primes.append(values)
        primes = np.array(primes, dtype=np.int64)
        if len(layers) != len(rows) or any(sum(values) > size for values, size in zip(primes, rows)):
            raise ValueError("Engram hash buckets exceed the declared checkpoint tables")
        offsets = np.cumsum(np.pad(primes[:, :-1], ((0, 0), (1, 0))), axis=1, dtype=np.int64)
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        multipliers = np.stack([
            np.random.default_rng(10007 * layer).integers(0, bound, size=(max_ngram, ), dtype=np.int64) * 2 + 1
            for layer in layers
        ])
        for array in (primes, offsets, multipliers):
            array.setflags(write=False)
        return cls(layers, rows, primes, offsets, multipliers, vocab_size, config["engram_pad_token_id"],
                   config["engram_head_dim"], max_ngram, heads)

    def head_shard(self, layer: int, tp_rank: int, tp_size: int = 2):
        if not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid Engram TP rank")
        sizes = self.primes[self.layer_ids.index(layer)]
        heads_per_rank = (len(sizes) + tp_size - 1) // tp_size
        first, last = tp_rank * heads_per_rank, min(len(sizes), (tp_rank + 1) * heads_per_rank)
        return {
            "head_start": first,
            "head_stop": last,
            "head_sizes": sizes[first:last].tolist(),
            "row_start": int(sizes[:first].sum()),
            "row_stop": int(sizes[:last].sum())
        }


@dataclass(frozen=True)
class EngramHashBatch:
    request_id: str
    generation: int
    start_position: int
    compressed_ids: np.ndarray
    hash_ids: np.ndarray
    active_mask: np.ndarray


class EngramTokenHistory:
    """Host-side hash preparation with explicit verify/commit ownership.

    Hash preparation never advances committed history. ``commit`` receives the
    number of *input* tokens whose KV was committed, including a target token
    when appropriate; the DSpark scheduler owns that translation. A discarded
    verification cannot leak rejected draft IDs into the next n-gram.
    """

    def __init__(self, layout: EngramHashLayout, token_map):
        self.layout = layout
        self.token_map = np.array(token_map, dtype=np.int64, copy=True)
        if (self.token_map.ndim != 1 or self.token_map.size == 0 or self.token_map.min() < 0
                or self.token_map.max() + 1 != layout.vocab_size):
            raise ValueError("Compressed tokenizer map does not match the frozen hash vocabulary")
        self.token_map.setflags(write=False)
        self.pad_id = int(self.token_map[layout.pad_token_id])
        self.lock = threading.Lock()
        self.generation = 0
        self.request_id = None
        self.position = 0
        self.history = np.empty(0, dtype=np.int64)
        self.pending = None

    def reset(self, request_id: str):
        with self.lock:
            self.request_id = request_id
            self.position = 0
            self.history = np.empty(0, dtype=np.int64)
            self.pending = None
            self.generation += 1

    def restore_prefix(self, request_id: str, token_ids, image_mask=None):
        """Rebuild hash history for an independently validated prefix hit.

        This restores only host token history, not KV or compressor state.
        The caller must restore those before publishing a cache hit. A fresh
        request owner is required so this cannot overwrite a live transaction.
        """
        with self.lock:
            if (request_id != self.request_id or self.pending is not None or self.position != 0 or self.history.size):
                raise RuntimeError("Engram prefix restore requires a fresh request owner without pending work")
            tokens = np.asarray(token_ids, dtype=np.int64)
            if tokens.ndim != 1 or (tokens.size and (tokens.min() < 0 or tokens.max() >= len(self.token_map))):
                raise ValueError("Invalid Engram prefix token IDs")
            dead = np.zeros(tokens.size, dtype=bool) if image_mask is None else np.asarray(image_mask, dtype=bool)
            if dead.shape != tokens.shape:
                raise ValueError("Image span mask does not match prefix tokens")
            tail = min(tokens.size, self.layout.max_ngram - 1)
            history = self.token_map[tokens[-tail:]].copy() if tail else np.empty(0, dtype=np.int64)
            if tail:
                history[dead[-tail:]] = -1
            self.history = history
            self.position = int(tokens.size)
            self.generation += 1

    def snapshot_prefix(self, request_id: str):
        """Keep the compressed tail, including image sentinels, at commit."""
        with self.lock:
            if self.request_id != request_id or self.pending is not None:
                raise RuntimeError("Engram checkpoint requires a committed request owner")
            return self.position, tuple(int(value) for value in self.history)

    def restore_checkpoint(self, request_id: str, checkpoint):
        with self.lock:
            if (request_id != self.request_id or self.pending is not None or self.position != 0 or self.history.size):
                raise RuntimeError("Engram checkpoint restore requires a fresh request owner")
            position, tail = checkpoint
            if (not isinstance(position, int) or not 0 <= position <= 1048576
                    or len(tail) != min(position, self.layout.max_ngram - 1)
                    or any(not isinstance(value, int) or not -1 <= value < self.layout.vocab_size for value in tail)):
                raise ValueError("Invalid committed Engram checkpoint")
            self.history = np.array(tail, dtype=np.int64)
            self.position = position
            self.generation += 1

    def prepare(self, request_id: str, token_ids, image_mask=None) -> EngramHashBatch:
        with self.lock:
            if request_id != self.request_id or self.pending is not None:
                raise RuntimeError("Engram request changed without reset, or verification is still pending")
            tokens = np.asarray(token_ids, dtype=np.int64)
            if tokens.ndim != 1 or not len(tokens) or tokens.min() < 0 or tokens.max() >= len(self.token_map):
                raise ValueError("Invalid Engram input token IDs")
            dead = np.zeros(len(tokens), dtype=bool) if image_mask is None else np.array(
                image_mask, dtype=bool, copy=True)
            if dead.shape != tokens.shape:
                raise ValueError("Image span mask does not match input tokens")
            compressed = self.token_map[tokens].copy()
            compressed[dead] = -1
            stream = np.concatenate((self.history, compressed))
            positions = np.arange(len(tokens), dtype=np.int64) + len(self.history)
            rolling = np.zeros((len(tokens), len(self.layout.layer_ids)), dtype=np.int64)
            hashes = np.empty(
                (len(tokens), len(self.layout.layer_ids), (self.layout.max_ngram - 1) * self.layout.heads),
                dtype=np.int32)
            blocked = np.zeros(len(tokens), dtype=bool)
            for shift in range(self.layout.max_ngram):
                lookback = positions - shift
                source = stream[np.maximum(lookback, 0)]
                blocked |= (lookback < 0) | (source == -1)
                values = np.where(blocked, self.pad_id, source)
                rolling ^= values[:, None] * self.layout.multipliers[None, :, shift]
                if shift:
                    first, last = (shift - 1) * self.layout.heads, shift * self.layout.heads
                    hashes[:, :, first:last] = (rolling[:, :, None] % self.layout.primes[None, :, first:last] +
                                                self.layout.offsets[None, :, first:last])
            active = ~dead
            for array in (compressed, hashes, active):
                array.setflags(write=False)
            self.pending = EngramHashBatch(request_id, self.generation, self.position, compressed, hashes, active)
            return self.pending

    def commit(self, batch: EngramHashBatch, committed_input_tokens: int):
        with self.lock:
            if batch is not self.pending or batch.generation != self.generation or batch.request_id != self.request_id:
                raise RuntimeError("Stale Engram verification cannot commit into another generation")
            if not 0 <= committed_input_tokens <= len(batch.compressed_ids):
                raise ValueError("Committed prefix is outside the prepared token block")
            combined = np.concatenate((self.history, batch.compressed_ids[:committed_input_tokens]))
            self.history = combined[-(self.layout.max_ngram - 1):].copy()
            self.position += committed_input_tokens
            self.pending = None
            self.generation += 1

    def prepare_c1(self, request_id, token_id, image, native, transfer_generation, slot, late_only=False):
        """Native hash/gather shares the ordinary history transaction owner."""
        with self.lock:
            if request_id != self.request_id or self.pending is not None:
                raise RuntimeError("Engram request changed or its previous input is still pending")
            arguments = (request_id, self.generation, transfer_generation, slot, token_id, image, self.history)
            compressed, hashes, faults = native.prepare(*arguments, True) if late_only else native.prepare(*arguments)
            active = np.array([not image], dtype=bool)
            for array in (compressed, hashes, active):
                array.setflags(write=False)
            self.pending = EngramHashBatch(request_id, self.generation, self.position, compressed, hashes, active)
            return self.pending, faults

    def discard(self, batch: EngramHashBatch):
        self.commit(batch, 0)


@dataclass(frozen=True)
class EngramHistoryBatch:
    """One gather packet containing independent request histories.

    Absolute starts follow the upstream microbatch history contract. Padding
    is a transfer concern and is never represented as committed tokens here.
    """
    members: tuple
    hash_ids: np.ndarray
    capacity: int

    @classmethod
    def prepare(cls, histories, spans, capacity):
        spans = tuple(spans)
        if not spans or len(spans) > 64 or len({item[0] for item in spans}) != len(spans):
            raise ValueError("Engram batch requires distinct request owners")
        count = sum(len(item[2]) for item in spans)
        if not count <= capacity <= 8192:
            raise ValueError("Engram batch exceeds its fixed staging capacity")
        owners = tuple(histories[item[0]] for item in spans)
        if (len(spans) > 1 and all(len(item[2]) == 1 for item in spans)
                and all(owner.layout is owners[0].layout for owner in owners)):
            return cls._prepare_decode(owners, spans, capacity)
        members = []
        try:
            for request_id, start, tokens, image_mask in spans:
                history = histories[request_id]
                if history.position != start:
                    raise RuntimeError("Engram batch absolute position differs from committed history")
                members.append((history, history.prepare(request_id, tokens, image_mask)))
        except Exception:
            # No device state has been submitted at preparation time.
            for history, batch in members:
                history.discard(batch)
            raise
        hashes = np.concatenate([batch.hash_ids for _, batch in members], axis=0)
        hashes.setflags(write=False)
        return cls(tuple(members), hashes, capacity)

    @classmethod
    def _prepare_decode(cls, owners, spans, capacity):
        """Hash independent one-token rows together, retaining each transaction.

        The request axis is vectorized; the n-gram XOR order, int64 overflow,
        image/start padding and per-head modulo are the ordinary scalar-row
        contract. No committed state changes until the existing commit path.
        """
        if len({id(owner) for owner in owners}) != len(owners):
            raise RuntimeError("Engram requests cannot share a history owner")
        layout, count = owners[0].layout, len(owners)
        with ExitStack() as locks:
            # Stable lock ordering also covers callers using a different row order.
            for owner in sorted(owners, key=id):
                locks.enter_context(owner.lock)
            compressed = np.empty(count, dtype=np.int64)
            active = np.empty(count, dtype=bool)
            padding = np.empty(count, dtype=np.int64)
            lookback = np.full((layout.max_ngram, count), -1, dtype=np.int64)
            for row, (owner, span) in enumerate(zip(owners, spans)):
                request_id, start, tokens, image_mask = span
                if owner.position != start:
                    raise RuntimeError("Engram batch absolute position differs from committed history")
                if owner.request_id != request_id or owner.pending is not None:
                    raise RuntimeError("Engram request changed or its previous input is still pending")
                token = np.asarray(tokens, dtype=np.int64)
                if token.shape != (1, ) or not 0 <= token[0] < len(owner.token_map):
                    raise ValueError("Invalid Engram input token IDs")
                dead = np.zeros(1, dtype=bool) if image_mask is None else np.asarray(image_mask, dtype=bool)
                if dead.shape != (1, ):
                    raise ValueError("Image span mask does not match input tokens")
                compressed[row] = -1 if dead[0] else owner.token_map[token[0]]
                active[row], padding[row] = not dead[0], owner.pad_id
                lookback[0, row] = compressed[row]
                tail = owner.history[-(layout.max_ngram - 1):]
                lookback[1:len(tail) + 1, row] = tail[::-1]
            rolling = np.zeros((count, len(layout.layer_ids)), dtype=np.int64)
            hashes = np.empty((count, len(layout.layer_ids), (layout.max_ngram - 1) * layout.heads), dtype=np.int32)
            blocked = np.zeros(count, dtype=bool)
            for shift in range(layout.max_ngram):
                blocked |= lookback[shift] == -1
                values = np.where(blocked, padding, lookback[shift])
                rolling ^= values[:, None] * layout.multipliers[None, :, shift]
                if shift:
                    first, last = (shift - 1) * layout.heads, shift * layout.heads
                    hashes[:, :, first:last] = (rolling[:, :, None] % layout.primes[None, :, first:last] +
                                                layout.offsets[None, :, first:last])
            for array in (compressed, hashes, active):
                array.setflags(write=False)
            members = []
            for row, (owner, span) in enumerate(zip(owners, spans)):
                selection = slice(row, row + 1)
                batch = EngramHashBatch(span[0], owner.generation, owner.position, compressed[selection],
                                        hashes[selection], active[selection])
                members.append((owner, batch))
            for owner, batch in members:
                owner.pending = batch
            return cls(tuple(members), hashes, capacity)

    def commit(self, counts):
        counts = tuple(counts)
        if len(counts) != len(self.members):
            raise ValueError("Engram batch commit must account for every request")
        # Validate every owner before changing any committed history.
        for (history, batch), count in zip(self.members, counts):
            if (batch is not history.pending or batch.generation != history.generation
                    or batch.request_id != history.request_id):
                raise RuntimeError("Engram batch completion contains a stale request generation")
            if type(count) is not int or not 0 <= count <= len(batch.compressed_ids):
                raise ValueError("Engram batch commit exceeds a prepared request interval")
        for (history, batch), count in zip(self.members, counts):
            history.commit(batch, count)
