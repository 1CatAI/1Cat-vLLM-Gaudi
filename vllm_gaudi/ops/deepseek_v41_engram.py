# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Engram hash layout; tokenizer normalization and primality reuse vLLM #56214.

Source: e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba,
vllm/models/deepseek_v4_1/common/engram.py. These three functions are copied
unchanged. Hash arithmetic below adapts its Triton kernel to NumPy host input.
"""

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
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

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
        multipliers = np.stack([np.random.default_rng(10007 * layer).integers(
            0, bound, size=(max_ngram,), dtype=np.int64) * 2 + 1 for layer in layers])
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
        return {"head_start": first, "head_stop": last, "head_sizes": sizes[first:last].tolist(),
                "row_start": int(sizes[:first].sum()), "row_stop": int(sizes[:last].sum())}


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

    def prepare(self, request_id: str, token_ids, image_mask=None) -> EngramHashBatch:
        with self.lock:
            if request_id != self.request_id or self.pending is not None:
                raise RuntimeError("Engram request changed without reset, or verification is still pending")
            tokens = np.asarray(token_ids, dtype=np.int64)
            if tokens.ndim != 1 or not len(tokens) or tokens.min() < 0 or tokens.max() >= len(self.token_map):
                raise ValueError("Invalid Engram input token IDs")
            dead = np.zeros(len(tokens), dtype=bool) if image_mask is None else np.array(image_mask, dtype=bool, copy=True)
            if dead.shape != tokens.shape:
                raise ValueError("Image span mask does not match input tokens")
            compressed = self.token_map[tokens].copy()
            compressed[dead] = -1
            stream = np.concatenate((self.history, compressed))
            positions = np.arange(len(tokens), dtype=np.int64) + len(self.history)
            rolling = np.zeros((len(tokens), len(self.layout.layer_ids)), dtype=np.int64)
            hashes = np.empty((len(tokens), len(self.layout.layer_ids), (self.layout.max_ngram - 1) * self.layout.heads),
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
                    hashes[:, :, first:last] = (rolling[:, :, None] % self.layout.primes[None, :, first:last]
                                               + self.layout.offsets[None, :, first:last])
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

    def discard(self, batch: EngramHashBatch):
        self.commit(batch, 0)
