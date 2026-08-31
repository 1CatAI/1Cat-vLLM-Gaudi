import argparse
import statistics
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128
EPSILON = 1e-6


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + EPSILON)


def expand_then_normalize(q: torch.Tensor, k: torch.Tensor):
    repeat = VALUE_HEADS // QK_HEADS
    q = q.repeat_interleave(repeat, dim=2)
    k = k.repeat_interleave(repeat, dim=2)
    return _l2norm(q).to(torch.bfloat16), _l2norm(k).to(torch.bfloat16)


def normalize_then_expand(q: torch.Tensor, k: torch.Tensor):
    repeat = VALUE_HEADS // QK_HEADS
    q = _l2norm(q).to(torch.bfloat16)
    k = _l2norm(k).to(torch.bfloat16)
    return (
        q.repeat_interleave(repeat, dim=2),
        k.repeat_interleave(repeat, dim=2),
    )


def benchmark(name, function, inputs, warmups, iterations):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(warmups):
        output = function(*inputs)
    torch.hpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)

    median = statistics.median(samples)
    print(
        f"{name} median_ms={median:.6f} min_ms={min(samples):.6f} "
        f"samples_ms={samples}",
        flush=True,
    )
    return tuple(tensor.clone() for tensor in output), median


def compare(reference, candidate):
    for name, expected, actual in zip(("q", "k"), reference, candidate):
        difference = actual.float() - expected.float()
        relative_l2 = torch.linalg.vector_norm(difference)
        relative_l2 /= torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        print(
            f"quality_{name} max_abs={float(difference.abs().max().cpu()):.9e} "
            f"equal_fraction={float((actual == expected).float().mean().cpu()):.9f} "
            f"relative_l2={float(relative_l2.cpu()):.9e}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=21)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    q = torch.randn(
        (1, TOKENS, QK_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="hpu",
    )
    k = torch.randn_like(q)
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    old = torch.compile(expand_then_normalize, **compile_args)
    new = torch.compile(normalize_then_expand, **compile_args)
    print(
        f"shape={tuple(q.shape)} repeat={VALUE_HEADS // QK_HEADS} "
        f"compile_args={compile_args}",
        flush=True,
    )

    reference, old_ms = benchmark(
        "expand_then_normalize",
        old,
        (q, k),
        args.warmups,
        args.iterations,
    )
    candidate, new_ms = benchmark(
        "normalize_then_expand",
        new,
        (q, k),
        args.warmups,
        args.iterations,
    )
    compare(reference, candidate)
    print(
        f"speedup={old_ms / new_ms:.6f} saved_ms={old_ms - new_ms:.6f} "
        f"projected_48_layer_saved_ms={(old_ms - new_ms) * 48:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
