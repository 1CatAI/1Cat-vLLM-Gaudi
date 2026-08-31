import argparse
import statistics
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


BATCH = 12_288
CHUNK_SIZE = 64
HEAD_DIM = 128


def rhs_baseline(matrix, first, second):
    return torch.bmm(matrix, first), torch.bmm(matrix, second)


def rhs_fused(matrix, first, second):
    combined = torch.bmm(matrix, torch.cat((first, second), dim=-1))
    return combined.split(HEAD_DIM, dim=-1)


def left_baseline(first, second, matrix):
    return (
        torch.bmm(first.transpose(1, 2), matrix),
        torch.bmm(second.transpose(1, 2), matrix),
    )


def left_fused(first, second, matrix):
    combined_left = torch.cat(
        (first.transpose(1, 2), second.transpose(1, 2)),
        dim=1,
    )
    combined = torch.bmm(combined_left, matrix)
    return combined.split(HEAD_DIM, dim=1)


def benchmark(name, function, inputs, iterations):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(3):
        output = function(*inputs)
    torch.hpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000)
    print(
        f"{name} median_ms={statistics.median(samples):.6f} "
        f"min_ms={min(samples):.6f} samples_ms={samples}",
        flush=True,
    )
    return output


def check_equal(actual, expected, name):
    for index, (actual_tensor, expected_tensor) in enumerate(
        zip(actual, expected, strict=True)
    ):
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        relative_l2 = torch.linalg.vector_norm(
            actual_tensor.float() - expected_tensor.float()
        )
        relative_l2 /= torch.linalg.vector_norm(
            expected_tensor.float()
        ).clamp_min(1e-12)
        print(
            f"{name}_{index}_max_abs={float(difference.max().cpu()):.9e} "
            f"relative_l2={float(relative_l2.cpu()):.9e} "
            f"equal_fraction="
            f"{float((actual_tensor == expected_tensor).float().mean().cpu()):.9f}",
            flush=True,
        )
        torch.testing.assert_close(
            actual_tensor[::1536].cpu(),
            expected_tensor[::1536].cpu(),
            rtol=0,
            atol=0,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=11)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    matrix = torch.randn(
        BATCH,
        CHUNK_SIZE,
        CHUNK_SIZE,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.01
    first = torch.randn(
        BATCH,
        CHUNK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.01
    second = torch.randn_like(first)

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    functions = {
        "rhs_baseline": torch.compile(rhs_baseline, **compile_args),
        "rhs_fused": torch.compile(rhs_fused, **compile_args),
        "left_baseline": torch.compile(left_baseline, **compile_args),
        "left_fused": torch.compile(left_fused, **compile_args),
    }
    print(
        f"batch={BATCH} chunk={CHUNK_SIZE} dim={HEAD_DIM} "
        f"dtype={matrix.dtype}",
        flush=True,
    )
    rhs_inputs = matrix, first, second
    rhs_expected = benchmark(
        "rhs_baseline",
        functions["rhs_baseline"],
        rhs_inputs,
        args.iterations,
    )
    rhs_actual = benchmark(
        "rhs_fused",
        functions["rhs_fused"],
        rhs_inputs,
        args.iterations,
    )
    check_equal(rhs_actual, rhs_expected, "rhs")

    left_inputs = first, second, first
    left_expected = benchmark(
        "left_baseline",
        functions["left_baseline"],
        left_inputs,
        args.iterations,
    )
    left_actual = benchmark(
        "left_fused",
        functions["left_fused"],
        left_inputs,
        args.iterations,
    )
    check_equal(left_actual, left_expected, "left")


if __name__ == "__main__":
    main()
