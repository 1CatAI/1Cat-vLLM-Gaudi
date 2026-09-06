import argparse
import statistics
import time

import torch

from vllm_gaudi.ops.hpu_gdn_pytorch import (
    _hpu_recursive_unit_lower_inverse,
    _solve_bmm,
)
from vllm_gaudi.utils import HPUCompileConfig


BATCH = 12_288
SIZE = 64
BASE = 16


def current_inverse(lmat: torch.Tensor) -> torch.Tensor:
    return _hpu_recursive_unit_lower_inverse(lmat, BASE)


def _residual_recursive_inverse(
    lmat: torch.Tensor,
    base: int,
) -> torch.Tensor:
    size = lmat.shape[-1]
    if size == base:
        eye = torch.eye(
            size,
            dtype=lmat.dtype,
            device=lmat.device,
        ).unsqueeze(0)
        inverse = 2 * eye - lmat
        for _ in range((size - 1).bit_length() - 1):
            residual = torch.bmm(lmat, inverse) - eye
            inverse = inverse - torch.bmm(inverse, residual)
        return inverse

    half = size // 2
    batch = lmat.shape[0]
    top_left = lmat[:, :half, :half]
    bottom_right = lmat[:, half:, half:]
    bottom_left = lmat[:, half:, :half]
    diagonal_inverse = _residual_recursive_inverse(
        torch.cat((top_left, bottom_right), dim=0),
        base,
    )
    top_left_inverse = diagonal_inverse[:batch]
    bottom_right_inverse = diagonal_inverse[batch:]
    bottom_left_inverse = -_solve_bmm(
        _solve_bmm(bottom_right_inverse, bottom_left),
        top_left_inverse,
    )
    zeros = torch.zeros_like(top_left)
    top = torch.cat((top_left_inverse, zeros), dim=-1)
    bottom = torch.cat((bottom_left_inverse, bottom_right_inverse), dim=-1)
    return torch.cat((top, bottom), dim=-2)


def residual_inverse(lmat: torch.Tensor) -> torch.Tensor:
    return _residual_recursive_inverse(lmat, BASE)


def cholesky_factor_inverse(lmat: torch.Tensor) -> torch.Tensor:
    gram_inverse = torch.cholesky_inverse(lmat, upper=False)
    return torch.bmm(lmat.transpose(1, 2), gram_inverse)


def benchmark(name, function, inputs, warmups, samples):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(warmups):
        output = function(*inputs)
    torch.hpu.synchronize()

    times_ms = []
    for _ in range(samples):
        start = time.perf_counter()
        output = function(*inputs)
        torch.hpu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1_000)
    print(
        f"{name} median_ms={statistics.median(times_ms):.6f} "
        f"min_ms={min(times_ms):.6f} samples_ms={times_ms}",
        flush=True,
    )
    return output.clone()


def report_quality(reference, candidate):
    reference_f32 = reference.float()
    difference = candidate.float() - reference_f32
    relative_l2 = difference.norm() / reference_f32.norm().clamp_min(1e-12)
    print(
        f"quality equal={torch.equal(reference, candidate)} "
        f"rel_l2={float(relative_l2.cpu()):.9e} "
        f"mean_abs={float(difference.abs().mean().cpu()):.9e} "
        f"max_abs={float(difference.abs().max().cpu()):.9e}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(739251)
    lower = torch.tril(
        torch.randn(
            BATCH,
            SIZE,
            SIZE,
            dtype=torch.float32,
            device="hpu",
        )
        * 0.01,
        diagonal=-1,
    )
    identity = torch.eye(SIZE, dtype=torch.float32, device="hpu")
    lmat = lower + identity.unsqueeze(0)
    torch.hpu.synchronize()

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    compiled_current = torch.compile(current_inverse, **compile_args)
    compiled_residual = torch.compile(residual_inverse, **compile_args)
    compiled_cholesky = torch.compile(cholesky_factor_inverse, **compile_args)
    print(
        f"shape={tuple(lmat.shape)} base={BASE} "
        f"compile_args={compile_args}",
        flush=True,
    )
    reference = benchmark(
        "current_inverse",
        compiled_current,
        (lmat,),
        args.warmups,
        args.samples,
    )
    candidate = benchmark(
        "residual_inverse",
        compiled_residual,
        (lmat,),
        args.warmups,
        args.samples,
    )
    report_quality(reference, candidate)
    cholesky = benchmark(
        "cholesky_factor_inverse",
        compiled_cholesky,
        (lmat,),
        args.warmups,
        args.samples,
    )
    report_quality(reference, cholesky)


if __name__ == "__main__":
    main()
