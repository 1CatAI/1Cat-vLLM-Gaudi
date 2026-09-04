import argparse
import statistics
import time

import torch

from vllm_gaudi.utils import HPUCompileConfig


NUM_CHUNKS = 256
QK_HEADS = 16
VALUE_HEADS = 48
CHUNK_SIZE = 64
HEAD_DIM = 128


def _causal_block_bmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    block_size: int,
    diagonal: int,
) -> torch.Tensor:
    rows = []
    num_blocks = CHUNK_SIZE // block_size
    for row in range(num_blocks):
        lhs_block = lhs[:, row * block_size:(row + 1) * block_size]
        blocks = []
        for column in range(num_blocks):
            if column < row:
                rhs_block = rhs[:, column * block_size:(column + 1) * block_size]
                block = torch.bmm(lhs_block, rhs_block.transpose(1, 2))
            elif column == row:
                rhs_block = rhs[:, column * block_size:(column + 1) * block_size]
                block = torch.tril(
                    torch.bmm(lhs_block, rhs_block.transpose(1, 2)),
                    diagonal=diagonal,
                )
            else:
                block = torch.zeros(
                    lhs.shape[0],
                    block_size,
                    block_size,
                    dtype=lhs.dtype,
                    device=lhs.device,
                )
            blocks.append(block)
        rows.append(torch.cat(blocks, dim=-1))
    return torch.cat(rows, dim=-2)


def full_strict(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    row_scale: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    del lower_mask
    product = torch.bmm(lhs, rhs.transpose(1, 2))
    return torch.tril(product * row_scale, diagonal=-1)


def mask_strict(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    row_scale: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    product = torch.bmm(lhs, rhs.transpose(1, 2))
    return product * row_scale * lower_mask


def block16_strict(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    row_scale: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    del lower_mask
    return _causal_block_bmm(lhs, rhs, 16, -1) * row_scale


def block32_strict(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    row_scale: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    del lower_mask
    return _causal_block_bmm(lhs, rhs, 32, -1) * row_scale


def full_causal(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    del lower_mask
    return torch.tril(torch.bmm(lhs, rhs.transpose(1, 2)))


def mask_causal(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(lhs, rhs.transpose(1, 2)) * lower_mask


def block16_causal(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    del lower_mask
    return _causal_block_bmm(lhs, rhs, 16, 0)


def block32_causal(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lower_mask: torch.Tensor,
) -> torch.Tensor:
    del lower_mask
    return _causal_block_bmm(lhs, rhs, 32, 0)


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


def report_quality(name, reference, candidate):
    reference_f32 = reference.float()
    difference = candidate.float() - reference_f32
    relative_l2 = difference.norm() / reference_f32.norm().clamp_min(1e-12)
    print(
        f"quality {name} equal={torch.equal(reference, candidate)} "
        f"rel_l2={float(relative_l2.cpu()):.9e} "
        f"mean_abs={float(difference.abs().mean().cpu()):.9e} "
        f"max_abs={float(difference.abs().max().cpu()):.9e}",
        flush=True,
    )


def run_case(name, batch, strict, warmups, samples, compile_args):
    torch.manual_seed(739251 + batch)
    lhs = torch.randn(
        batch,
        CHUNK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    )
    rhs = torch.randn_like(lhs)
    row_scale = torch.rand(
        batch,
        CHUNK_SIZE,
        1,
        dtype=torch.bfloat16,
        device="hpu",
    )
    diagonal = -1 if strict else 0
    lower_mask = torch.tril(
        torch.ones(
            CHUNK_SIZE,
            CHUNK_SIZE,
            dtype=torch.bfloat16,
            device="hpu",
        ),
        diagonal=diagonal,
    )
    torch.hpu.synchronize()

    if strict:
        inputs = lhs, rhs, row_scale, lower_mask
        functions = {
            "full": full_strict,
            "mask": mask_strict,
            "block32": block32_strict,
            "block16": block16_strict,
        }
    else:
        inputs = lhs, rhs, lower_mask
        functions = {
            "full": full_causal,
            "mask": mask_causal,
            "block32": block32_causal,
            "block16": block16_causal,
        }

    compiled = {
        variant: torch.compile(function, **compile_args)
        for variant, function in functions.items()
    }
    print(
        f"case={name} batch={batch} strict={strict} "
        f"shape={tuple(lhs.shape)}",
        flush=True,
    )
    outputs = {
        variant: benchmark(
            f"{name}_{variant}",
            function,
            inputs,
            warmups,
            samples,
        )
        for variant, function in compiled.items()
    }
    for variant in ("mask", "block32", "block16"):
        report_quality(f"{name}_{variant}", outputs["full"], outputs[variant])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument(
        "--only",
        choices=("kkt", "local", "both"),
        default="both",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    print(f"compile_args={compile_args}", flush=True)
    if args.only in ("kkt", "both"):
        run_case(
            "kkt",
            NUM_CHUNKS * QK_HEADS,
            True,
            args.warmups,
            args.samples,
            compile_args,
        )
    if args.only in ("local", "both"):
        run_case(
            "local",
            NUM_CHUNKS * VALUE_HEADS,
            False,
            args.warmups,
            args.samples,
            compile_args,
        )


if __name__ == "__main__":
    main()
