import argparse
import statistics
import time

import torch

from vllm_gaudi.ops.hpu_gdn_pytorch import hpu_chunk_gated_delta_rule
from vllm_gaudi.utils import HPUCompileConfig


TOKENS = 16_384
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_DIM = 128


def make_gdn(chunk_size):
    def gdn(q, k, v, gate, beta, initial_state):
        return hpu_chunk_gated_delta_rule(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            chunk_size=chunk_size,
            prefill_num_seqs=1,
            prefill_seq_len=TOKENS,
            neumann_iters=14,
            fused_state_matmul=True,
            deferred_output_add=True,
            recursive_solver_base=16,
            compact_repeated_kkt=True,
        )

    return gdn


def make_inputs():
    torch.manual_seed(739251)
    q = torch.randn(
        1,
        TOKENS,
        QK_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    )
    k = torch.randn_like(q)
    q /= torch.linalg.vector_norm(q.float(), dim=-1, keepdim=True)
    k /= torch.linalg.vector_norm(k.float(), dim=-1, keepdim=True)
    v = torch.randn(
        1,
        TOKENS,
        VALUE_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.01
    gate = -torch.rand(
        1,
        TOKENS,
        VALUE_HEADS,
        dtype=torch.float32,
        device="hpu",
    ) * 0.02
    beta = torch.rand(
        1,
        TOKENS,
        VALUE_HEADS,
        dtype=torch.bfloat16,
        device="hpu",
    )
    initial_state = torch.randn(
        1,
        VALUE_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        dtype=torch.float32,
        device="hpu",
    ) * 0.01
    return q, k, v, gate, beta, initial_state


def benchmark(name, function, inputs, iterations):
    output = function(*inputs)
    torch.hpu.synchronize()
    for _ in range(2):
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


def report_difference(actual, expected, name):
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    relative_l2 = torch.linalg.vector_norm(actual_float - expected_float)
    relative_l2 /= torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    print(
        f"{name}_max_abs={float(difference.max().cpu()):.9e} "
        f"mean_abs={float(difference.mean().cpu()):.9e} "
        f"relative_l2={float(relative_l2.cpu()):.9e} "
        f"equal_fraction="
        f"{float((actual == expected).float().mean().cpu()):.9f}",
        flush=True,
    )
    if float(relative_l2.cpu()) >= 0.02:
        raise AssertionError(f"{name} exceeded the 2% relative-L2 gate")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, default=32)
    parser.add_argument("--reference", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    for chunk_size in (args.candidate, args.reference):
        if chunk_size <= 0 or TOKENS % chunk_size:
            raise ValueError(
                f"chunk size must divide {TOKENS}, got {chunk_size}"
            )

    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    candidate = torch.compile(make_gdn(args.candidate), **compile_args)
    reference = torch.compile(make_gdn(args.reference), **compile_args)
    inputs = make_inputs()
    print(
        f"tokens={TOKENS} candidate_chunk={args.candidate} "
        f"reference_chunk={args.reference} qk_heads={QK_HEADS} "
        f"value_heads={VALUE_HEADS} dim={HEAD_DIM}",
        flush=True,
    )
    expected = benchmark(
        f"chunk_{args.reference}",
        reference,
        inputs,
        args.iterations,
    )
    actual = benchmark(
        f"chunk_{args.candidate}",
        candidate,
        inputs,
        args.iterations,
    )
    report_difference(actual[0], expected[0], "output")
    report_difference(actual[1], expected[1], "final_state")


if __name__ == "__main__":
    main()
