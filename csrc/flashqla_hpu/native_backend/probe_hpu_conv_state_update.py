import argparse
from pathlib import Path

import torch

from vllm_gaudi.utils import HPUCompileConfig


PACKED_WIDTH = 10_240
CONV_WIDTH = 4


def functional_conv(
    packed,
    conv_state,
    weight,
    bias,
    has_initial_state,
    query_start_loc,
    cache_indices,
):
    output, updated_state = torch.ops.hpu.causal_conv1d_fwd(
        packed,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
        activation=True,
        pad_slot_id=-1,
    )
    query, key = torch.ops.custom_op.qwen38_post_conv_qk_bf16_gaudi2(
        output
    )
    return query, key, output, updated_state


@torch._dynamo.disable
def persist_state_at_graph_tail(
    query,
    key,
    output,
    updated_state,
    conv_state,
    cache_indices,
):
    selected_state = updated_state.index_select(0, cache_indices)
    conv_state.index_copy_(0, cache_indices, selected_state)
    return query, key, output


def graph_tail_update_conv(
    packed,
    conv_state,
    weight,
    bias,
    has_initial_state,
    query_start_loc,
    cache_indices,
):
    output, updated_state = torch.ops.hpu.causal_conv1d_fwd(
        packed,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
        activation=True,
        pad_slot_id=-1,
    )
    query, key = torch.ops.custom_op.qwen38_post_conv_qk_bf16_gaudi2(
        output
    )
    return persist_state_at_graph_tail(
        query,
        key,
        output,
        updated_state,
        conv_state,
        cache_indices,
    )


def make_inputs(tokens, slots):
    torch.manual_seed(739251)
    packed = torch.randn(
        (tokens, PACKED_WIDTH), dtype=torch.bfloat16, device="hpu"
    ) * 0.08
    weight = torch.randn(
        (CONV_WIDTH, PACKED_WIDTH),
        dtype=torch.bfloat16,
        device="hpu",
    ) * 0.08
    bias = torch.randn(
        (PACKED_WIDTH,), dtype=torch.bfloat16, device="hpu"
    ) * 0.01
    conv_state = torch.zeros(
        (slots, CONV_WIDTH - 1, PACKED_WIDTH),
        dtype=torch.bfloat16,
        device="hpu",
    )
    has_initial_state = torch.zeros(1, dtype=torch.bool, device="hpu")
    query_start_loc = torch.tensor(
        [0, tokens], dtype=torch.int32, device="hpu"
    )
    cache_indices = torch.tensor([0], dtype=torch.int64, device="hpu")
    return (
        packed,
        conv_state,
        weight,
        bias,
        has_initial_state,
        query_start_loc,
        cache_indices,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--slots", type=int, default=5)
    args = parser.parse_args()

    torch.ops.load_library(str(args.extension.resolve()))
    compile_args = HPUCompileConfig().get_compile_args()
    compile_args["fullgraph"] = True
    reference = torch.compile(functional_conv, **compile_args)
    candidate_args = dict(compile_args)
    candidate_args["fullgraph"] = False
    candidate = torch.compile(graph_tail_update_conv, **candidate_args)

    inputs = make_inputs(args.tokens, args.slots)
    reference_inputs = tuple(tensor.clone() for tensor in inputs)
    query_ref, key_ref, output_ref, expected_state = reference(
        *reference_inputs
    )
    query, key, output = candidate(*inputs)
    torch.hpu.synchronize()

    metrics = {
        "query_max_abs": (query.float() - query_ref.float()).abs().max(),
        "key_max_abs": (key.float() - key_ref.float()).abs().max(),
        "output_max_abs": (output.float() - output_ref.float()).abs().max(),
        "state_max_abs": (
            inputs[1].float() - expected_state.float()
        ).abs().max(),
        "state_changed": (
            inputs[1].float() - reference_inputs[1].float()
        ).abs().max(),
    }
    for name, value in metrics.items():
        print(f"{name}={float(value.cpu()):.9e}")
    assert float(metrics["query_max_abs"].cpu()) == 0.0
    assert float(metrics["key_max_abs"].cpu()) == 0.0
    assert float(metrics["output_max_abs"].cpu()) == 0.0
    assert float(metrics["state_max_abs"].cpu()) == 0.0
    assert float(metrics["state_changed"].cpu()) > 0.0
    print("PASS")


if __name__ == "__main__":
    main()
