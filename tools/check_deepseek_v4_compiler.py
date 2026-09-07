# SPDX-License-Identifier: Apache-2.0
"""Check source compiler coverage with synthetic boundaries, without model weights."""

import os


def main():
    if not os.environ.get("HABANA_VISIBLE_MODULES") or not os.environ.get("HLS_MODULE_ID"):
        raise RuntimeError("Select one available module with HABANA_VISIBLE_MODULES and HLS_MODULE_ID")
    os.environ.setdefault("PT_HPU_LAZY_MODE", "0")
    os.environ.setdefault("PT_HPU_ENABLE_EAGER_CACHE", "0")
    import torch
    import habana_frameworks.torch.core  # noqa: F401
    from habana_frameworks.torch.dynamo.compile_backend import passes
    from vllm_gaudi.compilation.deepseek_v4 import make_backend

    @torch.library.custom_op("vllm::deepseek_v4_attention", mutates_args={"out"})
    def attention(hidden_states: torch.Tensor, positions: torch.Tensor, out: torch.Tensor,
                  layer_name: str) -> None:
        out.copy_(hidden_states + positions.reshape(-1, 1, 1))

    @attention.register_fake
    def fake(hidden_states, positions, out, layer_name):
        return None

    def function(x, positions):
        for layer in range(4):
            output = torch.empty_like(x)
            attention(x, positions, output, str(layer))
            x = output * 2
        return x

    records = []

    def inspect_compiled_graph(ctx):
        nodes = list(ctx.graph_module.graph.nodes)
        records.append({
            "remaining_hops": sum(node.target is torch.ops.higher_order.auto_functionalized_v2 for node in nodes),
            "ordered_outputs": sum(node.target is torch.ops.dsv4_ordered.attention_out.default for node in nodes),
        })
        return False

    # This observer only inspects this process's synthetic compilation. It is
    # never installed in a model worker or used to modify a running service.
    passes.custom_pass_at_post_partition.append(inspect_compiled_graph)
    try:
        compiled = torch.compile(function, backend=make_backend(), fullgraph=True, dynamic=False)
        with torch.inference_mode():
            for tokens in (64, 1, 1):
                x = torch.ones((tokens, 64, 512), device="hpu", dtype=torch.bfloat16)
                positions = torch.arange(tokens, device="hpu", dtype=torch.bfloat16)
                actual = compiled(x, positions).cpu()
                torch.testing.assert_close(actual, function(x, positions).cpu(), rtol=0, atol=0)
                torch.testing.assert_close(x.cpu(), torch.ones_like(x, device="cpu"), rtol=0, atol=0)
        assert records == [{"remaining_hops": 4, "ordered_outputs": 0},
                           {"remaining_hops": 0, "ordered_outputs": 4}], records
    finally:
        passes.custom_pass_at_post_partition.remove(inspect_compiled_graph)
    print("Source compiler: prefill preserved, decode producer dependencies lowered, exact outputs, no recompile")


if __name__ == "__main__":
    main()
