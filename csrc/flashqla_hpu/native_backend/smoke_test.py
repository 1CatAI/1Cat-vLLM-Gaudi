from pathlib import Path

import torch


torch.ops.load_library(
    str(next(Path(__file__).parent.glob("flashqla_compound_probe*.so")).resolve())
)


def probe(lhs: torch.Tensor, rhs: torch.Tensor, bias: torch.Tensor):
    return torch.ops.custom_op.flashqla_compound_probe(lhs, rhs, bias)


def main() -> None:
    torch.manual_seed(739251)
    lhs = torch.randn(16, 64, 128, dtype=torch.bfloat16, device="hpu")
    rhs = torch.randn(16, 128, 128, dtype=torch.bfloat16, device="hpu")
    bias = torch.randn(16, 64, 128, dtype=torch.bfloat16, device="hpu")
    compiled = torch.compile(probe, backend="hpu_backend", fullgraph=True)
    actual = compiled(lhs, rhs, bias)
    torch.hpu.synchronize()
    expected = torch.bmm(lhs, rhs) + bias
    torch.testing.assert_close(actual.cpu(), expected.cpu())
    print(tuple(actual.shape), actual.dtype)


if __name__ == "__main__":
    main()
