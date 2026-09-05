# SPDX-License-Identifier: Apache-2.0
"""Experimental native mixed-engine plans. No auto dispatch or reference fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import torch

from flashinfer_gaudi._bridge import load_bridge_adapter
from flashinfer_gaudi._dispatch import BackendUnavailableError


@dataclass(frozen=True)
class GemmSiluArtifactV1:
    """Portable graph specification, not a serialized Synapse recipe or weights."""

    m: int
    k: int
    d: int
    adapter_id: str
    schema_version: int = 1
    operation: str = "bf16_gemm_silu_mul"
    target: str = "gaudi2"

    def __post_init__(self):
        if (type(self.schema_version) is not int or self.schema_version != 1 or self.operation != "bf16_gemm_silu_mul"
                or self.target != "gaudi2"
                or any(type(value) is not int or not 0 < value <= 2**31 - 1
                       for value in (self.m, self.k, self.d)) or self.d % 128 or self.d * 2 > 2**31 - 1):
            raise ValueError("Unsupported GEMM-SiLU graph specification")
        if (not isinstance(self.adapter_id, str) or len(self.adapter_id) != 64
                or any(char not in "0123456789abcdef" for char in self.adapter_id)):
            raise ValueError("Graph specification requires an adapter SHA256 identity")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> GemmSiluArtifactV1:
        return cls(**json.loads(value))


class GemmSiluPlan:
    """Static BF16 x[M,K] @ weight[K,2D] followed by BF16 SiLU(gate)*up.

    Plan creation loads and verifies the pinned adapter but does not execute or
    synchronize. First run builds/caches the recipe in Bridge. Scratch lifetime
    belongs to Synapse; no caller-provided workspace is implemented yet. The
    plan owns no Tensor and can serve separate output buffers on separate
    streams. Callers own stream ordering and may not concurrently reuse out.

    run(out=...) uses native eager recipe replay, not torch.compile mutation
    functionalization. To compile the allocating operation in a larger graph,
    use functional_op after construction. No performance promotion is implied.
    """

    def __init__(self, m: int, k: int, d: int, *, artifact: GemmSiluArtifactV1 | None = None):
        # Reject malformed shapes before importing or loading any HPU library.
        GemmSiluArtifactV1(m, k, d, "0" * 64)
        identity = load_bridge_adapter()
        expected = GemmSiluArtifactV1(m, k, d, identity["artifact_id"])
        if artifact is not None and artifact != expected:
            raise BackendUnavailableError("GEMM-SiLU graph specification does not match this shape/adapter")
        if str(torch.hpu.get_device_name()).upper() != "GAUDI2":
            raise BackendUnavailableError("Native GEMM-SiLU is qualified for Gaudi2 only")
        self._artifact = expected
        self._functional_op = torch.ops.custom_op.flashinfer_gaudi_gemm_silu
        self._out_op = torch.ops.custom_op.flashinfer_gaudi_gemm_silu_out

    @property
    def artifact(self) -> GemmSiluArtifactV1:
        return self._artifact

    @property
    def functional_op(self):
        """Registered allocating op for fullgraph HPU compilation; not an out wrapper."""
        return self._functional_op

    def run(self, x: torch.Tensor, weight: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        if torch.compiler.is_compiling():
            raise BackendUnavailableError("Compile plan.functional_op; native out replay is not functionalized")
        spec = self._artifact
        for tensor, shape in ((x, (spec.m, spec.k)), (weight, (spec.k, 2 * spec.d))):
            if (tensor.device.type != "hpu" or tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous() or tensor.requires_grad):
                raise BackendUnavailableError("GEMM-SiLU input does not match the native static plan")
        if x.device != weight.device:
            raise BackendUnavailableError("GEMM-SiLU inputs must share a device")
        if out is None:
            return self._functional_op(x, weight)
        if (out.device != x.device or out.dtype != x.dtype or tuple(out.shape) != (spec.m, spec.d)
                or not out.is_contiguous() or out.requires_grad or torch._C._is_alias_of(out, x)
                or torch._C._is_alias_of(out, weight)):
            raise BackendUnavailableError("GEMM-SiLU out contract mismatch or input alias; no output was written")
        return self._out_op(x, weight, out)
