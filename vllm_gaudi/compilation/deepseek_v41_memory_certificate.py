# SPDX-License-Identifier: Apache-2.0
"""Reuse measured admission headroom while retaining serving graph warmup."""
import hashlib
import json
from pathlib import Path
import uuid


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class MemoryCertificate:
    """No allocation, recipe or communicator identity survives a restart."""

    def __init__(self, root, rank, contract):
        self.root, self.rank = Path(root), rank
        self.identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.path = self.root / "memory-admission" / f"rank{rank}-{self.identity}.json"
        self.contract = contract
        self.pending = None
        self.rejection = None

    def _inventory(self):
        result = {}
        for directory in self.root.rglob(f"rank{self.rank}"):
            for manifest in directory.rglob("*.json"):
                binary = manifest.with_suffix(".bin")
                if not binary.is_file():
                    continue
                for path in (manifest, binary):
                    result[str(path.relative_to(self.root))] = _digest(path)
        if not result:
            raise ValueError("Memory admission requires an established graph inventory")
        recipes = self.root.parent / f"rank{self.rank}"
        if recipes.is_dir():
            for path in recipes.rglob("*"):
                if path.is_file():
                    result[f"sdk/{path.relative_to(recipes)}"] = _digest(path)
        return result

    def restore(self):
        try:
            envelope = json.loads(self.path.read_text())
            record = envelope["record"]
            if hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() != envelope["sha256"]:
                raise ValueError("Memory certificate digest mismatch")
            if record["schema"] != 1 or record["identity"] != self.identity or record["contract"] != self.contract:
                raise ValueError("Memory measurement contract changed")
            for name, expected in record["inventory"].items():
                relative = Path(name)
                path = (self.root.parent / f"rank{self.rank}" /
                        Path(*relative.parts[1:]) if relative.parts and relative.parts[0] == "sdk" else self.root /
                        relative)
                if relative.is_absolute() or ".." in relative.parts or _digest(path) != expected:
                    raise ValueError("Memory certificate graph inventory changed")
            if not record["inventory"]:
                raise ValueError("Memory certificate has no graph inventory")
            resident, peak, workspace = (record[name]
                                         for name in ("resident_growth_bytes", "peak_growth_bytes", "workspace_bytes"))
            if any(type(value) is not int or value < 0 for value in (resident, peak, workspace)) or peak < resident:
                raise ValueError("Invalid measured memory headroom")
            return max(peak, resident + workspace)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.rejection = str(error)
            return None

    def measured(self, before, peak, resident, workspace):
        self.pending = dict(schema=1,
                            identity=self.identity,
                            contract=self.contract,
                            resident_growth_bytes=max(0, int(resident - before)),
                            peak_growth_bytes=max(0, int(max(peak, resident) - before)),
                            workspace_bytes=int(workspace))

    def publish_after_warmup(self):
        if self.pending is None:
            return
        # Publish only after actual serving prefill, native C1/C6 and protocol
        # preparation succeed. A failed warmup cannot certify a hot restart.
        record = dict(self.pending, inventory=self._inventory())
        envelope = dict(record=record, sha256=hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(envelope, sort_keys=True) + "\n")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.pending = None


def runner_certificate(runner, rank, specs, total_device_bytes):
    import os
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import computation_dependencies, normalized_source

    root = os.environ.get("VLLM_HPU_DSV41_FRONTEND_CACHE_DIR")
    if not root or not getattr(runner, "use_dspark", False):
        return None
    program = runner.model.program
    sources = computation_dependencies(type(runner)._dummy_run, program)
    for name in ("_forward", "profile_run", "warmup_model"):
        sources[f"runner:{name}"] = hashlib.sha256(normalized_source(getattr(type(runner), name)).encode()).hexdigest()
    contract = dict(sources=sources,
                    serving_identity=os.environ.get("DSV41_SERVING_COMPILE_IDENTITY"),
                    runtime=os.environ.get("DSV41_SERVING_RUNTIME"),
                    precision=program.precision_fingerprint,
                    total_device_bytes=int(total_device_bytes),
                    profile_pages=runner.profile_kv_cache_blocks(),
                    prefill_capacity=runner.prefill_capacity,
                    configured_workspace_bytes=int(runner.serving_workspace_reserve),
                    specs={
                        name:
                        dict(page_size_bytes=value.page_size_bytes,
                             state_shape=list(value.state_shape),
                             state_dtype=str(value.state_dtype))
                        for name, value in sorted(specs.items())
                    })
    if not contract["serving_identity"] or not contract["runtime"]:
        return None
    return MemoryCertificate(root, rank, contract)
