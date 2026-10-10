# SPDX-License-Identifier: Apache-2.0
import json

from vllm_gaudi.compilation.deepseek_v41_memory_certificate import MemoryCertificate


def established(tmp_path):
    directory = tmp_path / "group" / "rank0"
    directory.mkdir(parents=True)
    (directory / "frontend.json").write_text('{"schema":3}')
    (directory / "frontend.bin").write_bytes(b"immutable graph")
    cold = MemoryCertificate(tmp_path, 0, dict(model="weights", precision="fp8", shapes=[1, 6, 8192]))
    cold.measured(before=100, peak=150, resident=125, workspace=30)
    return cold, directory


def test_measured_budget_requires_completed_serving_warmup(tmp_path):
    cold, _ = established(tmp_path)
    assert cold.restore() is None
    cold.publish_after_warmup()
    fresh = MemoryCertificate(tmp_path, 0, cold.contract)
    assert fresh.restore() == 55  # Both resident growth and working headroom are reserved.


def test_missing_and_corrupted_artifacts_repeat_profiling(tmp_path):
    cold, directory = established(tmp_path)
    cold.publish_after_warmup()
    (directory / "frontend.bin").write_bytes(b"corrupt")
    assert cold.restore() is None
    (directory / "frontend.bin").unlink()
    assert cold.restore() is None


def test_changed_precision_or_shape_cannot_borrow_old_budget(tmp_path):
    cold, _ = established(tmp_path)
    cold.publish_after_warmup()
    for change in (dict(precision="bf16"), dict(shapes=[1, 6, 16384]), dict(model="new weights")):
        assert MemoryCertificate(tmp_path, 0, dict(cold.contract, **change)).restore() is None
    assert MemoryCertificate(tmp_path, 1, cold.contract).restore() is None


def test_corrupted_certificate_cannot_underreserve(tmp_path):
    cold, _ = established(tmp_path)
    cold.publish_after_warmup()
    envelope = json.loads(cold.path.read_text())
    envelope["record"]["workspace_bytes"] = 0
    cold.path.write_text(json.dumps(envelope))
    assert cold.restore() is None


def test_missing_sdk_recipe_invalidates_memory_measurement(tmp_path):
    root = tmp_path / "frontend"
    root.mkdir()
    cold, _ = established(root)
    recipes = tmp_path / "rank0"
    recipes.mkdir()
    recipe = recipes / "recipe.bin"
    recipe.write_bytes(b"compiled recipe")
    cold.publish_after_warmup()
    assert cold.restore() == 55
    recipe.unlink()
    assert cold.restore() is None
