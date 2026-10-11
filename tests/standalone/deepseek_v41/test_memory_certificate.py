# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

from vllm_gaudi.compilation.deepseek_v41_memory_certificate import MemoryCertificate, relocate_certificates


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


def test_larger_pool_retains_measured_reserve_only_with_identical_inventory(tmp_path):
    template, directory = established(tmp_path)
    cold = MemoryCertificate(tmp_path, 0, dict(template.contract, total_device_bytes=1000))
    cold.measured(100, 150, 125, 30)
    cold.publish_after_warmup()
    larger = MemoryCertificate(tmp_path, 0, dict(cold.contract, total_device_bytes=1200))
    assert larger.restore() == 55
    assert not larger.path.exists()  # No invented new measurement.
    assert MemoryCertificate(tmp_path, 0, dict(cold.contract, total_device_bytes=900)).restore() is None
    assert MemoryCertificate(tmp_path, 0, dict(larger.contract, precision="bf16")).restore() is None
    (directory / "frontend.bin").write_bytes(b"changed")
    assert larger.restore() is None


def test_invalid_current_certificate_does_not_fall_back_to_smaller_pool(tmp_path):
    template, _ = established(tmp_path)
    cold = MemoryCertificate(tmp_path, 0, dict(template.contract, total_device_bytes=1000))
    cold.measured(100, 150, 125, 30)
    cold.publish_after_warmup()
    larger = MemoryCertificate(tmp_path, 0, dict(cold.contract, total_device_bytes=1200))
    larger.path.write_text('{"invalid":true}')
    assert larger.restore() is None


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


def test_malformed_inventory_repeats_profiling(tmp_path):
    cold, _ = established(tmp_path)
    cold.publish_after_warmup()
    envelope = json.loads(cold.path.read_text())
    envelope["record"]["inventory"] = ["not a mapping"]
    envelope["sha256"] = hashlib.sha256(json.dumps(envelope["record"], sort_keys=True).encode()).hexdigest()
    cold.path.write_text(json.dumps(envelope))
    assert cold.restore() is None


def test_relocated_profile_migration_keeps_measured_inventory(tmp_path):
    from vllm_gaudi.compilation.deepseek_v41_cache_identity import runtime_content_identity

    profiles = []
    for name in ("old", "relocated"):
        directory = tmp_path / name
        directory.mkdir()
        binary = directory / "runtime.so"
        binary.write_bytes(b"same runtime")
        profile = directory / "runtime.json"
        profile.write_text(
            json.dumps(
                dict(environment={"TMPDIR": str(directory)},
                     additional_libraries=[
                         dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                     ])))
        profiles.append(profile)
    root = tmp_path / "frontend"
    root.mkdir()
    cold, _ = established(root)
    contract = dict(cold.contract, runtime=hashlib.sha256(profiles[0].read_bytes()).hexdigest())
    cold = MemoryCertificate(root, 0, contract)
    cold.measured(100, 150, 125, 30)
    cold.publish_after_warmup()
    assert relocate_certificates(root, *profiles) == 1
    target = MemoryCertificate(root, 0,
                               dict(contract, runtime=runtime_content_identity(json.loads(profiles[1].read_text()))))
    assert target.restore() == 55
    (profiles[1].parent / "runtime.so").write_bytes(b"changed runtime")
    import pytest
    with pytest.raises(ValueError, match="not frozen"):
        relocate_certificates(root, *profiles)
