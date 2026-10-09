# SPDX-License-Identifier: Apache-2.0
"""Restore the unchanged Router PT2 registration in a private additive installation.

This does not replace the GC provider or compile a different arithmetic path. The
caller must use a provider containing the Router/shared GUIDs; runtime loading
and native capture verify that contract before any measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--registration-parent', required=True, type=Path)
    parser.add_argument('--build-root', required=True, type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    parent, registration = args.parent.resolve(), args.registration_parent.resolve()
    old_path = registration / 'deepseek_v41_unique_build.json'
    old = json.loads(old_path.read_text())
    sources = [workspace / 'csrc/deepseek_v41_unique/pytorch/router_shared_fused.cpp',
               workspace / 'csrc/deepseek_v41_unique/kernels/deepseek_v41_router_shared_scaled_gaudi2.c']
    for source in sources:
        if old['sources'].get(str(source)) != digest(source):
            raise ValueError(f'Registration source differs from its compiled manifest: {source}')
    addons = list(registration.glob('hpu_dsv41_router_shared_fused_pt2*.so'))
    if len(addons) != 1:
        raise ValueError('Expected exactly one fingerprinted Router PT2 addon')
    addon = addons[0]
    if old['binaries'].get(addon.name) != digest(addon):
        raise ValueError('Router PT2 addon differs from its compiled manifest')
    manifest = json.loads((parent / 'deepseek_v41_unique_build.json').read_text())
    for name, expected in manifest['binaries'].items():
        if digest(parent / name) != expected:
            raise ValueError(f'Parent native binary changed: {name}')
    build = args.build_root.resolve()
    build.mkdir(parents=True, exist_ok=False)
    installed = build / 'installed'
    installed.mkdir()
    for path in parent.iterdir():
        if path.is_file():
            if path.suffix == '.json':
                shutil.copy2(path, installed / path.name)
            else:
                (installed / path.name).symlink_to(path.resolve())
    (installed / addon.name).symlink_to(addon.resolve())
    manifest['binaries'][addon.name] = digest(addon)
    manifest.setdefault('extra_registrations', []).append(addon.name)
    manifest['reused_router_registration'] = dict(
        path=str(addon), sha256=digest(addon), source_proof={str(p): digest(p) for p in sources},
        parent_manifest=str(old_path), parent_manifest_sha256=digest(old_path), new_arithmetic=False)
    (installed / 'deepseek_v41_unique_build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (build / 'build.json').write_text(json.dumps(dict(
        parent=str(parent), parent_manifest_sha256=digest(parent / 'deepseek_v41_unique_build.json'),
        registration=manifest['reused_router_registration']), indent=2) + '\n')
    print(installed)


if __name__ == '__main__':
    main()
