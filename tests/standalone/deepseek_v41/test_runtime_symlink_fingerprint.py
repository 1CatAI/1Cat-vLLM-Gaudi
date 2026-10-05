# SPDX-License-Identifier: Apache-2.0
"""Relocating an identical runtime through a symlink retains strict ABI checks."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).resolve().parents[3]/'vllm_gaudi/distributed/tp2_fused_ar_norm.py'
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_verify_prepared_runtime')
namespace={'Path':Path,'torch':SimpleNamespace(__version__='test')}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),namespace)
verify=namespace['_verify_prepared_runtime']


def runtime_fixture(tmp_path):
    # Use a real loaded ELF so this checks the same canonical path identity
    # as the kernel's maps table, rather than a fabricated runtime record.
    runtime=next(Path(row.split(maxsplit=5)[-1]).resolve() for row in Path('/proc/self/maps').read_text().splitlines()
                 if len(row.split(maxsplit=5))==6 and row.split(maxsplit=5)[-1].startswith('/')
                 and row.split(maxsplit=5)[-1].endswith('.so.6'))
    alias=tmp_path/'runtime.so';alias.symlink_to(runtime)
    binary=tmp_path/'bridge.so';binary.write_bytes(b'fixture bridge fingerprint')
    metadata=dict(schema=1,torch_version='test',binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  eager_runtime=[dict(path=str(alias),sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())])
    return binary,metadata


def test_identical_runtime_symlink_is_accepted(tmp_path):
    binary,metadata=runtime_fixture(tmp_path)
    binary.with_suffix('.abi.json').write_text(json.dumps(metadata))
    verify(binary)


def test_changed_runtime_bytes_are_still_rejected(tmp_path):
    binary,metadata=runtime_fixture(tmp_path);metadata['eager_runtime'][0]['sha256']='0'*64
    binary.with_suffix('.abi.json').write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError,match='GraphExec runtime changed'):verify(binary)


def test_changed_bridge_is_still_rejected(tmp_path):
    binary,metadata=runtime_fixture(tmp_path)
    binary.with_suffix('.abi.json').write_text(json.dumps(metadata));binary.write_bytes(b'changed bridge')
    with pytest.raises(RuntimeError,match='binary/Torch fingerprint'):verify(binary)
