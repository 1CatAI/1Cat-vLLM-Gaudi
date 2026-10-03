# SPDX-License-Identifier: Apache-2.0
"""Scoped cold-compile experiments; never change the captured replay loop."""
from contextlib import contextmanager
import ctypes


ALLOWED_SETTINGS = frozenset({'ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING'})


@contextmanager
def compiler_configuration(settings, *, library=None):
    # The caller owns the common HPU compile lock for the complete scope.
    if not settings:
        yield
        return
    if set(settings) - ALLOWED_SETTINGS or any(value not in ('true', 'false') for value in settings.values()):
        raise ValueError('Unsupported C1 compiler experiment')
    if library is None:
        library = ctypes.CDLL('libSynapse.so')
        library.synConfigurationGet.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64]
        library.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    saved = {}
    try:
        for key, value in settings.items():
            buffer = ctypes.create_string_buffer(256)
            status = library.synConfigurationGet(key.encode(), buffer, len(buffer))
            if status:
                raise RuntimeError(f'Synapse read {key}: {status}')
            saved[key] = buffer.value
            status = library.synConfigurationSet(key.encode(), value.encode())
            if status:
                raise RuntimeError(f'Synapse set {key}: {status}')
            status = library.synConfigurationGet(key.encode(), buffer, len(buffer))
            expected = (b'true', b'1') if value == 'true' else (b'false', b'0')
            if status or buffer.value.lower() not in expected:
                raise RuntimeError(f'Synapse did not apply {key}={value}: {status}, {buffer.value!r}')
        yield
    finally:
        for key, value in reversed(list(saved.items())):
            status = library.synConfigurationSet(key.encode(), value)
            if status:
                raise RuntimeError(f'Synapse restore {key}: {status}')
