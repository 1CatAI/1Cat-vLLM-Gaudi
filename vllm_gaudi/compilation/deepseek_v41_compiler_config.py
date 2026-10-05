# SPDX-License-Identifier: Apache-2.0
"""Scoped cold-compile experiments; never change the captured replay loop."""
from contextlib import contextmanager
import ctypes


ALLOWED_SETTINGS = frozenset({'ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING',
                              'SYN_SRAM_BGEMM_SLICER_MULTIPLE_TINY_GEMMS_PER_SLICE',
                              'NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING',
                              'SYN_RMW_SECTION_MAX_SIZE_BYTES'})
INTEGER_VALUES = {'NON_COMMON_DIM_MIN_SLICE_NUM_FOR_PIPELINING': ('2', '4'),
                  'SYN_RMW_SECTION_MAX_SIZE_BYTES': ('16777216', '41943040', '67108864')}


@contextmanager
def compiler_configuration(settings, *, library=None):
    # The caller owns the common HPU compile lock for the complete scope.
    if not settings:
        yield
        return
    if set(settings) - ALLOWED_SETTINGS or any(value not in INTEGER_VALUES.get(key, ('true', 'false'))
                                                 for key, value in settings.items()):
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
            expected = ((value.encode(),) if key in INTEGER_VALUES else
                        (b'true', b'1') if value == 'true' else (b'false', b'0'))
            if status or buffer.value.lower() not in expected:
                raise RuntimeError(f'Synapse did not apply {key}={value}: {status}, {buffer.value!r}')
        yield
    finally:
        for key, value in reversed(list(saved.items())):
            status = library.synConfigurationSet(key.encode(), value)
            if status:
                raise RuntimeError(f'Synapse restore {key}: {status}')
