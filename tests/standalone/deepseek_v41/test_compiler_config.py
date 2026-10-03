# SPDX-License-Identifier: Apache-2.0
import pytest
from vllm_gaudi.compilation.deepseek_v41_compiler_config import compiler_configuration


class Configuration:
    def __init__(self):
        self.values = {b'ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING': b'true'}

    def synConfigurationGet(self, key, buffer, size):
        value = self.values[key]
        buffer.value = b'0' if value == b'false' else value
        return 0

    def synConfigurationSet(self, key, value):
        self.values[key] = value
        return 0


@pytest.mark.parametrize('fails', [False, True])
def test_compiler_option_restored_after_cold_compile(fails):
    library = Configuration()
    key = 'ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING'
    try:
        with compiler_configuration({key: 'false'}, library=library):
            assert library.values[key.encode()] == b'false'
            if fails:
                raise LookupError('compilation failed')
    except LookupError:
        assert fails
    assert library.values[key.encode()] == b'true'


def test_unknown_or_invalid_options_do_not_mutate_sdk():
    library = Configuration()
    for settings in ({'SRAM_SLICER_MAX_CAPACITY_BYTES': '0'},
                     {'ENABLE_BGEMM_FLATTEN_TO_GEMM_FOR_SLICING': '2'}):
        with pytest.raises(ValueError), compiler_configuration(settings, library=library):
            pass
    assert set(library.values.values()) == {b'true'}
