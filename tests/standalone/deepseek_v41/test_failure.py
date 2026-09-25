# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys

from vllm_gaudi.ops.deepseek_v41_failure import execute_guarded


def test_success_keeps_return_value_and_arguments():
    assert execute_guarded(lambda a, b: a + b, 3, b=7, phase="test") == 10


def test_failed_worker_cannot_continue_or_wait_on_device_cleanup():
    code = """
from vllm_gaudi.ops.deepseek_v41_failure import execute_guarded
def failed():
    raise RuntimeError('in-flight Engram/KV transaction')
execute_guarded(failed, phase='contract')
print('UNSAFE_CONTINUATION', flush=True)
"""
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert process.returncode == 1
    assert "UNSAFE_CONTINUATION" not in process.stdout
    assert "in-flight Engram/KV transaction" in process.stderr
