# SPDX-License-Identifier: Apache-2.0
"""Fail closed after an error in a stateful distributed model transaction."""
import logging
import os

logger = logging.getLogger(__name__)


def fail_worker(phase, error):
    # A rank may have already changed KV/Engram or submitted a collective. RPC
    # error replies from non-output ranks are not sufficient to wake the engine.
    # Exiting wakes the executor's worker monitor, which tears down all ranks
    # and withdraws readiness. Do not drain potentially broken device streams.
    logger.critical("V4.1 %s failed; terminating worker to invalidate distributed request state",
                    phase,
                    exc_info=(type(error), error, error.__traceback__))
    logging.shutdown()
    os._exit(1)


def execute_guarded(function, *args, phase, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        fail_worker(phase, error)


def guard_async(output):
    from vllm.v1.outputs import AsyncModelRunnerOutput

    if not isinstance(output, AsyncModelRunnerOutput):
        return output

    class GuardedOutput(AsyncModelRunnerOutput):

        def get_output(self):
            return execute_guarded(output.get_output, phase="async completion")

    return GuardedOutput()
