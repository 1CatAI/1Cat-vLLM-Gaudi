# V4.1 engine compatibility patches

`0001-idempotent-shm-cancel.patch` applies to vLLM commit `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba` with the existing Gaudi engine integration. It makes shared-memory notification cancellation idempotent and serializes access to its PAIR socket. The parent-death monitor and worker cleanup can otherwise send simultaneously. It changes shutdown only, not model arithmetic or steady serving.

Apply to an isolated engine checkout with `git apply --check` and `git apply`, then use that checkout through the normal engine installation. Do not inject this patch into a running worker. The regression is in the existing shared-memory communication suite:

```sh
.venv/bin/python -m pytest tests/distributed/test_shm_broadcast.py -k spin_condition_concurrent_cancel
```

The isolated CPU regression fails on the original source and passes with the patch. Full-model graceful shutdown still needs verification in the next combined candidate; passing the local cancellation check does not qualify all runtime cleanup.

Socket threading contract: [ZeroMQ guide](https://zguide.zeromq.org/docs/chapter2/). No upstream PR is submitted by this patch bundle.
