# DeepSeek V4.1 startup cache

The serving installation reuses compilation artifacts without retaining another
process's tensors, native recipe IDs, streams, or communicator handles. It keeps
the configured context capacity, prefix caching, speculative decoding, tool
calling, and the full production warmup inventory.

## Restoration boundaries

1. A computation identity binds the prepared model, numerical implementation,
   precision, runtime binaries, and serving tensor contract. Installation paths,
   log directories, and unrelated tools are not computation dependencies.
2. The guarded frontend restores only a matching variant. Input layout, tensor
   qualification, callable defaults, owner configuration, and alias guards still
   apply. Other saved variants remain available for their matching shapes.
3. The lowered backend restores partitioned execution modules, JIT descriptions,
   output metadata, constant indexes, and input mappings. The original frontend
   FX module is reconstructed only if the lowered artifact is unavailable or
   invalid. Existing SDK recipe caching remains responsible for device recipes.
4. Native preparation binds current-process inputs, model weights, mutable state,
   communication resources, and output buffers. These resources are freshly
   allocated even when compilation artifacts are reused.

Source normalization is memoized by source content. Filesystem timestamps are
insufficient to identify edits. Syntax discovery is memoized separately; global
values, module contents, and deferred imports are still resolved when building
each computation identity.

## Memory admission certificate

DSpark startup historically measured the temporary profiling pool before warming
the scheduler's actual serving pool. A hot restart may reuse that measured
headroom only when its admission certificate and complete recorded graph/SDK
recipe inventory validate.

The certificate binds numerical sources, runtime/model identity, precision,
prefill capacity, profiling page count, state layout, device capacity, and the
configured working reserve. It records allocation growth and peak headroom as
byte counts. Admission reserves the greater of peak growth and persistent growth
plus working headroom, since profiling allocations have not yet been recreated.

Publication occurs only after full serving warmup succeeds. Missing, malformed,
oversized, corrupted, or incompatible certificates repeat ordinary profiling.
Missing or changed recorded artifacts also repeat profiling. No warmup shape is
removed: actual serving prefill, C1/C6 native replay, prefix capture, draft commit
lengths, and official-sampling protocols still run before API readiness.

## Qualification

Record initialization/weight preparation, prefill preparation, C1 native
preparation, C6 native preparation, and protocol/API readiness separately. Reader,
conversion, and upload timers can overlap and must not be added as exclusive
wall-clock categories.

Establish the complete cache first, then qualify two independent hot restarts.
Both must satisfy the configured launch-to-readiness budget and restore without
frontend capture or backend repartitioning. Component improvements alone do not
qualify startup. After readiness, exercise fixed quality samples, prefix reuse,
tool calls, and one unprofiled official long-input request to natural EOS. Keep
those results tied to the serving installation and its effective settings.
