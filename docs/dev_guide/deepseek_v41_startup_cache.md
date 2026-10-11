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

The allocation-pool fraction is a machine resource policy, not a numerical
compilation dependency. Cache fingerprints preserve its previous canonical
spelling for compatibility; the runtime profile records the actual configured
fraction. Memory admission still binds the actual pool size. An expanded pool
may retain a smaller pool's measured byte reserve only with every other contract
field and all recorded artifacts unchanged. A reduced pool cannot borrow it,
and a malformed current certificate must repeat profiling.

After all native and protocol preparation, the worker checks actual free pool
bytes against its serving workspace reserve before publishing readiness. A
successful warmup alone does not prove a subsequent long prompt can allocate
its working buffers. Retain context capacity and correct the resource budget
when this final admission fails.

Publication occurs only after full serving warmup succeeds. Missing, malformed,
oversized, corrupted, or incompatible certificates repeat ordinary profiling.
Missing or changed recorded artifacts also repeat profiling. No warmup shape is
removed: actual serving prefill, C1/C6 native replay, prefix capture, draft commit
lengths, and official-sampling protocols still run before API readiness.

## Qualification

Immutable Engram CPU tables can have an owner independent of the model service.
`tools/keep_deepseek_v41_engram_tables.py` prepares sealed shared backings and
keeps them resident across service restarts. Configure the live `ready.json`
through the machine setting `engram_shared_table_manifest`. Each service validates
the checkpoint identities, sealed extents and current residency, retains its own
descriptors, then creates fresh device registrations. Missing or stale owners
fall back to ordinary loading with the full host-memory admission requirement.

This is a host-memory cache, not a persisted device binding or an expanded disk
weight copy. Record initial backing creation separately from hot restart time;
after a machine reboot the owner must establish those backings again. The owner
must be managed separately from the model process group, and stopped when that
cache is no longer needed. It consumes the configured table memory budget.

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
