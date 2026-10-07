# TP4 decode campaign: rejected and pending paths

These are diagnostic decisions, outside the microbenchmark gain ledger. None
establishes a new end-to-end baseline or satisfies the official-sampling target.
The accepted formal baseline is now 8.239257597 ms/token; the current target is
7.0 ms/token at 16K, temperature 1.0, top_p 0.95 and seed 42.
The table below preserves earlier decisions; later accepted defaults and pending
totals are maintained in the optimization ledger.

| Candidate | Actual evidence | Decision |
| --- | --- | --- |
| All-route expert decode | Activated real-16 trials were slower. The subsequent independent N/batch/K access-map repair retained exact production-shape consumer bytes, but the compiler still placed weights in DRAM and changed six batched GEMMs into twelve GEMMs. Physical computation nodes increased from 18 to 23 in the bounded consumer screen. | Disabled. The access-map repair does not justify another real-16 run. |
| Combined mHC gates and FFN norm/quant | C1/C2/C6 normalized rows, FP8 operands, scales, gates and router outputs matched byte-for-byte. Physical TPC nodes decreased from two to one. The native consumer chain measured 37.790 versus 40.926 microseconds, with a 4.101-microsecond significance threshold. | No gain. Archive the experimental kernel rather than shipping its registration. Combining these independent consumers does not validate control-GEMV or post-collapse fusion. |
| Device-produced continuation position | Four-card real-16 tokens, 33 mutable states and hidden bytes matched. The latest ABABAB measured 5.576310 versus 5.211879 ms; the 0.364431-ms difference remained below twice baseline IQR, 0.546232 ms. | No microbenchmark gain credit or default promotion. Independent serving regression repair remains pending. |

## The cold-compilation gate must include Bridge recipes

The original position-bank reference produced 232 different single-row I32
`memcpy_dma` recipes for positions 16384 through 16615. Their storage offsets
were exactly position times four. Counting native-plan captures alone missed
these deferred Bridge compilations, so the first timed period was not a valid
warm comparison. The resident tool now also counts serialized GC recipe files
before and after timing. Preserve the invalid period and its provenance.

The device-position candidate keeps its next coordinate at a fixed output
address. It preserves request/step ownership checks and waits for the same
candidate coverage certificate before subsequent state-consuming replay.
Partial-chain noise has not established a qualified latency gain.

## Full sampling repair is a startup contract

Startup must warm both the unfiltered and top-p-filtered full-vocabulary repair
with the native tail's actual tensor layout, including selected-token commit
and scalar readback. A startup check of top_p=1 alone leaves a real top_p=0.95
miss exposed to cold compilation. Repairs reuse the bounded attempt's uniform;
the continuation coordinate is independent of token selection.

Raw measurements, compiler allocations, numerical checks and launch manifests
are indexed under the existing local decode campaign evidence root. Failed
screens do not contribute to cumulative savings. Graph export is a serialized
compiler diagnostic and remains disabled during formal serving measurement.

## Receive readiness and transport retirement

The ordered completion-group variant passed five checkpoint-derived fixtures
on four ranks, with all thirteen outputs byte-exact between arms. Its complete
native WO/control → peer → post/statistics → norm/quant → router/shared-W13
chain was slower in all three pairs: 0.210, 0.204 and 0.251 microseconds per
boundary. Reject this variant; it supplies no gain-ledger entry or new baseline.
Evidence: SSD `decode-hcl-data-ready-01/chain-04/DECISION.json`.

The short-monitor prototype initially retained a long-monitor capture validator,
then passed a SOB number where the helper expected a SOB group number. Both
cold-gate failures are archived and supply no timing result. Checking the actual
SDK helper and candidate call now covers all sixty-four slots and both flag
values. A separate million-phase CPU model covers additive signal ordering and
slot reuse; neither test establishes device correctness or performance.

The owned read-only hardware snapshot found the remaining cold-wait cause:
the received-data SOB is in SM1, while the existing long monitor is in SM3.
The data reached its expected flag, but a short monitor in SM3 cannot consume
SM1's SOB. The revised private prototype reused the receiver's
regular SM1 monitor and leases the unassigned tail of the first network monitor
pool for the compute stream. The fourth monitor pool was absent in the actual
runtime and its failed preflight is archived. A public query reports 384 entries
in the first pool; the source assigns five complete quotients of 76 entries,
leaving four unassigned entries. Public geometry and all source-derived reserved
monitor ranges are checked before commands, including HFC and long-monitor pools.
Original long monitors and full SEND/RECV/copy
retirement remain unchanged. The private wait-domain API has a distinct version;
unmatched runtime versions are rejected.

Evidence: SSD `decode-hcl-short-ready-01/PLAN.json` and
`cold-snapshot-01/DECISION.json`. The later same-SM run stopped on first slot
reuse: each rank's automatic DFA kernel log reports calculated SO value
overflow/underflow at SOB 0x4c0. Fifteen-bit register storage does not imply
modular increment semantics. This invalidates the CPU model's wrapping-add
assumption and closes the short-monitor prototype without timing or gain credit.
It remained outside installed serving source and defaults. All owned workers
retired and modules returned to 768 MiB; no manual reset or PCIe experiment was
performed. See `chain-05/DECISION.json`. Pending savings remain zero. The next
independent hypothesis preserves the original completion and retirement
mechanism while preposting receives with a proved epoch boundary.

The separate receive-prepost chain passed five checkpoint fixtures on four
ranks, with three consistent native A/B savings. It is recorded in the gain
ledger and remains default-off pending full serving qualification. Its epoch
callback and source-fingerprinted optional runtime overlays are maintained in
the shared replay path. Legacy standalone/native-batch entrypoints reject an
epoch-configured graph instead of omitting its prior-compute dependency.

Removing repeated epoch arm/fence packets from each receive scheduler stream
passed the SDK packet checks and all five four-rank fixtures, but slowed all
three complete-chain pairs by 0.069, 0.062 and 0.062 microseconds per boundary.
Reject this optional hoist. Auxiliary receive streams without such waits remain
byte-identical; the initial overly strict cold guard is retained as an invalid
untimed attempt. No defaults, gain credit, or formal request follow this trial.
Evidence: SSD `decode-receive-epoch-hoist-01/chain-02/DECISION.json`.
