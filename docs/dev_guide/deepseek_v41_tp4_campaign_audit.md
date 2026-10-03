# TP4 decode campaign: rejected and pending paths

These are diagnostic decisions, outside the microbenchmark gain ledger. None
establishes a new end-to-end baseline or satisfies the official-sampling target.
The accepted formal baseline remains 10.568 ms/token; the current target is
7.0 ms/token at 16K, temperature 1.0, top_p 0.95 and seed 42.

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
