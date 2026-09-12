# V4.1 candidate qualification

Use a separate plugin/engine checkout, build output, recipe cache and evidence
directory for each source candidate. Preserve the parent timings and source,
runtime, workload, CPU and device fingerprints. Failed attempts remain archived.

Build native extensions with `tools/build_deepseek_v41.py`. Its default output
is the package's `vllm_gaudi/lib` directory. If an independent output directory
is used, also install that build's `dsv41_host_gather` extension into the package
directory for normal Python imports, retaining its build manifest and hash.
The configured native library directory must refer to the same verified build.

Before full-model measurement, check changed inputs, routing and state; complete
layer/collective execution; compiled SRAM placement; actual MME operand types;
and output equivalence within the affected scope. The experimental
`VLLM_HPU_DSV41_EXPERT_K128` path changes TPC work distribution while retaining
full-K BF16 MME. Verify the resulting graph instead of assuming this guarantees
a particular placement, core count or speedup.

Measure only the new candidate with `tools/measure_deepseek_v41_candidate.py`.
Set `--target-ms` explicitly. It retains the established three-request timing
protocol. `--trace` performs an independent profiler request after the timed
requests; never convert the device window into unprofiled ITL by rescaling.

For a completed capture, run these tools in dependency order:

1. `collect_deepseek_v41_trace.py RUN --jobs 4` extracts the four existing
   captures and recipe identities. It does not request another trace.
2. `analyze_deepseek_v41_trace.py ANALYSIS --rank RANK` reconstructs the token
   sequence for each rank. Independent ranks may run concurrently.
3. `map_deepseek_v41_trace_contracts.py ANALYSIS` joins the archived graphs.
4. `report_deepseek_v41_trace.py ANALYSIS` produces kernel contracts and timings.
5. `account_deepseek_v41_trace.py ANALYSIS` produces the disjoint functional
   ledger, overlap accounting and searchable kernel table.
6. `analyze_deepseek_v41_resources.py ANALYSIS --output-dir RESOURCES` reports
   whole-token and stage activity and TPC concurrency.
7. `analyze_deepseek_v41_dataflow.py ANALYSIS --rank RANK --output-dir RESOURCES`
   joins expert/attention weight producers with their actual MME consumers.
   It also reconstructs selected-KV and sparse-attention TPC work distribution.
8. When a host-side hypothesis needs scope evidence,
   `analyze_deepseek_v41_host_scopes.py ANALYSIS --rank RANK --output-dir SCOPES`
   extracts Python calls from the preserved raw trace and intersects their
   intervals with the global device gaps. It caches the extraction and retains
   parent identifiers. Nested scopes and wait envelopes are correlations, not
   additive costs or proof that the gaps can be removed.

The disjoint ledger reconciles the capture; its group ordering does not establish
causal critical-path ownership. Resource unions overlap. Busy cores do not prove
instruction efficiency, and engine activity does not establish peak throughput.
The tools retain unknown invocation boundaries, resource counters and gaps.

Use `compare_deepseek_v41_candidate.py PARENT_RESULT CANDIDATE_RESULT --checks
CHECKS_JSON --output DECISION_JSON --target-ms TARGET --structural-change` to
apply the per-round target and minimum net-gain rule. `CHECKS_JSON` records
evidenced Boolean contracts for `source_and_runtime`,
`changing_inputs_and_routing`, `layers_and_collectives`, `state_updates`,
`numerics`, `dataflow`, `memory_and_generation`, `no_fallback`, and
`trace_mechanism` when execution structure changes. Missing evidence cannot
qualify a candidate. Passing speed starts full quality/lifecycle qualification;
the comparison tool never enables a production default.
