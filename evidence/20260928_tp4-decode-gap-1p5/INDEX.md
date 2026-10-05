
- continuation-chain-01: invalid startup, generic V2 Triton guard. Normal isolated engine patch added; real EngineArgs CPU preflight passed.
- continuation-chain-02: invalid startup, stock Synapse H2D mount compilation SIGSEGV before first model compute. No numerical or performance result; exact failure retained.
- build-continuation-02: bridge rebuilt against previously validated Synapse segmented-05/HCL segmented-04. Requires mapped-host capability gate before complete-chain retry.
- device-engram-capability-01: queued one-card six-head capability and exact numerical gate, all four shard layouts, image/history transitions. No speed claim.
- device-engram-capability-01/02: invalid diagnostic fixture; FileStore URI percent-encoding of Chinese path caused startup wait, confirmed by Python stack. No producer ran. Replaced with world-size-one HashStore.
- device-engram-capability-03: PASS on module0 with continuation runtime-v2; all 4 TP4 shard layouts × 11 changing/image tokens exact BF16 rows and I32 history. This establishes mapped-host ABI/geometry only, not large-table or performance qualification.
- continuation-cpu-05: 19 passed including precision/generation/search invalidation. New diagnostic tools passed Ruff.
- continuation-chain-03: authorized real four-rank component retry after isolated engine validation and mapped-host producer capability passed; same missing component reference protocol, no full-model baseline rerun.
- continuation-chain-03: all four ranks passed exact C1/C2/C6 and state ownership; 18.473337 -> 14.861044 ms/token in complete16-layer component, no hot compilation, no Engram major faults. Eligible for candidate-only normal serving, not E2E-qualified.
- serving-continuation-01: candidate-only normal-source serving qualification queued after complete-chain gate; six 16K-to-EOS requests including prefill and decode hardware captures. Saved full-model baseline reused. Native TP4 joint command replay still pending, no target attainment claimed.
- submission-archive-cpu: cold-path host graph exporter passes no-mutation CPU check and Ruff. Added after serving-continuation-01 snapshot froze; this run does not claim to contain those extra host-graph dumps. Existing hardware/recipe graph capture remains enabled.
- serving-continuation-01: INVALID before requests; C8192/C4096/C2048 pre-KV warmups passed, C1024 failed compiling concat node4083. Error surfaced at route descriptor CPU wait, but matching saved node is an attention I32 index concat. Old runtime integration is not E2E-qualified. Owned group retired, zero survivors.
- build-continuation-03: implement public synDeviceGetInfo.fd + hlthunk_host_memory_map device VA, DATA_TENSOR recipe inputs for TP4. Restores original stock Synapse/HCL and avoids legacy HOST_TO_DEVICE H2D mount semantics. Capability/numerical gate required before any full chain or serving retry.
- device-engram-capability-04: rejected before first device kernel; public raw device VA for mapped host memory fails DATA_TENSOR HBM section validation (status30). No validation disabled; raw-driver prototype removed from maintained code.
- build-continuation-05: stock runtime, graph DATA geometry + registered Synapse HOST_TO_DEVICE launch backing, ABI3. Source-based separation of static geometry and validated address translation; requires sealed shared backing capability gate.
- device-engram-capability-05: PASS on original stock Synapse/HCL, ABI3, all44 six-head shared-memfd outputs/history exact and explicit close succeeds. Workspace256bytes; static-table/mapped-launch contract validated.
- continuation-chain-04: retry with stock runtime and missing C8192→C1024 shape-transition gate. The old-runtime component B is not comparable to stock-runtime timing; measure only this missing component reference once, never the existing full-model B. Compiler submission graphs requested during cold preparation.

- continuation-chain-04: PASS stock runtime/ABI3, all ranks exact C1/C2/C6/full state, no hot compile; median component 12.676646 -> 9.414769 ms/token. All owned processes retired.
- continuation-cpu-06: 21 passed including actual bound IDs and image-to-text decode transitions.
- serving-continuation-02: candidate-only full40-layer normal service queued; original stock runtime, bridge ABI3, unchanged capacities, saved baseline reused.

- serving-continuation-02: complete16K→2989EOS exact, but rejected before any speed rounds: TP4 sampler repeated ABI hashing/maps traversal per token. Diagnostic trace acquired within warm request and preserved; no valid performance result. Qualified process group retired.
- continuation-cpu-07: 42 passed; added guard that TP4 sampler uses already verified readback and cannot re-resolve runtime in hot path. No arithmetic/device changes; reuse chain04 device gate.
- serving-continuation-03: justified normal-source retry after exact host integration bug fix. Uses explicit raw/scopes-only instrumentation matching preserved baseline; still no full-model B rerun.

- serving-continuation-03: invalid before requests; filesystem ENOSPC while graph compiler wrote post_graph JSON. No model/performance verdict. Retired owned processes and retain exact failure. Storage cleanup archives only this task’s retired diagnostic artifacts before retry.

- storage recovery: failed-run compiler graphs/post_graphs compressed losslessly and verified by zstd CRC + full file-name/size manifest before original removal; raw traces/logs retained. Next run and fresh recipe cache use /opt/optane/evidence-archive/dsv41-tp4-decode-gap-20260928, with >=20GiB archive and >=12GiB root free-space admission. Failed ENOSPC cache left unchanged and not reused.

- serving-continuation-04: REJECTED severe end-to-end regression; exact16K→2989EOS in warm + speed01 + speed02; unprofiled speed01 60.799047ms/token and speed02 70.622697ms/token, versus saved qualified22.135194ms. Stop further qualification at user correction; speed03 excluded after short diagnostic capture. Native four-rank HLTV preserved; CPU scope export failed clock-rate guard. Owned group retired with zero survivors. Offline parser at regression04-native-analysis; no replacement full-model baseline.
- 2026-09-29 user goal: TP4 PP1 normal serving <=10ms/token / >=100tokens/s, reduce unnecessary host/device dependencies and improve kernel/bandwidth efficiency. Keep16K naturalEOS semantics and production capacities; component14.861ms does not qualify full40-layer E2E.

- regression04-native-analysis: bounded200ms core+50ms padding parsed all4 ranks offline; each full Engram cycle has107 recipe starts, ~14–15ms local visible activity but ~47–72ms cadence. Per-rank post-sampleD2H→next first submit gaps reach28–36ms; global simultaneous noTPC/MME/DMA gap19.47ms. No CPU labels survived04, so these locate the stall, not a causal IPC/framework allocation. Raw CSV and per-node contracts retained.
- raw-clock-cpu / continuation-trace-cpu-08: explicit CPU scopes now keep CLOCK_MONOTONIC_RAW shared with SDK device events; no NTP wall stretching. Backward/forwardwallstep fixtures plus oldTorch calibration guards pass;80CPU tests + Ruff. Capture-only completion wait/value/async-output/position/Engram markers added. Admission ownership uses immutable/proc start ticks after observed create_time failure.
- serving-continuation-05: one bounded192-output diagnostic after16K input,128tokenswarm before32tokencapture. No performance candidate/noEOS claim. Justified by missing CPU scope data in04; normal fresh source with fixed recorder, unchanged math/runtime/capacities; reuse chain04 gate. Actual modules0/1/4/5 leased automatically.

- serving-continuation-05: COMPLETE diagnostic only;16Kinput/192outputprefix exact,finish=length,32profile cycles; both controlHTTP200,4CPU scope files CLOCK_MONOTONIC_RAW+4raw HLTV; no new EOS/performance claim. Owned group retired with0survivors. CPU subphase mean commit35ms:beforeposition13.15/15.71/27.01/27.87ms; position+Engram<0.21ms, prefixcompiled~2.2–3.1ms; token completion waits overlap async output.
- Packet completion root path: EngramHost.complete->_C1Packet.complete->HPUEvent.record->joinEagerThreadsCB precedes nextprefix; history/nativeC1complete onlybounded3-token metadata. MaintainedTP4deviceEngram cold init now retains existingrecord_native_completion, queuedlowering/executeFIFO. Generation/stream checks and reuse-only wait remain. Needs realdevice chain gate; no E2E claim.
- mhc-lane-simd-01: parent Gaudi2spills32ld+20st, candidate6ld+5st; instructionlines629→439. Isolatedlibrarybuilt after diagnosticretirement with112CPUs; sourceonlymHCkernelchanged versusqualifiedparentmanifest. Numerical/consumer gates pending.

- packet-completion-cpu-01:46passed + Ruff. packet-completion-hpu-01 only launcher preflight failed because servingRAWprofile lacks--enable-profiler; no card acquired/no device execution. packet-completion-hpu-02 uses original nonprofiler componentprofile, module0;18generations×3-slot ring/full+late-only uploads/retained real device consumers exact,1HPUtest passed,exit0. This validates ownership, not throughput.

- mhc-gate-chain-parent-01: fixture HPU backend import missing; before any gate/timing. Retired by launcher; fixed explicit import and retry as02.

- mhc-gate-chain-parent-02 / candidate-01:496 numerical/consumer cases bitwise exact, C1/C2 device chain medians improve2.62%/2.80%; C1 wall only0.42% and noisy. Same memory. Record mhc-lane-simd-01/QUALIFICATION.json. Advance to actual combined16-layer component, not E2E yet.

- LONG_CONTEXT_FASTPATH_AUDIT.md:16K尚未覆盖decoded-KV/MLA投影历史融合；静态确认2560镜像和C++行数上限，扩大镜像32slot最坏+2.3046875GiB/卡。优先审计packed MLA直接连接已有精确projection消费者；尚无新候选计时。

- continuation-chain-05:combined nativepacketcompletion+MHC SIMD passed complete16layer4rankgate,C1/C2/C6/prefill/stateexact;9.414769→8.670933ms medianrankhost (7.9007%),device8.664ms. OldBtiming reused,missingstateoracle reconstructed untimed. Allprocessesretired. Full40/API speedpending.

- paged-mla-projection-01:normalC++newcompound implemented and built112jobs74seconds outside device timing; existingpackedSRAM→batchGEMM path unchanged, connects existingproductRoPE/wo_a/wo_b→hc_post. Newopnotselectedinmodel andnotloadedbyserving06. Meta/device gatesseparate; microtool prepared.

- LONG_CONTEXT_FASTPATH_AUDIT纠正：TP4 runtime indexer实际关闭，decoded_main每source512行。32768token镜像增量78MiB/working，32saved+1working最坏2.5137GiB/卡。前一2.3GiB估算来自旧TP2常量，不作为TP4预算。实际长bucketphase32768行；微测试按此修正。

- serving-continuation-06 formalONEresult:39.828676ms/token,25.107538tokens/s,prefill26.664423s;warm/formal2989EOSexact.PerformanceREJECTEDvs22.135194msB; tracesinprogress. Most client250ms stalls near128-token page boundaries, plus other outliers;causalitynotyetassigned. SeePERFORMANCE_DECISION.json.No newdevicecandidate orfullmodelrepeat beforediagnosis.

- serving-continuation-06 retired with zero survivors. Prefill raw/CPU/Engram timing files all4 published, but stop_profile failed host profiling pending-transaction guard, campaign failed; decode not acquired. Formal regression retained. Offline audit of Engram residency/major faults and recorded host scopes next; no unchanged repetition.

- eng ram profiling ownership correction: native gather timers may change after wait has released all host gathers while device transaction remains owned. CPU02:23 passed + Ruff; CPU01 fixture environment selected installed older host ABI and failed collection, no device execution.
- regression06 offline recovery: all4 raw+CPU exact2989 tokens; traced prefill fault delta0, host gathers3–16ms exclude major faults as its dominant cause. Previous ~1000 prefill figure is tokens/s; serving01 speed01 actual prefill16.440499s, continuation04 23.740875s,06 26.664423s.
- Deterministic decode boundary bug found in saved06 graphs: stage_device_c1_reference executes unpack_swa eagerly,32 tiny decode recipes outside compiled graph with long submission gaps. First step and KV-page fallback affected; compiled fixed-destination producer implemented, awaiting real consumer gate. Ordinary alternating20.8/50.9ms still requires separate diagnosis.

- engram-reference-chain-01 invalid fixture before producer execution: generic model-less VllmConfig reached generic V2 Triton guard. Fixed fixture to existing passing real EngineArgs production config. Candidate not measured; owned group retired. Retry02 only.

- engram-reference-chain-02 invalid fixture before first producer: unexpanded rank{rank} recipe cache path rejected by runtime. Fixed per-rank expansion before runtime init as in passing chain05. No model/performance verdict.

- engram-reference-chain-03 PASS: four ranks,16realconsumer cases+3staging geometries perrank exact. Host medians acrossranks 22.996384→1.313609ms for fallback stage+actualEngram/Wkv/mHC chain. BothAPI/model absent;notE2Egain. CPU ownership/continuation suite45passed.

- build-position-01 stock bridge compiled112jobs; new prepared position bank capability1. position-copy-hpu-01 fixture relative test path absent in frozen source, no execution; corrected absolute test locator. position-copy-hpu-02 PASS realRoPE retained consumers, C1/C2/C6,1M bounds and stream guard.55CPU+Ruff passed. Candidate-only combined16layer gate06 next, reusechain05 timing/state hashes.

- continuation-chain-06 FAILED full-state hash gate on all4ranks;32step tokens match archive butstate hashes differ. Stop at correctness; no serving launch/no speed accepted. Old harness did not persist actual state beforeassertion; diagnostic07 fixes this evidence gap with untimed native-vs-original position copies underoneprocess and saves named tensors.

- Position v1 missing dependency confirmed statically: stockcopy_data_within_device only synchronizes prior compute when both stream allocator flags enabled; installed getters False/False. Early position overwrite needs explicitcompute→DMAevent. Sourcefixed andcapabilitybumped2; oldv1mustberejected. Diagnostic07 keepsoriginalfrozenv1 toexplain actualstate discrepancy, notperformance. v2CPUbuild waitsforitsretirement.

- diagnostic07 confirmsv1 native/original states differ all4 whiletokensmatch; namedtensor diffs preserved. Correct original copies match allnon-packedstate archivehashes; somepacked main/index archivesdiffer independently. v2addsmandatorycompute→DMAevent,cap2;build112CPU passed. position-copy-hpu-03 passed32retained early+lateRoPE16MME chains,C1/C2/C6 and1Mrange. Chain08reusesoldBtiming and reconstructs onlyuntimed sameprocess original-copy fullstateoracle; no byteexcluded.

- `continuation-chain-08`: v2 exact full state/C2/C6, 8.670933→8.993627ms component regression; rejected, retired exit0. No full model. See QUALIFICATION.json.

- `build-position-03`: same-compute-stream position copy capability3;112jobs build0. `position-copy-hpu-04`: retained delayed consumers exact,1passed;not performance.

- `engram-diagnostic-spans-cpu-01.log`:19 CPU native C1/profiling checks passed,30 HPU-only cases skipped intentionally without lease. New RAW labels separate packet reuse, native gather and upload. Ruff/py_compile passed.

- `continuation-chain-09/QUALIFICATION.json`: position v3 C1 exact, 8.873217ms vs8.670933; rejected, larger bucket checks finishing. Normal path restored to original position copy.

- chain09 final:4×component_exact includingC2/C6;exit0,zero group survivors. v3 remains rejected.

- `COMPANION_TP4_QUALIFIED_INVENTORY.json`: recovered sibling branch actual20.533651ms EOS/memory qualification. Not yet merged into this continuation source;19.42ms later projection is not measured. Reuse component/quality records;do not remeasure B.

- `DIAGNOSTIC07_REPORT.md`:192tokens exact;fourRAW captured;32CPUcycles;6commonhardwarecycles27.777971ms(diagnostic),globalempty10.024833ms/cycle;two11ms gaps before scalarpositioncopy. Callback0.02–0.18ms;servicefullyretiredafterforcedcleanup.

- `position-copy-compiled-hpu-01`:32retained delayed consumers/sharedC1C2C6/full1M bounds exact, fixed3compiles. Framework-copy graph, no rawDMA. Not a latency measurement.

- `combined-native-01`:exactMHC parent native source plus7 verified expert/scale files;112jobs build+ctest passed71.42s;new profilecombined-v1. `combined-source-cpu-01/02` failures preserved:legacy testfixture mislabeledcompact export and TP2 constructor touchedTP4-only metadata/options. Updatedv3/v4 CPUfixtures, scopedTP4 constructor;`combined-source-cpu-03`:63passed,Ruff passed.

## Combined chain10 passed
Four ranks, actual 16-layer feedback chain: reused B 8.670932719 ms → 8.199178703 ms, 5.441% gain. 32 C1 tokens and same-allocation original-copy mutable-state oracle exact; C2/C6 slot rebind/reuse exact; fixed 3 position-copy preparations, no hot compile. Exit0, no surviving owned processes. Contains compiled position copies, fallbackEngram, packetcompletion, MHCSIMD plus recovered qualified pair2/compactscales/nativeindexscore. Full40/API/EOS still required. Next serving08: one formal16K→EOS, both requested traces; archived20.533651ms reference reused. [Evidence](continuation-chain-10/QUALIFICATION.json).

- Held packed MLA projection gate audited: single-card proposed fixture omits production AllReduce between wo_b and hc_post; not a valid promotion gate. No device run. Extend to true four-rank producer→reduction→consumer before any performance claim. See PACKED_PROJECTION_GATE_AUDIT.json.

- serving08 stopped before sending any warm/formalrequest: allfour ready copypreparations7 vs expected6. Loader prepares outside inference mode, unlike decode and passing component. Extra C1 graph0007 archived; fix cold preparation mode, reproduce changedcontract in HPU02. No speed result/no B rerun; ownedservice retired. See serving-continuation-08/STARTUP_FAILURE.json.

- Startup mode mismatch closed: CPU same-mode6→inferencefirstC1 count7 reproduction; prepare_compiled_copies now inference-mode decorated. HPU02 mirrors normal loader allocation/preparation outside inference;6shapes and32 delayed-consumer cases exact, counter stays6, exit0. Reuse chain10 execution/timing because its prepare was already inside inference. Fresh serving09 resumes the still-unmeasured combined candidate, one formalEOS and bothtraces.

## Serving09 formal decode improved; prefill regression remains
ONE formal16K→2989naturalEOS exact: 17.497869334ms/token, 57.149815tokens/s; reused archived20.533651ms, 14.784% latency improvement. Client ITL p50 16.698536ms/p95 23.231675ms. Full40 normalAPI and1M/8192/32 retained. Prefill 26.656797s vsarchived~16.3s regression; investigatebeforeproductionpromotion. Warmup58.328789ms is unscored cold request. Decodecapture succeeded; bothtraces/analysis pending. Target10msNOTachieved. See serving-continuation-09/FORMAL_RESULT.json.

- serving09 postcapture correction: bothtraces4ranks complete/all4requests2989EOSexact/ownedPGID retired(afterforcedworkercleanup). Tracedprefill16.483395s, C8192 CPUenvelopes16.090384s≈companion16.023441s; formal26.656797s remains a valid unexplainedoutlier, notprovenkernelregression. Do not replaceformalvalue with profiledtime. See PREFILL09_ANOMALY_REVIEW.json. Decode32cycles all4 preparationcountsunchanged; nativeprefix32starts/rank.

- serving09 full four-rank decode analysis COMPLETE: 31 complete periods, mean17.807635ms = attention6.319720 + MoE2.954786 + mHC/aux3.884831 + cross-group1.471799 + DMA0.876200 + null0.138283 + gap2.162017. Trace separate from formal17.497869ms. All rank raw pairing errors0. Full tables: serving-continuation-09/{coarse-decode,latency-report,kernel-report}.
- GAP_REVIEW: strict interval closure; <10us holes contribute1.749246ms/token; >=100us holes0.120553ms/token. Longest single gap1.755331ms occurs once across31periods. No evidence to label all holes CPU or avoidable NIC wait. Next work must reduce compute/dataflow too.
- FullR1 selection dependency audit: frozen qualified source recovered; old0.310958ms improvement is a four-layer component with hot-cache dependencies, not additive E2E credit. No code selected or hardware launched yet. Packed projection four-rank AllReduce+hc_post fixture corrected; held.

- packed-projection-chain-01: exact source/native dependency audit passed. Chosen attention output chain, required actualTP4 AllReduce+hc_post feedback, four distinct real projection weights, changing queries. Missing matched component parent measured once; no fullmodel or archivedB repetition. See PACKED_PROJECTION_CHAIN01_PLAN.json.

- packed-projection-chain-01 COMPLETE:64rank/fixture/layer bit-exact cases + fourrank feedback exact, real AllReduce/hc_post included,48x8samples; exit0 no survivors. Four-rank max interval median 1.489276125→1.384973ms per four projections. Graph/native/defaultC1/C2/C6 integration pending; no E2E credit.

- combined-projection-native-01 built112jobs70.3s+ctest; current combined parent plus only hpu_dsv41_selected_mla_pt2.cpp changed. DefaultTP4C1 integration has no opt-in and retainsC2/C6 fallback. Continuation-chain-11 queued: reuse chain10 timing, reconstruct original-projection same-buffer oracle untimed; no full model.

- continuation-chain-11 C1 token+fullsame-bufferoriginalprojectionstateexact, but component8.199179→8.42834871875ms regressed. StopremainingC2/C6 compilation; nofullmodel. Originalprojectionnativecomponentgain doesnotqualify actual16layerchain. Source/defaultdispatch rollbackto17.497869parent; graphplacementdiagnosisnext. See PERFORMANCE_DECISION.json.

- chain11 retirement correction: atstopcheckpointallranksalreadycompletedC2/C6/C1exact, no signal was sent. Runexit0 andnoownedgroupsurvivors. Performance rejectionunchanged2.795%slower.

- packed-gather-native01:112jobsbuild+ctestpassed71.6s; original168TPCcode sectionsidentical, newdirectdecoder406instructionlines,no vectorlocalspills. Rejectedprojectiondefaultrestored; newdirectopunselected. packed-gather-chain01 candidateonlyconsumer gate queued,reuseoriginalparentcomponenttimings,40additionalfull-serving-pool/invalidsource/lengthchecks.

- packed-gather-chain01 PASS:64realrank/fixture/layer +40fullpool/invalid/lengthcases +4feedback exact. OriginalB timingsreused; fourrankmaxmedian wall1.489276125→1.37767ms/fourprojections. NewdirectTPCexecuted; PVMMEBF16SRAMverified. Native/defaultnormal16layer gate12next, reusechain10B.

- IMPORTANT correction chain11/12: new independentparent-attentionoracle changeslast-executedrecipeworking set; no candidate rewarm before32timedsteps. Chain10 oracle usedsamecandidate16layergraphs32times. Chain12 first12iterationsslow thenlast16~6.7ms indicatesnonstationarysampling. Withdraw earlierprovenregression verdict; preservemeansascoldcontaminated/unqualified. See COMPONENT_WARMUP_PROTOCOL_CORRECTION.json. Newcandidate-onlywarm32+resetprotocol justified; nofullB repeat.

- VisibleprefixCPU01:53passed onfreeCPU20; retainedcompiledvariants andearlyprefixguards cover4Kboundary; same-tieFullR1/R2C1/C6mainIDs+candidateorder exact. SourceunqualifiedforHPU andexcludedfromchain13.
- chain13 justifiedcandidate-only warmprotocolretry: frozenchain12source/native plus32candidatewarmstepsafterdifferentoracle;resetbefore32timedsteps. NoownedCPUheavyworkduringrun. Baselinechain10reused.

- chain13 externalwarm32 didnotremove nonstationarity afterCPUfingerprints+reset: first8~11.3ms,last16~6.62ms,overall8.3615ms. Correctnesschecksretained; notqualifiedgain/regression. Nextprotocolmustwarmandtimeonecontinuousstatechain; nointerveningrestore/CPUstatehash. Existing32stepresetBcannotanswer newtimingboundary; measuremissingcomponentB oncebesidecombinedcandidate, notfullEOSB.

- chain14 queued: directpackedgather+visibleprefixdefaultsource combined. New continuous32warm+32timed boundary withone necessarymissingcomponentoriginalreference; oldshortB unsuitable afterdocumentednonstationarity, fullEOSB notremeasured.65tokens/fullmutablebytes andC2/C6/rebind/C1prefixboundrevisit gates;53CPU+Ruffpassed. NootherownedCPUheavyworkduringrun.

- chain14 continuousgate PASS: matchedmissingcomponentB 7.35820784375→6.577623031250001ms, 10.60835503801406%improvement.65tokens/fullcanonicalstateexactallranks;C2/C6/rebindandC1bound20K→24K→20Kexact, nohotcompile. Current16layergate notE2E. FullR1unchangedpublicationcontractreuse indexed. Nextoneformal40layer16K→EOS andrequestedtraces; oldformal17.497869ms retained.

- serving10 ready: component14 passed10.608%matchedcontinuousgain, allstatechecks; FullR1 unchangedpublicationarchive reviewed. One unprofiledformal16K→EOS versus17.497869msB, plusrequesteddecode/prefilltraces. Nativeeligibility and4Kbound/runtimepreparationstats nowexplicit; traceclassifier recognizesdirectgatherandMLAQK/PV. Sourceauditsnocalculationchangesaftergate.

- serving10 ONE formal16K→2989EOS exact: 16.563728644ms/token,60.372880tokens/s; reused17.497869msB, saved0.934141ms (5.3386%). Combineddirectgather+visibleprefix;individualcreditnotmeasured. Bothtracesrunning;runtimepathauditpending.10msnotmet. See serving-continuation-10/FORMAL_RESULT.json.

- TP4tile-selection CPU helper contract:21passed; source/reindexunsortedIDs+candidateorderexact onCPU, withcutoffties,1M-IDFP32transport,paddingandinvalidrows. Unselected/noHPUclaim. Firstfixturefailed4transportassertions becausehighIDwasnotselected; correctedfixtureandpreservedcpu01. See TP4_TILE_SELECTION_{PLAN,CPU}.json.

- serving10 fourrankdecodeanalysis COMPLETE:31periods17.021524ms = Attention5.829707 + MoE2.953686 + mHC/aux3.736889 + crossgroup1.295718 + DMA0.914965 + null0.163204 + gap2.127356. Rawpairerrorsall0; newpackedgatherobservedall4ranks. Gap<10us contributes1.746050ms; no gap>=0.5ms. See serving-continuation-10/{coarse-decode,latency-report,kernel-report,GAP_REVIEW.json}. Fullsemantic reviewpassed; formal16.563729ms staysseparate. Nexttile-selectiongate justifiedbyremainingreplicatedscoring/sort; CPU46passed.

- tile-selection-chain01: all4ranks11checks passed (sixsynthetictie/transport +continuousstate/output +C2/C6/24K/20K); exit0 noowned survivors. No measurablecompletechaingain: eventmedian3.132669→3.677774ms; hostwall5.584373→7.098503ms. Unpacedhostwall includeswarmqueuebacklog; do not treatasE2EorsteadyTPOT. No hotcompilation/fallback. Unselected; reviewextraF32collective andrecipeboundaries, nofullmodel. See QUALIFICATION.json.

- tile-selection-chain02 ready: concretegraphfix removes two existing BF16query/weightgathers by coldreplicated projectionweights; retainsoneF32resultgather. Sameoriginal shardGEMM arithmetic,53CPUchecks/Ruffpassed; extra estimated61.875MiB/card finalresidentweights (no largeKVmirror). Correctedhosttimer drainswarm work withoutreset beforetiming; one justifiedmatchedcomponentparent, nofullEOSB and no priorfailedcandidate rerun. See TP4_LOCAL_QUERY_PLAN.json and TP4_TILE_SELECTION_CHAIN02_PLAN.json.

- tile-selection-chain02 FAILED beforetiming: sixhardwaretie +15fullquery/headweightoracles perrank exact, but64-step realattention/hc_post feedbackoutput differs96.2%, maxabs10.0 (allranks same). No speedresults, noEOSrun. Newbatch_as_strided aliaswarning20occurrences absentchain01; causalroleunknown. Retainfailedsource/graphs andinspectalias/BF16boundaries before furtherhardware. Default flagsremainFalse. See QUALIFICATION.json.

- tile-selection-chain03 changed-source correctness gate ready: cold independent2D projectionbuffers removehotweightas_strided views (Bridgepass batches those as eagerbatch_as_stridedwithaliasmeta);53CPUchecks includingdistinctstoragespassed. Ifordinaryunpacedfeedbackfails, retainit andrunfirst-consumerdiagnostic, neverpromoteasynchronousfailure basedonpassing syncprobe. Correctedcomponenttimingonlyafterexactgate; B hasnotbeenmeasuredwithcorrectedboundarybecausechain02failedfirst. Nofullmodel.

- tile-selection-chain03 FAILED, no timing: coldindependent2Dweights didnotremovebatch_as_stridedwarning(20) orintegratedmismatch. Synchronizeddiagnosticlocalizesfirstdifferenceatstep1,L14,P16385: maxabs0.03125,13699/20480outputelements; currentSWA,selectedIDs,candidatepool exact, newlycompressedrow8192selected. This narrowsattentiontocompressor/history/packedmain/producer-readerdependencies; numericalcause notyetproved. Exit1 noownedPGsurvivors. See QUALIFICATION.json andfirst-difference-rank*.pt.

- State04 ROOTCAUSE: syntheticAttentionConsumermean losesBF16rounding whenlocalqueriesremovegraphcuts. CPUrealweightoracle: parentmatchesBF16-roundedmean F32projection(~3e-6), candidate matchesunroundedF32mean(~6e-6); wrongcomparison~0.0057/0.0079. ThusfirstFP32historydiff isfixtureBF16castelision, notTopK orprovenhistoryrace. Correctproducer tonormalnativeBF16mHCcollapse; preservedfailed02/03. Diagnosticexit0, no timing/EOS.

- State05 targetedfix ready: nativebf16_identity atnewlocalquery R2FP32compressorentrance preservesroundedinput exposedbyCPUoracle. Decodeusesgenericcollapse/RMS; earliernativecollapsewordingcorrected(prefillonly). Same2stepdiagnostic/no timing; reuse21unchangedquery/tieoracles withASTidentitychecks.

- State05: targeted native BF16 boundary restored exact histories, packed writes and outputs for all four ranks / first two steps / five real attention consumers. See `tp4-tile-selection-state-05/QUALIFICATION.json`; no timing. Next fixes fixture position offsets and evaluates continuous correctness and net benefit.

- Selection chain06: fixed BF16 boundary + zero-offset position inputs passed all26 checks/rank, including64 feedback and C2/C6/bounds. Corrected steady hostmedian 3.099097844 -> 3.506631813ms; device 3.075223625 -> 3.500239625ms. Slower, unselected, no E2E. First valid component B now archived. Four compiled parts remain despite2->1gathers; next query-only variant removes resultgather while retaining old sort. See QUALIFICATION/COLLECTIVE_DATAFLOW_REVIEW.

- Local-query chain07 ready: change from slower06 removes packet selection/gather entirely, leaves original localTopK. Tested query replication removes two original gathers; expect fewer compiler boundaries, verify actual plan. Reuse chain06 corrected parent timing, no B timing; full parent correctness run needed because prior outputtensor notsaved.

- Local query07 micro-qualified: parent reused 3.099097844 -> 2.903320391ms host; device 3.075223625 -> 2.884475250ms.26checks/rank exact;0hotcompile/fallback;4->3parts,2->0querygathers all20consumerplans. Nextnormal-loader integration+16-layer gate, notE2E gain.

- Continuation15 ready: localquery micro07 gains6.317%, normal loader/preparation/invalidation integrated,39CPUownership/lifecyclechecks+Ruff pass. Real16layer head/token/Engram/MoE chain;32warm32timed candidateonly; chain14 timing/65tokens/fullstate reused. No E2E yet; tilepartition disabled.

- Continuation15 unqualified:65tokens and26/32states exactly matchchain14; onlypacked main/index whole-pool hashes differ. These same6pools were alreadycrossrankdifferent inbaseline14. StaticR2prefill nullpage writesduplicate64slots, so crossprocessscratchmayinvalidatewholehash oracle; notyetproved. Host6.817812ms vs6.577623archive, no gainclaimed. Allworkersretired. State16 nowchecks64steps fromsameinitialbuffers, no timing, savesallbytes tolocalize; loaderautoenable removed.

- Offline reuse audit: literalconstant hoist andstandalone layout were already slower; no repeat. LogicalMLA consumer01 hascorrected42case/placementgate andcombined144rankstatecases, plusoldworkflow05EOS. Reuseimplementation, but oldB predatescurrentdirectpackedgather, so no additivegain. Existingselected_offsets8192vsmax6consumer wastes639.53125MiB/card; prior6-token implementation canbeported andstacked. See NEXT_DATAFLOW_REUSE_REVIEW.json; hardwarewaitingonstate16diagnostic.

- State16 PASS: sameinitialbuffers, full16layers,64steps/65tokens andall32mutabletensors byteexact all4ranks. Savedfullstatecrossrankdifference isexclusivelyreservednullpage rows32..63; allpopulatedlogicalpagesmapto>=1. Old14/15 onlywholehashes cannotlocalizeoldbytes; unsuitablecrossprocessoracle replacedwithdirectsamebuffer proof. No new timing. Querypath remainsunselected since15didnotbeat6.577623ms. LogicalMLAqualifiedsourcesnowstaged forcomposition, nohardwareuntilbuild/textverification.

- Logicalnative01 built112cores+ctestpassed;169existingTPC.text byteunchanged,newlogicalgather0vectorlocalspills.31CPUoffset/lifecycle/dispatchtests+Ruffpassed. Chain17 candidateonlytimingvsreused06currentdirectpackedgather parent; localqueryanddistributedtilesdisabled. Historicallogicalcorrectness reused; currentboundarymustmeasureincrementalgain.

- Logical17 micro-qualified: parent3.099097844 ->3.065881937ms host,3.075223625 ->3.041763750msdevice,26checks/rank exact. All4rankC1postgraphs retainKBF16SRAM/VFP32DRAM/PVBF16SRAM;169oldTPC.textunchanged. Incrementalgain1.07%, nothistorical5-13% andnotE2E. Querypathdisabled. FirstpostgraphclassifiermistookC2gathertilesforC1; correctedusingwholesoftmaxscopeandpreservedinvalidrecord.

- Continuation18 ready: logicalMLA+compactoffsetsnormal source; localqueriesdisabled. Reusechain14 timing, onecandidatecontinuous32warm32timed. Same-buffer untimedparentonlyforcanonicalstateoracle (state16provedcrossprocessscratchissue),65archivedtokens,C2/C6/rebind/bounds. No formalEOS yet.

- Offline boundedmirrorbudget: modelhas4KVowners(2/2/2/1), not8; full32Kdecodedmain+index=100MiB/card, oldmain2MiB alreadyallocated, logicalrows0.3125MiB. Compactoffsetssave639.53125MiB, netresidentestimate541.21875MiB savingbeforeworkspaces. V2prefix/request/pageinvalidationport remainsunimplemented; historicalhot-pathCIcrosseszero, noE2Eclaim. See BOUNDED_DECODED_CACHE_BUDGET_REVIEW.json.

- Continuation18 PASS: integrated16layer 6.577623031->6.192921906ms (5.8486%);65tokensandallcanonicalbytesexact,sameinitialbufferoracle;20rankcasesC2/C6/rebind/bound20K24K20Kpass; nohotcompile. Allownedprocessesretired. Logical+compactoffsetmicroqualified, E2Estillserving10 16.563729ms. SourcepathsandmemorylimitsinQUALIFICATION.json.

- Next scope narrowed to20MiB boundedindex-only mirrors; preservequalifiedlogicalMLA/packedmain, avoidporting100MiB main/SWAmirror behavior. Currentnativefullscorer still2048-columncap, so newC1wideeligibility/hostcontractmustbeexplicitlycomposedfromarchive. LifecycleV2prefix/request/pagebind tests required. No implementationorhardware yet. See BOUNDED_INDEX_ONLY_NEXT_PLAN.json.

- Offline current-serving10 Attention bottleneck review: all4 ranks MME union1.331–1.334ms vsTPC3.924–3.937ms/token. packedKV preparation~0.926 vsQK/PV~0.130; indexkeydecode~0.761–0.763 vsindexMME~0.052–0.053, per-rank row activity diagnostics, notadditivefourrankallocation. M1projectionweight-onceproxy1.66–1.92TB/s, no actualHBM/ALU saturationclaim. No newhardware. See serving-continuation-10/ATTENTION_BOTTLENECK_REVIEW.{md,json}. Boundedindexmirror partialimplementation remainsunqualified.

- Attention web/source research: official vLLM DCP, FlashInfer sparse-load pipeline, DeepGEMM 26/09 V4.1 Sparse Indexer (pinned78b6900), IntelTPC mapping/coherency. Currentindexkey allRequired input/output verified; docsidentify fine-grainTPC/MME pipeline limitation but no millisecondcausalcredit. Rejectedtile06 preserved3.099->3.507ms; no repeat. Plan retainsboundedmirror first, targetedmetadata/pipeline audit and newdistributeddesign onlyifchangeddependencycontract. No hardware. See attention-reference-review-20260929/REPORT.md andSOURCE_MANIFEST.json.

- Indexmirror19 ready: bounded20MiB derivedkeys defaultTP4capability; canonicalpage/request invalidation+V2firstconsumer guard integrated.44new/offset+48existingstate/V2 CPUcases,Ruff pass.112core nativebuild70s+ctest,7Metaboundary cases;170oldTPC.text unchanged. OriginalC8192prefill cap retained in frozen native; unrelatedW C16384 widening excluded. Reuse logical17 candidate3.0658819375ms, noB timing; botharmsmaintainmirror, originalscores asuntimednumericoracle. See TP4_INDEX_MIRROR_CHAIN19_PLAN.json. No E2Eclaim.

- Indexmirror19 PASS:3.0658819375->2.74498396875ms(10.4667%),26checks/rank exact inclC2/C6/bounds;all4rank candidateC1no rawindexkeydecoder, MLA KBF16SRAM/VFP32DRAM/PVBF16SRAM retained.40C1graphs(20mirror/20parent),old170TPCtextsame,retiredexit0. Firstpostgraphparser usedlowercaseGEMM only; corrected case matching, nothardwarefailure. Integrated20 nowuses18 timing6.192922ms andsamestatepackedscoreoracle, withactualrequest/page/32K hardwarechecks aftertiming.

- Continuation20 measured6.192921906->5.951498375ms,65tokens/allstate exact, nohotcompile,C2/C6/rebind and5mirrorlifecyclecases passed; overallUNQUALIFIED at>32K firstfallback: PreparedGroup sharedDynamo codehit8variantlimit onrotarysize. Retiredexit1. Fixcoldexport perfinitecontract+mirror modecachekeys,53CPUcases pass inclforcedlimit2/8buckets. Contract21 isminimal5actualattention/FullR1 transitiongate, no performance rerun orfull16reload. See QUALIFICATION andTP4_INDEX_MIRROR_CONTRACT21_PLAN.json.

- Contract21 PASS: all11cases×4ranks output/state exact, including C2/C6→C1, mode toggle, request/page reuse and32K→64K→32K. Separate cold prepared-code fix resolved compilelimit; sixvariants/group, no limit increase. Exit0/no survivors. Combined19/20/21 component-qualified;20 original failed record preserved withCONTRACT_COMPLETION. No new timing/E2E claim.

- Serving11 ready after19/20/21 combinedgate. One formal16K→naturalEOS plusrequesteddecode/prefilltraces, reuse16.563729ms formal10; no B timing. LogicalMLA+compactoffset+20MiB boundedindexmirror, defaultTP4selection, full1M/8192/32. Postgateonlyreadonlypathstats; native frozenqualifiedsource avoids unrelated16384nativewidening.3.898%component gain is notE2Eforecast.

- User deferred new E2E and traces before serving11launch (no run directory or requests). Persistent developer ledger: docs/dev_guide/deepseek_v41_tp4_optimization_ledger.md. PendingP01/P02 measured16layer6.577623→5.951498ms:0.626125ms/9.519% cumulative. Onehistoricalabsolute-savings calibration gives estimated0.749295ms/4.524% E2E,15.8144ms/63.233tps; NOTMEASURED. Source/state/planupdated; launchcontrollerguarded against accidentalstart. SeeE2E_ESTIMATE_20260929.json.

- Offlineattention dataflowaudit20260929: currentmirror removesrawindexdecoder; remainingstreamedTopK, replicated32headscoring,38selectedmainconsumers/8distinctgroups, KSRAM/VFP32DRAM, Reindex16gather/16GEMMs/8reducers observed. OldtraceF32bitonic~2.041ms/card diagnostic only, notnewtiming/additivewall. Newdirectionsnotgainentries; pendingP01/P02unchanged. Userclarifiedpositive-measured-micro-onlyledgeradmission; AGENTSandledgercorrected. SeeATTENTION_DATAFLOW_AUDIT_20260929.{md,json}.

- Attention proposal added toofflineaudit, NOTgainledger: AFullR1batchedlocalTopK preservesorderedmerge/ties; historical.310958ms completechain supports current-parentplanning.2–.3ms only. Bgroup4selectedmainreuse38→11prepares; equal-row-cost oldactivityproxy.526575ms, engineeringtarget.2–.4ms unmeasured. Conditionalcombined.4–.7ms target, no additivequalifiedcredit/noE2E/newmicro. P01/P02unchanged.

- FullR1batch22 ready: normal-source mirror prefix+batched independent local TopK; originalmerge/tie/IDpublication preserved, no newcollective/nativebuild.58CPUcases/Ruffpass; helperarithmetic AST matchesqualifiedarchive. Reusechain19candidate2.744984ms; untimedstreamedmirrororacle+9newhardwareties+realL20→L24consumer. No E2E or gainledgerentryyet.

- Batch22 completedexact35cases/rank (9newties+64feedback+C2/C6/bounds), nohotcompile/fallback; wall2.744984→3.661430ms, regression33.39%, defaultdisabled/noledgerentry. Native/runtime unchanged; graphmainTopK19→10 yetDMA29→47,4parts/twoquerygathers unchanged. Investigateactuallowereddataflow before any retry. Originalevidence preserved; allownedworkersretired.

- Batch23 changedcontract ready: remove unintended nativecarried-ID lowering from22, preserveparent score-onlyTopK+gather and onlybatchindependentlocalrows. Same MME contracts/collectives;22static DMA29→47 addedIDmovement notprovenwholecause.58CPUcasespass; candidate-only vs19, no newB/E2E.

- Batch23 exact35cases/rank, exit0 retired; score-onlyTopK restored2.742722ms vs19reference2.744984ms.2.26us delta unresolved; no gainclaimed/no ledgerentry/defaultdisabled. Changedsource removed22large regression, implicatingcarried-ID lowering; no timedkernelattribution. Nexttargetselected-mainreuse, notadditionalbatchsweeps.

## 2026-09-29 selected main reuse (component24, pending)

- Normal C1 TP4 four-layer groups create an ephemeral ownership-keyed workspace. R1/R2 publisher preserves the logical gather codec/mask and independently publishes BF16 rows; later consumers decode their own SWA and reuse main slots 128..639. No new collective or cross-token/request state.
- Full40 topology reduces packed-main preparation38→11 within existing group boundaries; per-layer K/V/mask materialization and QK/PV arithmetic remain. First publication adds640KiB BF16 output and2.5KiB mask per unique current-group selection. This explicitly trades extra first-layer stores for27 avoided page/FP4 decode sequences.
- Qualified parent: P01/P02 index-mirror native bundle. All170 previous TPC.text sections unchanged. New independent TPC publish/reuse kernels and compound attention wrappers; no unqualified native changes copied from the working tree.
- CPU gate:42passed, including TP4C1/C2/C6/prefill andTP2, index-owner/KV-owner changes and repeated calls with mutated contents. Native preflight adds16exact compiled edge cases/rank before the full16-layer feedback gate.
- Timing anchor: continuation20 candidate5.951498375ms, with original failure retained and contract21 completion linked. Candidate-only32warm/32timed; one untimed same-buffer parent. No new E2E or trace.
- Pending evidence: compiler K/PV placement and cache traffic, complete-chain timing/state. No gain-ledger entry before these pass.

### component24 closure

- Qualified micro gain:5.951498375→5.910531516ms, saving0.040966859ms (0.688345%). Device-event median5.8085415→5.783713ms. Allrankwall/event comparisons positive; one aggregate32-stepwindow, not four independent repeats.
- 65tokens/fullstate exact;16nativeedgecases+5bucket/rebindcases perrank; nohotcompile; process exited0 withnoowned survivors. Compiler K/PV SRAM contract preserved; stableC1 groups43parts,32AR,8AG unchanged.
- New P03 entered the fixed gain ledger; P01/P02/P03 measured cumulative0.667091516ms fromchain14. Only lowconfidence E2E forecast updated to0.798321103ms saving; old P01/P02 forecast preserved. No E2E/trace run.
- Limited-gain diagnosis: eachlayer still materializes640BF16K/F32V rows, gatherlaunches remain, andfirstpublisher addsBF16DRAMcopy. These are verified structure, not measured attribution of the unachieved0.2–0.4ms forecast. Further work must target the surviving producer-consumer handoff, with exact-arithmetic constraints, rather than repeat batching sweeps.

## 2026-09-29 whole-decode fusion campaign: component25 pending

User renewed the full-route audit and requested measured micro gains mapped into the ledger; E2E remains deferred. Current formal16.563728644ms means6.563728644ms reduction is required to reach10ms (6ms alone gives10.5637ms). Current pending forecast0.798321103ms is not achieved E2E.

Group-local decode coordinates are the first source-proven repeated producer: chain24 L3 graph still executes constant_i64/cast/div_mod for the same ring position and repeats640lengths. Lift to one bitmask coordinate and one length producer per existing four-layer group, retain C1/C2/C6 and unchanged state writes;49CPUcases passed. Reuse chain24 timing and native bundle; benchmark25 is candidate-only. No gain booked yet.

- component25 complete: fourrank65tokens/fullstate and5C1/C2/C6/rebind/boundarycases/rank exact; nohotcompile; retired. Host5.910531516→5.903398891ms (delta0.007132625); event5.783713000→5.757699250ms. Small host delta below saved parent rank interval spread; no standalone gain-ledger credit or newB. Combine with the independently designed feature-tiled post+collapse fusion and compare the actual combination to24.

- Native feature-parallel post+collapse built (112CPUs after25 scoredwindow ended),ctest and13CPUMeta/control tests passed; all172oldTPC.text unchanged. Old serial collapse+norm rejection avoided: retain40feature tiles, original control/RMSNorm consumers. Combined26 reuses24B, disablesboth metadata andpost/collapse in untimed oracle; no standalone25credit. See CONTINUATION_CHAIN26_PLAN.json.

- component26 rejected before modelweightload or timing: sameoneBF16residual mismatch onall4ranks atC2/0.03125; nativepreflight stopped. No performance result, no gain. Savedsource/graphs preserved; one-card rounding diagnostic27 captures missing operand values to identify arithmetic mismatch.

- Rounding27 startup failed beforedevice math because explicitHPUcoreimport was missing underAUTOLOAD=0; fixed in28. Rounding28 reproducesandarchives firstdifference. Full40960-elementCPU reconstruction singlesoutfinalFMA contraction vsseparate product/add; v2changes onlythis arithmetic boundary. Extended96cases/rank beforefurtherdevice timing; no gainbooked.

- Diagnostic29 observes originalFP32mix while preservingall5originaloutputs. Serialseparate sourceproducts matchall40960FP32elements exactly; pairwise/crosspair/serialFMA differ11401/15941/16526elements. Thus preserveoriginal0→3separate sourceaccumulation andcontractonlyfinalprojected product/add. Nativev2builtandold172TPC.textunchanged; combined30 requires96nativecases/rank beforefull16layergate.

- component30 qualified: combined post/collapse fusion + group metadata, chain24 5.910531516→5.747970984ms (−0.162560531ms/2.75035%); event5.783713→5.602245ms.96native+5lifecycle cases/rank,65tokens/fullstate exact, nohotcompile, exited0/retired. Five arithmetic/cast nodes→one feature-tiled native boundary; FFN/MME tensor contracts,43parts/32AR/8AG preserved. P04 added once;25 unresolved delta not added. Cumulative measured0.829652047ms, lowconfidence E2E forecast0.992860382ms saving,15.570868ms/64.2225tps. No E2E or trace; goal unmet. See continuation-chain-30/QUALIFICATION.json andDATAFLOW_REVIEW.json.

- component31 prepared: same qualifiednative extends post/collapse to10neighbor-layeredges per16layers (28/40source), excludesgroupend/draft/Engramentry. Ephemeralhandoffonly; explicitnativeBF16 replaces redundantidentity.13CPUcasecontracts passed including freshcalls/gaps/disabledlayer/C2/C6/prefill/TP2 andEngramguard; firstguardfixture lackedtensor, correctedwithout runtimechange. New16case actualattentionnorm/quant/QKV gate beforeweights; parent30reused; no E2E/nativebuild. No gaincredited pendingmeasurement.

- component31 rejected beforemodelweightload/timing: postresidual exact butC1genericattentionconsumerQKV1455/1792values differ onall4ranks. Nativev2BF16collapse contract isvalid forFFN, notyet for genericattentionfuser. Defaultinterlayerflagdisabled, no gaincredit; freeze/source/operands archived. StaticparentgraphshowsFP32collapsedoutput directlyfeedingRMSNorm; diagnostic32 will observeexistingFP32value while assertingunchangedoriginaloutputs, untimedonecardonly. P04 remainsqualified.

- Diagnostic32 originalresidual/QKV outputs unchanged. Captured5120FP32collapsevalues allmatchserialunroundedFP32; explicitlyroundingBF16changesall5120, pairwisesumdiffers1146. Thus genericattentionfuserelidescollapseBF16cast. New separateFP32collapse schema preserves deployedconsumer contract; FFN BF16native unchanged. Build03 onlyaddsnewFP32kernel/registration fromqualifiedv2frozen source, no arbitraryworkingtreenative overlay. No timingcredit.

- component33 ready after32rootcause: separateFP32collapse schema preservesgenericattentioncontract, BF16FFN pathunchanged.173oldTPC.textbyteidentical,112corebuild/ctest/23Metacasespass.16newQKVconsumerpreflightcases beforefull16layergate, parent30timingreused. No gainbooked.

- component33 stopped beforeweights/timing: BF16residual exact, newFP32collapse causedall1792QKV valuesdifferent. Newoutputuseddefault non-linear BF16all-lane conversions butstoredF32v1/v2contiguously; existingselectedKV usesexplicitSW_LINEAR. Build04correctsbothinputandBF16outputconversions tolinearorder, leavesarithmeticandqualifiedBF16kerneluntouched. No gaincredit; no fullmodelrun.

- component34 ready: fixnewFP32output tolinearTPCconversionlayout. OnlyoneTPCsourcechangedvs33; reuse23unchangedMetacontractcases;173qualifiedparentTPC.text unchanged. Parent30timingreused,16preflightQKVcases beforefull16layerload/timing. No E2E/trace.

- component34 decisiveC1 regression: 5.747970984→5.910173391ms(+2.822%), event5.602245000→5.762973750.16nativeQKVcases/rank and65tokens/fullstate exact, nohotcompile. C2/C6/rebindsweep intentionallystopped onownedparentSIGINT afternegativeC1; unfinished, notqualified. Allownedprocessesretired. Graph6post/collapse/identitynodes→1, MMEcontracts/43parts/32AR/8AG unchanged; no causalperkernelcostallocationwithoutnewtrace. Interlayerflagdisabled, P04 remainsdefault; no P05entry.

- Nextcandidate35 targetsmeasuredstructure ofpair2SiluQuant: two rowworkpoints eachloopover5tiles. Split BF16activation+tilemax andtilequantization into5xrows workpoints withoneextraTPCdependency andbounded2.6KiB intermediates/pair. ReusequalifiedP04parent, notslowerinterlayer. Native staticimplementationstarted, no timing orledgercredit; preserveoriginalscalar helpers/order/packing, requireFP8byte/scale checks andactualW2/fullchainconsumer.

- component35 ready: newfeature-parallel activation/max +quant stagedbody, defaultTP4C1specializedschema undernormalPreparedMoE.173oldTPC.text unchanged,22Metacases/ctestpassed. ExplicitGaudi2assemblyparent2ld_l_v/1st_l_v instruction sites, candidatebothstages0/0; nottimingproof.73nativebyte/scalegates beforeweights andcomplete16layerchain vsparent30; layerhandoffdisabled. Extra2.6KB intermediatesandoneTPCdependencyperpair included.

- User explicitlyresumedfullTP4E2E+traces aftercurrentMoEcandidate. Authorizationrecord E2E_RESUMPTION_20260929.json; previousdeferralrevoked, oneformal16K→EOS thenindependentdecode/prefilltraces, archivedB reuse. Candidate35stoppedatsharedDynamoOpOverloadPacketcachelimitbeforeweights/timing, no numericalfailureobserved.36changesonlydistincttestwrappers,nativeandcontractsunchanged; no globalrecompile-limitincrease.

- chain36: rejected feature-parallel MoE SiluQuant. Native 73 cases/rank and 65 feedback tokens/state exact, no hot compile; full chain 5.747970984->6.164299000 ms (+7.243%). C2/C6/lifecycle sweep stopped after decisive C1. Owned group retired. Two TPC stages replace one, W13 outputs promote to SRAM but net latency regresses; no P05/gain credit. Normal feature_silu=False; P01-P04 selected for user-authorized serving11.

- serving11 ACCEPTED TP4 C1 16K decode: one formal14.622559ms/68.387482tokens/s, vs16.563729ms(-11.72%). All4requests2989tokensEOS1exact; semantics passed. P01-P04 closed, pendinggain0. Decode31cycles15.363824ms fully allocated with4rankpathproof. Prefill captured but offlineanalysis cancelled bylatestuser; raw/partial artifacts preserved. Oldprefillclientwall-clock boundary incompatibility recorded, no fix/retest performed. Allowneddevice/analysisprocessesretired. See serving-continuation-11/REPORT.md.

- binding-preparation-37: source-guided hot binding candidate from serving11/chain30; native metadata comparisons + single flatten + fixed result reconstruction. Recipes and standard TP4 collectives unchanged. No timing/ledger credit yet; full joint replay still requires four-rank HCL template and completion relocation work.

- continuation-chain-37: rejected for no complete-chain gain: 5.747970984→5.943104531 ms; all65 tokens/full state and C2/C6/rebind exact, no hot compile. Binding-only CPU40.423→17.837us did not produce net gain. Python default restored to serving11/chain30; metadata helper remains unused. No E2E or gain-ledger entry.

- recipe-submission-38: next source-targeted candidate prepares per-recipe allocation/alias metadata and combines Python/C++ marshalling into one native call. Reuses ordinary GraphStorage and four-rank HCCL semantics; fresh allocations prevent replay-output lifetime changes. Not joint command replay. 37 defaults fully restored; no ledger credit.

- recipe38 build01: compile/link succeeded but import rejected unavailable torch::jit generic Python type-inference ABI symbol. No HPU run. Use the existing bridge scalar/Tensor conversion contract and explicit output wrappers; unsupported structured inputs retain ordinary calls. Failed build/source/log preserved under linkage-attempt-01.

- continuation-chain-38: no resolved full-chain gain 5.747970984→5.749763203ms; event effectively unchanged. Four-rank B1/B2/B6 preflight,65tokens/fullstate/slot rebind exact;157native recipes/rank,3425calls,0unsupported, nohotcompile. Exit0/no survivors. Defaults restored; no gain credit. Further Python-wrapper tuning stopped; full command replay remains separate from this adapter.

- mirror-partition39: targets fourfold replicated prefix scoring/local sort using P02 bounded mirrors. One contiguous native scorer per active rank, original2048 tile TopK and global sequential merge retained; no query weight replicas, no per-tile KV decode.45CPUchecks and5actual score/tie/transport cases perrank exact before weights. Candidate-only16layer chain running vsparent30; runtime/TPC unchanged; no gain credit yet.

- chain39 invalid before timing: rank3 empty range at5tiles made its candidate FP32 gather independent of query producers. Actual lowered collective order differs (packet gather3 vs8); rank0/rank2 orders match. Five standalone score/transport cases were exact but did not contain upstream query gathers. Stop owned launch, preserve graphs; fix balanced nonempty ranges and extend gate to include actual query gathers + downstream consumer. No performance/ledger credit.

- component40 closed/rejected: balanced nonempty ranges fix39collective ordering; all4ranks five preflight cases+65tokens/fullstate+C2/C6/rebind gates pass. Complete16-layer host5.747970984→6.253309750ms(+8.792%), event5.602245000→6.124193250ms. Exit0/no survivors; tp4_mirror_selection default disabled. No ledger gain, E2E or trace. See continuation-chain-40/QUALIFICATION.json.

- Candidate41: pack original BF16 query[1,8,128]+headweight[1,8] into one1032-element/rank exchange; unpack rank/head order before unchanged score/TopK consumers. Bound to normalTP4C1decode validmirror; no weight replication or partition40.74CPUcases/Ruff pass. Fullproducer+consumer preflight and candidate-only16chain planned against archived30; no gain yet. See query-packet-41/PLAN.json.

- component41 rejected: fourrank8preflightcases andC1 65tokens/fullstate exact, nohotcompile; host5.747970984→6.729515531ms, event5.602245→6.554410750ms. AG8→5 but compiledparts43unchanged; BF16DRAMpack/unpack observed. Remaininglifecycle sweep stopped, notqualified. Source fully restored/archived; no gain/E2E/trace. See query-packet-41/RESULT.json.

- Candidate42 source audit: single main_reuse_gather TPC body replaces8partial64BF16 mainrowcopies by4full128BF16copies, widens both FP32halves for samePV. Original128SWA codec/mask/outputshapes/accessmaps and allcollectivesunchanged. QualifiedGaudi2assembly confirms8partial loadsites; no timingclaim. Build fromfrozenP04withonechangedTPCsource, existingBridge/hostlibraryreused. See main-reuse-vector-42/PLAN.json.

- component42 QUALIFIED pendingE2E: host5.747970984→5.673781281ms, saving0.074189703ms(1.291%); event5.602245→5.533051750ms. All4ranks65tokens/fullstate,16nativecases/rank,5C2/C6/rebind/boundarycases/rank exact; nohotcompile,57commonplanordersmatch,43parts/32AR/8AG unchanged, sameKSRAM/VF32DRAM andQK/PVcontracts.172otherTPC.text unchanged, onlymainreusecopy4full128vectors. Exit0/nosurvivors. P05 addedonce; estimatedfull40 saving0.222569109ms by27/9calls, NOTmeasuredE2E. No newfullmodel/trace.

- Candidate43: resume archivedsibling TP4native replay afterprobe07stall. Nativecapture/firstreplay previouslyexact; shortmonitorSID/mask lacks relocation while EDMAcompletionSOBrotates. Copy frozenHCL04 into isolatedbuild, patchpairedproducer/consumercontract; CPUfield/ringwrap tests then tinyfourrankactualcompute+RS+AGconsumer withsavedexactreference. NormalP05source/runtimeunchanged, no gaincredit. See native-short-monitor-43/PLAN.json.

- nativeprobe43 didnotstart devicecommand: engine-source option placedafter positional evidence was parsedas command. ArchiveFAILURE and originalcommand; probe44movesoptionbeforeevidence, sameCPU-qualifiedmonitorfix andsame135savedreference outputs. No timing/performancecredit.

- nativeprobe44 PASS: shortmonitorSID/mask relocation fixes archivedprobe07hang; all4rankscapture+134replays exactagainst135savedreference steps, realjointSynapse/HCLplan134replays andcompletionringwrap4. No fallback/per-node graphlaunch onjoint path; normalservingunchanged, no performancecredit. Larger CCBwrap/longtargetcarry gate next.

- Native45: four ranks each passed capture+16388replays, all outputs exact, 48CCB wraps,512completion-ringboundary replays, target6→65546. No modelperformancegaincredit. [evidence](native-short-monitor-probe-45/QUALIFICATION.json).

- Native46: diagnostic configuration rejected before kernels (implicit runnerV2); gate47 explicitly selects runnerV1. No correctness/performance credit.

- Native47: synthetic shared query slice unsupported before capture; replaced by actual-indexer-style separate score projection in gate48. No speed credit.

- Native48: ordinary step exact, native end-capture rejects synInvalidArgument;49streamdiagnosis. No timing/credit. User: quick benefit decision before expanded checks.

- Native49: isolated stream diagnosis reproduced capture failure; no performance result. Native50 preserves the first failure: hcclTp2NativeGraphCapture internal error 6; no timed replay. Native51 completion-relocation-count candidate built on CPU only, no device run; held at user direction to compare actual TP2 12ms paths first. All owned device processes retired; speculative native group default disabled; P05 gain ledger unchanged.

- Ordered-selection52: minimal one-card compiled contract gate rejected direct TP2 threshold/emit substitution at the first BF16 score case: 507/512 KV row IDs shared, five cutoff-tied rows differ (all score7.46875). Stopped before weights or timing; preserved exact counterexample. No production selection change and no gain credit.

- Native53: smallest compiled chain still rejects HCL capture with code6; no timing. Completion-count change did not establish a fix.54 adds explicit diagnostics for the manual rejection branches and records loaded native libraries; no arithmetic or scheduling changes. First54CPU command had invalid taskset arguments and did not compile; corrected build02 passed.

- Actual TP2 12ms vs serving11 comparison: attention including the previously auxiliary-classified TopK 4.032573→6.485386ms four-card union; MoE4.587352→3.224686, mHC1.187068→1.005615. Function unions overlap; no context-length causal assignment, no gain credit. See compare-tp2pp2-12ms-01/REPORT.md.
- Native54 pinpointed capture rejection: standalone AllGather phase2/count8, valid normal/wrap templates27/26 commands, identical external/internal completion deltas1/1. Old TP2 aligns these small payloads to128BF16; TP4 missed that producer/consumer layout. Candidate pre-partition padding preserves rank payload and wait dependency;10CPU tests passed.55 rejected stale copied shape metadata before capture;56 clears only new-node geometry and lets normal compiler rederive it. No timing or default enablement.

- Native56: corrected padding compiles and ordinary step remains exact, but a1024-element standaloneAG also rejects valid27/26 wrap templates. This disproves small-payload alignment as the rejection cause. Restored padding-pass/test changes; failed source retained in55/56.57 narrows the fix to phase2 wrap validation: require nonempty independently validated commands and unchanged completion deltas/relocations, retaining the old size-order guard for other phases. Short timing probe follows exact capture; no large state sweep first.

- Native57: phase2 wrap validation fix allowed capture; ordinary and capture steps were exact. First actual replay differed on1184/5120 BF16 output elements on all4 ranks (max absolute0.000244140625); timing correctly stopped. No speed or ledger credit.58 attempted intermediate-value diagnostics but exported views lowered to unsupported hpu.batch_as_strided before capture;59 uses lossless FP32 observation buffers and saves compiler graphs/first differing tensors, not a new trace.

- Native59 first divergence localized: all four first-AllReduce outputs exact; every own-rank query1024/weight8 slice exact; rank0 remote1/2/3 wrong and ranks1/2/3 remote rank0 exact but other two peers wrong. No simple permutation match. Saved tensors/graphs and FIRST_DIVERGENCE.json; send/receive buffer binding versus dependency remains unresolved. All owned jobs exited/retired; no timing/model/trace/ledger credit. See compare-tp2pp2-12ms-01/REPORT.md for complete current conclusion.

- Candidate60: static cause of59 matches relative-rank NIC context. calculateRemoteIndex uses send offset0 for standaloneAG, own rank for AllReduce AG. Native capture restores DW0/1/3/4 but omitted DW_REMOTE_RANK; stale rank offset predicts peer0/own-slice-only correctness. Capture now restores RRI for TP4, no arithmetic or padding change. First compile lacked original local include directory (no device); corrected isolated build passed. Exact intermediate gate immediately followed by small-chain timing if correct, before wider integration.

- Native60 PASS: restoring TP4 DW_REMOTE_RANK fixes59remoteAllGather corruption; all4ranks every observed stage exact and64continuousfeedback outputs exact. Synthetic host1.0876–1.1602→0.2580–0.2878ms; device1.0804–1.1141→0.2576–0.2708ms. Smallchain submission diagnosis only, no model gain credit.61 immediately runs real16layer candidate speed screen vs archivedP05chain42, preserving normal source math and native default-off until qualified. No newB/E2E/trace.

- Chain61 rejected before weights/timing: primitive-only V1 profile conflicted with real continuation async config.62 uses separate continuation profile preserving archived42 runner/async selection; native60 binaries unchanged. No gain credit.

- Chain62: all4ranks completed actual16K cache preparation; first native C1 plan failed output allocation (StopIteration), before capture/timing.63 reduces reproduction to first4 real layers/initialized zeroKV, archives exact output/alias metadata, skips long prefill and timing. Native60 smallchain gain remains synthetic-only; no ledger addition.

- Binding63 fixture set visible prefix before search bucket; fixed order64. Binding64 first four-layer ordinary plan output/fullstate exact; nextcapture rejected TP4phase0 RS5120 validwrap30/28 withtargetdelta1/1.66 generalizes independently validated wrap rule to allTP4physicalphases, preservesTP2guard.65 stopped because BF16-only Engram fixture does not reproduce62packed-host warm contract and oldHCL would repeat64;66 uses packed fixture plus source-directed HCL fix. No speed/ledger credit.

- Binding66 PASS: fourrank first4 real layers with packed Engram, zeroKV, three changing C1 steps (prepare/capture/actualreplay) alloutputs/fullstate exact. No performance credit. HCL66 allphase wrap fix archived and maintained patch regenerates all40 files exactly.67 extends samezeroKV binding check to16layers to isolate62remaining plan-output issue before another real16K continuation.

- Binding67 localized62: fused_0 packs BF16 index vector to68 bytes then index_copy_ writes cache, returns(), Bridge exposesNone and0outputallocations. Native planner treatedNone as one output.68 normalizes absent output and retains real recipe execution/mutable inputs withzerooutputs, C++sealing allows this onlyTP4. Build compiled successfully; first import loadedstockSynapse and rejected; corrected isolated build environment imports samebinary and writes matching ABI.16layerzeroKV numerical/state gate before real timing.

- Binding68 PASS: all4ranks16real layers packedEngram/zeroKV; prepare/capture/actualreplay alloutputs/fullstate exact. Four nativegroups,40collectives,44packedinputcompute segments,4actualnativejointreplays/rank. No speedcredit.69 starts actual16K completefeedback against archivedP05chain42; new --qualify-positive keeps state/lifecycle qualification in the same process only after all-rank speed gain, no newB/E2E/trace.

- Chain69 rejected for no speed gain: real16K completehead→token→Engram→16layers→head,32warm+32timed, median5.673781→6.048064ms(+6.60%); device5.533052→5.902828ms. All4ranks65tokens exact/nohotcompile;252actualnativegroupreplays/rank,43compute/40collectives retained. Native model-entry replay counter0;4per-group callbacks. Speed gate automatically stopped wider/fullstate checks. No ledger credit/E2E/trace; ordinarydefault remainsP05. See RESULT.json. Next source audit: historicalFixedDecodeInputs/model-entry bypass versus TP4all-input registration/per-group wrapper, with compiler-runtime attribution still open.

- Entry70 source audit: historicalFixedDecodeInputs excludesweights and bypassescompiledwrappers. TP4group69doesneither. Candidate reusesFixedDecodeInputs/nativepreflight anddirectprepared-group replay, retainsactualstate/geometry/lifetime gates.14CPUchecks pass. Firsttiny70optionalrecipecachepath left{rank}literal, SDKrejectedbeforekernels;71fixessubstitution, unchangedcandidate. No speedcredit; graph fusionacrossgroups remainsseparate.

- Entry71: synthetic chain all4 ranks exact with97 direct group replays/rank; native host0.188948–0.197321ms, archived ordinary60 reused untimed. Positive synthetic only; no model credit. See native-entry-probe-71/RESULT.json.

- Chain72 rejected: direct entry really executed252 times/rank and changing input bindings reduced to18 across4groups; real16layer median5.673781→6.050068ms(+0.376287ms), device5.533052→5.905502ms, essentially unchanged from69. All4ranks65tokens exact/nohotcompile; fullstate/lifecycle stopped by speedgate. No ledger/E2E/trace. All owned runs retired. Static RS→AG redundant-compute-wait candidate remains unmeasured; preserve completion-target validation. See continuation-chain-72/REPORT.md and RESULT.json.

- Ordered-index port73: actualTP2 threshold/emit reused with unchangedTP4 MME scorer/four8head BF16 groups; ordered candidate publication and consumption migrated together. TP2 cutoff tie rule differs from current genericTopK, explicitly unqualified for model quality. One-card score→twoMLA-consumer quickgate74 before realweights; no model credit/default enablement. See ordered-index-port-73/PLAN.json.

- Ordered-attention74:8TP2ordered selection cases exact(B1/B2/B6,FullR2/FullR1/block publication/Reindex,ties); unchangedTP4score→threshold/emit→publishMLA→reuseMLA host0.242526→0.212799ms(12.257%), nohotcompile/fallback. First current/new outputs exact;96-step feedback finals differ under changed tie policy, so no model-quality/ledger credit. Candidate-only actual16layer tokens next versus archived42; experimental defaultFalse. See RESULT.json and FOLLOWUP.json.

- Chain75 rejected:TP2ordered selection on realTP4layers2/8/14;median5.673781→7.009699ms, event5.533052→6.869544ms;4rank65tokens exact/nohotcompile. No fullstate/modelquality acceptance or ledger credit. Owned run retired exit0. One-card74 gain did not transfer; audit archived1.708ms sorting vsactualTP2 .195788ms and altered compiled boundaries before another candidate. --dump-plans was not requested, so no plans collected despite profile env.

- Selection76 offline root audit:exact1.708150655ms union partitioned into57localTop512(.411583exclusive),49mergeTop512(.785846),31blockTopK(.313509),8I32sort(.157051),overlap-once(.040162). All bitonic nodes oneTPC/ROI.75serializedrecipes reveal per-step unused dummy candidate DMA memset absent from74/TP2 cold-buffer contract; fixed cold placeholder, no performance claim. Next selective kernel-only rawtrace at3R2+1R1+4Reindex+block publication shapes; no wholedecode timer substitute.

- Selection77 planned: fourrank isolated selector kernel trace at actual3FullR2+1FullR1+4Reindex plus candidate-block publication,9threshold/9emit pairs pertoken/rank, alternating BF16 score banks and canonical ordered-ID oracle. No score/MME/model timing substitution; no model load. CPU barrier perstep confines cross-token overlap. Component trace can budget kernels but cannot qualify full-model0.2ms or changed tie quality.

- Selection77 analyzed:9threshold+9emit pertoken/rank; all canonicalIDs exact. Per-rank kernel union0.696–0.697ms =threshold~0.464+emit~0.232. R2threshold37.09us,R1/Reindex~70–71us each. Actual fourrank componentunion1.482825ms with8iterations retained; different alignment frommodel, no subtraction/ledger. Root now fixed16global scans on1TPC and scalar-emission scans; candidate78 exact16bit key cache aims at repeated global reads. No newmodel.

- Selection78:exact64KiB key cache compiled without LUT,33 contract cases passed. Selector-only fourrank trace threshold perrank0.464→0.140ms; total0.696→0.373ms; fourrankunion1.482825→0.858342ms. Fullmodel targetnotmet/noledger. Allownedruns retired; see RESULT.json/REPORT.md and preserved failed compilation/launcher/test attempts. Next target is scalar emission.

- Selection79:bitmap output passed33 exact ID/prefix/bitplane cases after16-bit shuffle packing bug caught beforetiming and fixed via32-bit reduction. Per-rank emitter0.232→0.109ms, threshold0.140→0.181ms; netselection0.373→0.290ms. Fourcardcomponentunion0.858→0.752ms. No fullmodel/ledgercredit; trace retired.80 targets measured cacheloop instruction stalls.

- Selection80:four independent cached counters and predicated add remove inner-loop NOPs.33exact contracts passed. Threshold perrank0.181→0.125ms; selector0.290→0.233ms; fourcardcomponentunion0.752→0.657ms. All8steps retained; no fullmodel/ledger.81 native build completed6.96s before capture started (CPU_BUILD_TIMING.json).

- Selection81:vector-buffered output passed33 exact contract cases. Emitter perrank0.109→0.084ms, total0.233→0.209ms; fourcardcomponentunion0.657→0.677ms (increase retained, asynchronous alignment). Not0.2ms achievement/noledger. Complete real16layer chain gate next to resolve dataflow transfer instead of more isolated tuning.

- Chain82:4rank65tokens exact/nohotcompile; host median7.774994ms vs saved42 5.673781ms, diagnostic only becauseCPU cores/competing load changed. Gatefailed/noledger/defaultFalse. Postgraphs confirmsame scoringMME/MLA shapes andplacement, replacedTopK/sort, no added compiledsegments. Eightstep83 trace required to resolve kernel transfer and waits; no unchanged speed repeat.

- Real trace83:real16layer chain,8captured steps after32warm,4rank41savedtokens exact/nohotcompile.24threshold+576emit lane events/rank;3R2owner selection union perrank0.0812–0.0820ms, actualfourrank0.177714ms. Covers3of8indexedowners only; NOTfull40layer0.2ms acceptance. No attentionbitonic events; threshold1TPC/emit24TPC. Meanthreshold15.642us, emit11.550us, emission lanesmin6.582/median7.827/max11.453us. Fourrankrank-union sum/fourunion≈1.836; serialthreshold andcrossrankoffsetremain. Larger-than-isolatedkerneldurationcausenotestablished. Allownedrunsretired; noledger. account.log premature-before-normalization failure retained; account02.log valid.

- Selection84: offline83 audit locates selector skew before its compiled recipe; no cosmetic barrier proposed. Candidate packs unchanged50-word threshold/prefix header into one contiguous store instead of50 scalar-partial stores. Reuses81 native parent;33 exact cases and candidate-only kernel gate planned, no ledger/fullmodel claim.

- Selection84 rejected/reverted:33contracts passed; threshold~0.125ms/selector~0.209ms unchanged. Fourcard0.684172ms CPUchanged, no claimedgain. Offline83: recipe-start skew150.672us, threshold skew151.369us, inside-recipe spread2.274us, upstream cause not host/network-attributed. No fullmodel/ledger. Next85 batches existing query+headweight AllGathers into one packet, actual score consumer gate required.

- 85 rejected: packed-query gather B1/B2 scores exact, median complete smallchain0.714115→0.778210ms, no ledger; failed initial codec shape retained.86 offline detected installed coalesced-gather one-entry metadata allocations for multiple tensors; no unsafe call/build. User interrupted optimization and explicitly requested latestfullpath E2E+decode trace.87 uses immutable83 Python +defaultTP4 ordered selector and81 native(P05 included), no runtime injection, rejectedpaths excluded; one formal16K→EOS and separatedecode32trace. Ordered path benefit/quality remain qualification questions, not claims.

- Serving87 CLOSED/notpromoted: oneformal16K→3891EOS15.599333ms vs14.622559B(+6.68%); threecandidateoutputs exact/semantics pass, baseline differs at token38. Full4rank31cycletrace15.945273ms: selection1.708151→0.544479, modelcompute−0.548473, no-visibledevicegap+0.959067, DMA/null+0.170855. Meanrankcompute−1.539547 but overlap deteriorates;110recipes/step unchanged. Allpaths audited, nohotcompile/bitonic, noledger gain/baselinepromotion. User now requests targeted recovery to13ms. See serving-fullaccel-87/REPORT.md.

- Index-gather-pair88: targeted110recipe/submission audit after87; paired query/gain AllGathers useonefrontend enqueue andHCCLgroup onstockcommstream, originalseparateoutputs, no pack/unpack. Built57.72s frompinnedstockbridge parent withonlynewheader/binding; unselectedinmodel. Candidate-only projection→communication→native-score gate reuses85separate timings; B1/B2 changing/reordered inputs before timing. Notledger/E2Egain.

- Pair88 launch rejectedbeforedevices: servingprofile raw_trace requiresprofilerregistration. No HPUrun/timing. User askedwhetherregressionrootcauseisknown; candidateheld. TwoAGsexist inbothB/A, so pairingisnotaprovenrootcausefix. Continueoffline87causalqueue/dependency audit first; build remainsunselected.

- Recipe-gap offline audit: serving87/RECIPE_GAP_CAUSAL_AUDIT.json; 110 ordinal comparison shows +0.514ms/rank no-visible-device at eight pre-selector boundaries, while selector recipe intervals save0.976ms start-to-start. Same two query/gain AllGathers lie between the unchanged producer and the selector consumer. These are locations, not proven NIC/host causes.
- Chain89/90/91/92 startup failures before HPU timing: precreated evidence dir, missing executable, nested distributed parent, and insufficient restricted CPU set, respectively; retained rejection records. Chain93 correct parent entry and archived42 card/CPU layout: real16layer host6.290869 vs5.667650ms (+0.623218), device6.188952 vs5.533380ms (+0.655572); four-rank tokens exact, no hot compile. Rejected for speed, no ledger/E2E credit. See continuation-chain-93/RESULT.json.

- Pair88 micro02: archived85 separate projection→twoAG→native score host0.721496ms vs paired0.505474ms, saving0.216022ms; device saving0.204062ms. All four B1/B2 changing/reordered scores exact,96 paired launches/rank, no hot compile. Score-consumer component positive but real16chain and source integration pending; no ledger/E2E gain yet. See index-gather-pair-88/micro02/RESULT.json.

- Chain94 queue relocated before HPU execution because the matching module group was acquired by another job. Chain95 paired-gather real16candidate on equivalent NUMA ordering [0,4,5,1] exact/nohotcompile but one synchronous42ms step and a second12-15ms step inflated mean to7.528460ms. Not a qualified speed decision. Archived original timings; one candidate-only exact-card replacement justified, no B repeat. See continuation-chain-95/RESULT.json.

- Chain96 queued original modules then relocated before hardware because other work owned them. Pair97 three-owners-per-step stress on free four cards:64 measured steps after32 warm,192 paired launches/rank, B1/B2 scores exact,nohotcompile, per-step device max4.58–5.90ms; the chain95 synchronized42ms stall did not reproduce in this narrower projection→pair→score chain. Full model interaction/ambient source unresolved; see index-gather-pair-97/micro01/RESULT.json.

- Chain98 same candidate and equivalent per-rank NUMA layout with cache reuse and saved plans: exact/nohotcompile but host8.943229ms/16layers, step median7.85–8.27ms and synchronized26.8ms spike; other4card model active. All4rank9indexplans replace2AG with1nativepair without changing compiled recipe or AR counts (PAIR_PLAN_DIFF.json). No qualified complete-chain gain; do not E2E/promote. See continuation-chain-98/RESULT.json.

- Pair100 stock-style ungrouped variant built from88 with one frontend task/two separate hcclAllGather calls, no GroupEnd; C1/B2/reorder exact,96launches/rank,nohotcompile. Score-consumer host0.539193ms/device0.526847ms vs saved separate85 host0.721496ms, but physical cards/load differ. Real16 gate pending; no ledger/E2E gain. See index-gather-pair-100/micro01/RESULT.json.

- Chain101 ungrouped pair real16 on same group/CPU as grouped98: exact/nohot, host6.273049ms/device6.166522ms, no >12ms step; grouped98 was8.943229ms with synchronized spikes under concurrent4card model. However original ordered chain93 6.290869ms on another physical group, so paired path has no credible complete-chain gain sufficient for promotion or full serving. GroupEnd causal role remains correlated, not proven. See continuation-chain-101/RESULT.json.

- Chain102 targeted full HCCL/Synapse/SCAL 8step real16 capture complete, raw bundles all4ranks, rank0 normalized. HostProfiler inflated rank0 device-recipe periods to43.509/55.629/39.031/50.856/44.014/41.763/43.008ms vs unprofiled real16~6.29ms; 15,161 raw host events but durations cannot be projected to regression cause. Preserve raw, stop other-rank parser to save disk/CPU; no gain or E2E. See continuation-chain-102/RESULT.json.

- Lightweight host probe103 first full-chain launch invalid: all four workers SIGSEGV during distributed initialization before weights/decode, probe itself suspected. No perf/correctness claim; retained raw failure and process retirement. Direct pinned libSynapse resolution build v2 requires small component preflight before any full model run. See continuation-chain-103/RESULT.json.

- Lightprobe103 v2 direct pinned Synapse symbols passed four-rank score-consumer preflight: B1/B2/reorder exact,96 paired HCCL launches/rank,nonempty API event files,exit0. Diagnostic meanhost0.59525ms; not a performance gain. First full-chain probe crashed before device and is retained separately. See lightweight-hccl-probe-103/micro01/RESULT.json.

- 2026-09-30 W0: latest user target supersedes 13ms/10ms goals: formal <=12.3ms only. Read-only model audit of17 flags; operator counts rebuilt with recipe execution boundaries. See W0_FASTPATH_AUDIT.md / W0_LAYER_RANK_OPERATORS.csv. 104 temporary host-probe chain is diagnostic, 105 invalid flag and106 FileNotFoundError: no performance credit; all model processes exited.

- W1 shared native indexer: shared-c1-W1-small-02 passed head-sum B1/B2 for8/16 local heads, TP2 ordered protocol and real MLA consumer. Small01 relative reference path invalid. Real16-01 failed device acquisition before weights/timing; four-device health check then passed. No formal/performance credit.

- W1 real16-02: shared computation retained, same modules[2,6,7,3]/mains[10,38,43,15], median6.549110ms vs chain42 5.673781ms; no hot compile, four-rank tokens exact, native entry0. No gain/E2E credit. Continue W2/W3; no isolated W1 repeat.
- W2/W3 small01 invalid before communication: bare VllmConfig selected unsupported runner V2; corrected to production EngineArgs. No timing.
- W2/W3 small02 invalid before communicator: per-rank cache template not expanded by new harness; fixed before imports. No timing.
- W2/W3 small03 failed before replay: profile still loaded standard SDK. ELF audit identified all missing native API symbols; chain69/72 isolated SDK has full required set. New bridge links that archived SDK; no global library replacement. No timing.
- W2/W3 small04: normal four-card peer payload, fixed-rank sum and padded weights gather passed; no timing. Cold baseline export rejected unused slice alias in probe output; diagnostic now uses a compiled mean.
- W2/W3 small05: complete native exchange/fixed-rank FP32 sum/MME feedback passed on four ranks at B1/B2, no hot preparation. Peer one phase vs native HCCL RS+AG; RESULT.json stores both timings. Component screen only, no model gain credit. Advance once to shared real16 stage replay.
- W2/W3 real16-01 interrupted before native capture/timing: fixed old component-harness warm_reference/prefix_groups assumptions. No speed result; not counted as negative direction. Also added visible-prefix graph identity and bounded prewarming to prevent old-extent reuse.
- W2/W3 real16-02: both actual8192 prefill chunks completed on all four cards. Cold stage capture rejected fixed12-head Engram input while TP4 supplies6; common validator parameterized24/tp_size. W0 correction recorded, no timing or model-gain credit.
- W2/W3 real16-03: all four shared plans prepared; first cold pass correctly flushed prefixes during later-group discovery and therefore had no whole-stage entry. Harness readiness incorrectly required full capture after one call; warm up to4 calls as production. No timing/no direction rejection.
- W2/W3 real16-04: four group plans now reach whole-stage native capture. Dependency analyzer rejects expanded gather because of legacy equal-size input/output. Parameterize outputCopies with world_size; exact byte count, disjoint and overwrite guards retained and tested. No scored16-layer result/no gain credit.
- W3 real16-05: complete41-point peer graph and segmented60-compute/33-point prefix captured on all ranks. Continuous feedback exposes shared host-open/execute-open epoch race before scored steps. Common epoch state now permits FIFO queued complete tokens while rejecting duplicate/misordered prefixes/suffixes; C++ contract passes. No timing/gain credit.
- W3 real16-06 shared complete stage: median5.260995ms vs5.673781ms chain42, all ranks5.260261..5.262352ms. One native72-compute/41-peer graph, no per-group native entry, no hot compile, all rank tokens agree. Model quality/formal pending; no E2E gain credited. Advance to one frozen16K/EOS serving request.
- W3 serving01 failed pre-KV C1024 workspace compile before API readiness/formal request. Shared decoded-state prefill region hits Invalid Node Params concat/4090 with decoded_swa[20480,512], offset12288. Reproduce small same-shape chain before reloading40 weights. Dumps moved to /dev/shm symlink to free root filesystem.

- SWA startup small02/03/04/05: standalone same shapes pass; all six serving prefill buckets pass with inactive profiler registration, native KV norm/RoPE producer, shared prepared pass and large position-bank view. Root is not established by these disproofs; formal request count remains0. Full startup retry will capture HABANA_NODE validation errors.

- SWA small06: free NUMA affinity admission failed before HPU/model/timing. No evidence on MME producer. Full serving02 remains the only model task; next diagnostic uses genuinely free22-27 during unscored startup.

- SWA small07: full MME combined Q/KV projection -> contiguous KV -> native norm/RoPE -> SWA region passes all six8192..256 buckets, without intermediate CPU synchronization. Diagnostic ran only during unscored owned startup and has exited; no model/performance credit. Full serving02 remains pending.

- W3 serving02: startup failure reproduced on all four ranks at C1024. HABANA_NODE validation: concat/4090 aggregation inputs disagree on dimension1. Complete standalone MME/KV/SWA diagnostics pass. Need actual SDK concat tensor geometries/axis and failing compiled FX. No request or performance credit. Own PGID3145239 retired.

- Concat geometry recorder preflight08 passes same MME/KV/SWA chain at1024/2048. Direct pinned SDK resolves without RTLD_NEXT; recorded cache concat has correct axis1 and raw shapes[512,127]+[512,1024]->[512,1151]. Startup diagnostic only; excluded from formal measurements.

- W3 serving03 diagnostic interrupted earlier at C2048 by SDK post-graph write failure5773 (ofstream.open failed; no errno). It did not reach C1024 concat. Inodes56%, disk584MiB free after failure; later filesystem write succeeded. Recorder preflight and full earlier concat geometries valid. Retired owned PGID3168636. Only this task12 PID post-graph directories moved to /dev/shm with original-path symlinks; no foreign artifacts deleted. Next startup redirects PID post-graphs before first compilation and samples descriptor limits.

- SDK initialization removes precreated PID dump symlinks. Detected before prefill; after initialization only owned worker PID directories atomically exchanged (renameat2 RENAME_EXCHANGE) with RAM-backed symlinks. Earlier original directories preserved, no path gap for writers. Common native-ready stat guard fixed in source for the later production launch.

- W3 serving04: after safe post-graph redirection, C1024 failure reproduced. All public SDK concat geometries valid (219/rank); SWA cache axis1/shapes correct. GC error5/concat/4090 belongs internal transformation. Need internal concat geometry and generating kernel, rather than modifying model dimensions. FD count132/max1048576; no descriptor-limit issue. Own PGID retired; zero formal requests.

- Internal concat recorder small10 preflight fails before SDK/device load: unresolved hl_logger TLS variable. No full model launched. Add explicit existing logger library dependency, retain failure.

- Internal observer preflights11/12 fail before tensor cases on early private logger/gcfg references. No full model launched. Reuse-state minimal check proceeds with the already-qualified public-only observer, independently of private observer linkage.

- Reuse-state small13 passes all8192..256 shapes and repeated1024 with shared module buffers/position bank. No isolated state-lifecycle repro. Internal observer v4 preflight14 passes, records exact matching native Tensor geometry without changing outputs. Explicit pinned SDK and logger/gcfg dependencies resolve before preload. Advance to full startup diagnostic05; no formal/performance credit.

- W3 serving05: ABI-matched internal observer captures exact invalid concat: raw[512,128]+[512,128]->[512,256], axis0 rather than1. Internal generated node5/concat4090; public model concat shapes/axis correct. Observer Eager extractor context absent: need static translator origin. Zero formal requests or latency credit; own PGID retired.

- Static trace metadata fix: shared prepared-plan JSON previously omitted generalized peer nodes. Cold dump now includes peer/scheduled-peer operations and graph input names for compute/exchange DAG reconstruction. Execution unchanged; source py_compile passes. Diagnostic06 retains earlier frozen source; later formal trace candidate includes these descriptors.

- Serving06 reproduces the internal axis/shape mismatch at C1024; node-translation hook has no caller context. No formal request/timing. Own PGID3255244 retired. Shared SWA tail now tries the already-qualified native activation roundtrip instead of expanding FP32 unpack; canonical packed-byte equality and all warmup shapes are checked in small16 before another full launch. This is a startup fix, not a gain.

- Small16/17: strict CPU signed-zero comparison rejects+0/-0 only; no nonzero mismatch. Small18 compares canonical numeric values and retains signed-zero differences: all7shapes/state reuse pass. Shared native codec boundary retained for full startup07. Production profile has no LD_PRELOAD observer. Formal card/CPU mapping [0,1,4,5]/[10,15,38,43], pending one unprofiled fixed16K request.

- Serving07: native tail-codec workaround did not fix C1024 compiler error. No request, no performance score. Change restored. Diagnostic08 freezes restored common source with v6 stack/precompile-JSON observer; all observer artifacts in RAM. Small19 validates the observer and captures a172-node workspace pregraph, without model weights. Compile-only SDK JSON loader is being built to avoid full reloads during root validation.

- Startup root proven: original eager concat axis0 reused for later axis1 shape; same stale ConcatFcdNode becomes invalid in AggregateFcdNode::extract. Tiny concat20 fails exactly at axis change; disabling PT_HPU_EAGER_SHAPE_AGNOSTIC_GRAPH in21 passes all7axis/shape cases. Shared default applies TP2/TP4 equally, retaining native StageReplay recipes. No MoE/mHC math change or performance gain claim. Recheck real16 gate07 before formal09.

- Real16-07 measured6.953042ms (device median6.806732500000001), regress versus gate5.673781 and previous W3positive5.260995. Same native72-compute/41-peer program topology and exact four-card feedback. Global eager shape-cache disable rejected and shared default restored. No full service request launched. Next fix is targeted cached concat axis rebinding, not broad cache disable.

- shared-stage-W3-serving-09: user-authorized temporary C8192-only startup; global SAG enabled; decode same positive16-06; formal16K EOS and independent decode trace pending. Concat compiler repair deferred before main.

- shared-stage-W3-serving-09 formal: 11.084135ms/token (90.219tokens/s), one unprofiled16K->2579 EOS; all fixed semantic checks pass; target<=12.3 passed. Temporary startup-only C8192 bypass; compiler fix still required before main. Decode trace pending.

- PR52 native isolation `pr52-rebased-old-ops-real16-02`: new Python/bridge with accepted TPC/MME library: median 5.790934 ms; all-new 7.166493, paired accepted 5.489099. Native regression 1.375559 ms; diagnostic only, no end-to-end request or gain-ledger entry. Device-before-residency gate worked; next inspect active native kernel/operator deltas.

- PR52 bounded native operator isolation: `pr52-native-micro-moe-{old,new}-02`: 0.07688 vs 0.07620 ms/layer; `pr52-native-micro-qscale-{old,new}-01`: 0.14742 vs 0.15103 ms/16 calls through QK consumer. Neither explains the real-chain regression; no gain credit. `pr52-native-micro-moe-old-01` failed before device acquisition because CPU lease required >=6 cores; expanded same main/helper selection in attempt 02. `pr52-native-regression-trace-01` failed parser before acquisition; added required --raw-profiler for trace-02.

- `pr52-native-regression-trace-02`: 8 real16 steps exported CPU scopes and raw device bundles on all four ranks. Post-capture trace-only generic TopK token oracle assertion failed; future trace tool now uses the same four-rank shared-index policy as timing tool. Preserve captures for diagnosis, quality pending; do not repeat capture merely for the final tool assertion.

- PR52 attribution correction: `real16-04` vs `old-ops-real16-02` differed in component startup ordering (device-before-lock corrected in eebac70b), so 1.3756 ms cannot be assigned causally to native library. Local MoE/Q scale checks did not reproduce it. Trace restores cached recipe metadata with binary debug-table identity checks; 7 complete 72-compute frames all ranks, 41 prefix TP dependencies, 0 causal violations. Producer skew localized to embedding/Engram entry; consumer P95 4.551 us. Preserve all startup-skew frames; no scaled performance claim. Next real16-05 uses all-new bundle and corrected startup, matching old-ops-02.

- `pr52-rebased-real16-05`: complete all-new native bundle with corrected device-before-residency startup passed real16 speed gate. Four rank host [4.87623828125, 4.87599059375, 4.8743924375, 4.87474121875], median 4.875365906 ms; device [4.762855, 4.761289, 4.7632485, 4.7612]. 72 compute/41 peer points, no hot compilation/per-group fallback, four ranks agree. Previous native-regression attribution retracted; no formal request yet and no gain-ledger entry. Advance to formal rebased serving integration checks.

- `pr52-rebased-decode-01`: complete prefill warmup 56/56; formal unprofiled 16K -> 2532 tokens -> natural EOS; 11.105891782 ms/token passes integration speed threshold. Token/text identity differs from accepted 2579-token output, first token difference at index 11. Final answer preserves all 14 fixed semantic facts/constraints (project, owner, date, batches, revised budget, difference, dependency order, temperature, rollback, archive and contact limitation); record exact=false and semantic=true separately. Controller initially stopped on stricter token-identity assertion; retain this error and perform remaining prefill/smoke gates, no repeated formal decode.

- PR52 rebased prefill-01 formal measured 4346.850477 tokens/s, below integration gate; failed, not a gain. Runtime audit proves C1 default inherited PREFILL_COMPUTE_TOKENS=8192, eliminating the qualified C16384 bucket despite scheduler capacity16384 and disabling decoder halo eligibility. Remove implicit cap, retain explicit diagnostic override and unchanged decode tile. CPU capacity/halo regression passes (1158 passed,58 device-conditional skips); earlier CPU retry used wrong V2/native-host test environment, preserve failed log as harness error. Next verify effective C16384 bucket before corrected formal prefill.
- TP2 smoke-01 exited before loading on local regression fixture upstream_lock_sha256 plan/manifest mismatch. Correct only test fixture metadata to current regenerated plan, preserving model revision, quantization and all tensor payloads; rebind sidecar manifest identity. Smoke-02 retains aggregate20min startup/request limit across attempts; assigned-card locks acquired. No TP2 performance trial.

- User scope change: cancel PR52 merge/integration gates, prefill qualification and TP2 smoke. Preserve14afd2dc; remain on rebased codex/tp4-decode-gap-1p5. Only TP4 C1 decode<=10ms, formal baseline11.105891782 and same-card real16baseline4.875365906. First direction denseFP8; shared replay must preserve C2-C6 DSpark entry.

- DenseFP8 naive single-operator screen (dense-fp8-single-02): QKV15.550->17.022us/layer; shared23.226->29.803us/layer. Negative local speed gate; no16layer/E2E or ledger credit. C1-C6K576 quant-tail exact, QKVrelativeL2 7.477e-5,shared8.213e-3. TP4N576 aligns18UE32blocks. Extended dense quant/scale supports64-element tail instead of truncation. Next diagnose extra quant/scale nodes and fold them into existing consumers; do not integrate the naive slow path.

- DenseFP8 single03 (native scale tensors in GEMM): QKV15.548->16.433us/layer; shared23.223->27.591us/layer. Extra activation roundtrip remains; negative operator gate, no real16/formal.
- DenseFP8 single04 (raw UE32 block-scaled GEMM): installed C1 runtime rejects first QKV compile with synStatus26 Genericfailure. Archive run log; no numerical/timing result. Retain existing offline UE32-aware sidecar scheme, do not infer universal hardware incompatibility.
- DenseFP8 single05 (reuse routed FFN FP8): shared relL2=.0749282, rejected numerical gate before timing. Routed producer truncates and uses BF16max/240, unlike shared activation RNE; direct reuse not adopted.
- DenseFP8 single06 (direct RNE plus fused shared SiLU): QKV15.590->13.805us/layer,shared23.190->21.111us/layer; four ABBA trials, four actual layer weights beyond48MB SRAM. Max relL2 QKV6.885e-5/shared1.109e-2; C1-C6 checked. Local40 projection estimate.154582ms, not an actual stage/E2E gain and no ledger entry. Other four cards resident ~89-90GB and idle at snapshots.
- DenseFP8 tail07: strengthened varying-column/row K576 tail values; explicit partial load and both FP32 conversion halves; C1-C6 encoded bytes/scales exact. Full-width dense quant hot loop restored. Original routed SiLU .text hash identical to accepted native library; new shared RNE uses same math with separate registered entry.
- DenseFP8 integration CPU: existing 1158passed/58conditional-device-skipped; selected FP8/catalog/reload contracts12passed/2skipped (includes2new). Native build passed after correcting SetOutputMetaFn lambda signature. All shared FP8 dimensions parameterized; BF16 attention uses common QKV mixin. Real16-01 queued for fixed0/1/4/5 due foreign controller211253 holding all-card locks while using2/3/6/7; no formal or gain credit.

- DenseFP8 real16-01 completed: rank host[6.069944906,6.146378938,6.145733781,6.133832219], median6.139783ms, device median6.0313755. Regress1.264417 versussame-card4.875365906 baseline; four ranks same tokens, no hot compilation, same72compute/41peer1066commands. Archived graph confirms actual FP8 path. Do not enter gain ledger or formal run. Host submissions alternate~4.5/7.5ms; cause unassigned pending candidate-only8-step trace. Other group~2.9GB/card idle at final timing snapshots, preserve full load timeline (not whole-host isolation).

- DenseFP8 real16-trace-01:8steps all-rank captures exported;7complete72-compute/41-peer frames, exact binary/graph cached symbol identity restored,0causal violations. Whole-frame producer skew P50 2.796us/P95 6.786us; entrypoint0 P50 2638.309us, Engramexternalpoint3 P50 2635.987us. Consumer P50 2.022us/P95 3.899us. Single-card TPC/MME means3.034/3.042/3.037/3.040ms; internal exposedhandoff union.653545ms. First attachment outlier31.588ms retained. Most added waits localized to front-of-token readiness; triggering cause not proven and no formal gain credit. Default v1 precision remains accepted path; move to common native-tail producer/consumer proof.

- Native-tail small01 failed in EngineArgs validation before weight loading/timing: harness default prefix caching requested without auxiliary engine ABI. Align config with the accepted profile (BF16, prefix disabled), retry small02. CPU native-FP32 peer bit/rank checks TP2/TP4 atC1/C2/C6 all6pass. No gain/16layer/formal credit.

- Native-tail small02 failed before weights/timing: harness used wrong initialize_model_parallel keyword. Correct to pipeline_model_parallel_size matching the archived fixture, retry small03. No speed or correctness conclusion.

- Native-tail small03 failed before weight loading: launcher rank placeholder in PT_HPU_RECIPE_CACHE_CONFIG was not expanded in the new tool. Copy the existing rank environment/HLS_MODULE_ID initialization before Habana import; retry small04. These three harness failures contain no timing and do not count as negative 16-layer directions.

- Native-tail small04 failed before numerical cases/timing: generic dense() assumes block-FP8 scale, while head is BF16 without head.scale. Read head via the same raw tensor path as the model BF16 loader; retry small05. No gain credit.

- Native-tail small05 proves actual four-card head/peer/next-embedding equality atC1/C2/C6, then rejects before timing because native replay count0. The identical Tail.forward code was compiled outside native context first; Dynamo reused that cached unprepared entry. Use the existing independent _compile_group entry for the native candidate, retry small06. No speed/16layer/formal gain credit.

- Native-tail small06 enters actual native graph capture after independent code entry, then correctly rejects full-stage mHC topology: standalone tail has2peer points, while the preserved mHC contract requires >=8/group. Disable TP_MHC_OVERLAP only in both arms of this no-mHC component profile; full16/service keep it enabled. Retry small07. No timing was collected, C1/C2/C6 direct checks remain exact.

- Native-tail small07 actual1graph/3compute/2peer replay, all4rankC1/C2/C6exact: rank-median device baseline.644649969ms vs candidate.647875999ms; negative/no16layer. Baseline had already combined head and embedding in one compiled entry, unlike service two boundaries; candidate additionally stages10KB hidden from clone. This screen does not qualify the proposed service boundary change. Small08 measures the missing split-boundary baseline once, aliases the fixed head input as actual internal stage tail, and checks queued changing-token consumers before timing. Keep07 for archive; do not subtract from08 newprotocol.

- Native-tail small08: same card/CPU, missing actual split-boundary baseline measured once; fixed internal hidden-buffer ownership; four-rank median device.971337219->.656010094ms, .315327125ms local saving. C1/C2/C6 tokens+embedding exact, queued changing-token consumers exact. Component only, not gain-ledger eligible. Common StageVariant final group now prepares head/argmax/next embedding in its native replay; public three-output interface retained, owned token getter used in fixture/runner. Next embedding cache consumption deferred until first16gate. CPU1167passed58device-skipped plus11entry/ownership checks; native-tail-real16-01 running with original v1 precision/MHCoverlap enabled.

- Native-tail real16-01 completed: host median5.760058125ms vs reference4.875365906; regression.884692219ms. Four ranks agree,64 cached native-tail sampler hits/rank,no hot compilation,1graph74compute43peer (reference72/41). Tail next embedding is not consumed by next step, so adds duplicate work. Reject speed gate; no formal/gain-ledger. Capture8candidate steps to locate actual critical path before fixing.

- Native-tail real16-trace-01:8candidate steps captured all4ranks, no speed/formal request. Offline normalization first used system Python withoutijson; failed before parsing. Retry onlyoffline normalization using projectvenv; hardware capture reused. Other modules2/3/6/7 free(768MiB,0util) at load snapshots and immediately after capture.

- Native-tail trace01 restored exact binary/graph metadata:7complete74compute/43peer frames,0causal violations. Entryembeddingpoint0 producer skewP50 3458.650us; Engramexternalpoint3 P503035.617us. Internal peer handoff union.670986ms; redundant tail embedding computation~.028ms andfinalize~.0027ms. Duplicate work cannot explain full.885ms stage regression. Next8step diagnostic adds raw-clock host-phase stamps+threadCPU/context-switch counters to distinguish late enqueue from device dependency delay; no candidate change/formal/timing credit.

- Native-tail-entry-trace02 failed before weights: parent fixture did not forward new diagnostic flag. Add parent parser/forwarding, retry03 with same candidate and trace-only protocol; no timing or gain result.

- Native-tail-entry-trace03:8host-phase captured allranks. Main-thread CPU active: input/hash .32-.36ms,prefix enqueue.19-.24ms,suffix enqueue.27-.30ms; no involuntary context switches in these phases. Dominant wall wait is previous-token readback6.27-7.01ms under profiler; main CPU~.04ms. This rules out main-thread descheduling in this capture, not actual unprofiled regression cause. Next8step capture enables existing TorchCPU ranges to observe LOWERING/EXEC native publication and producer registration, absent from scope-only capture. No candidate change/formal timing; retain profiler overhead.

- Native-tail-runtime-trace04 completed8steps, CPU profiler recorded239ranges/rank onmainthread; execute-thread native publish ranges absent(TLS profiling not inherited). Retain capture; do not launch another profiler to guess. Next native-tail-queue-diagnostic01 reuses already-built VerifyPhaseMarker to record actual EXEC submission and device completion in8unprofiled steps, no speed/formal result. No runtime rebuild or numerical implementation change. Source audit also locates native compute CCB host-wait at handleCommandsBarrier, but this is a hypothesis pending executed-queue evidence.

- User measurement workflow supersedes cross-process real16 gate: resident same-process separate native plans, ABABAB >=200steps/period, delta >2 baseline token-delivery IQR, other-card loading gate, 45min/candidate. resident-ab-01 loads immutable weights once; BF16/v1/no-tail baseline vs explicit denseFP8 and corrected head-only tail candidates. No profiler/formal request. RESIDENT_AB01_HARNESS_CORRECTION records diagnostic-only helper fix before import (remove unavailable result APIs), model source unchanged.

- FP4 instruction simulator01: installed TPCsim Gaudi2, real kernel ELF executes 98304 output bytes perarm with exact signs/zeros, changing N/K nibbles, group/channel offsets and invalid expert; reference2275 vs streaming2185 VLIWs (-3.96%), load549->645/store770->854. No hardware/latency gain claim. Mask-rewrite gives no static instruction reduction, drop without using cards. Retain simulator source and both binaries; no gain-ledger credit.

- FP4 simulator correction: production entry_points.cpp:1004 substitutes prefetch16; actual registered reference2169VLIWs vs streaming2185 (+0.74%); memory transactions also higher. Eight-vector2275 comparison is historical only, superseded for candidate decision. Reject offline without card run.

- Resident denseFP8 AB01: six200-step same-process periods, baseline median4.7736915/IQR.047426ms, candidate4.6041265/IQR.04227575ms, delta.169565 >2IQR.094852, A-period drift.0144515;72compute/41nativepeer+1externalhead eacharm, no hot compilation, four ranks agree.114.898s including cold candidate capture/loading checks. First valid real16 gain, formal pending. Corrected head-only tail cold capture rejected old odd-only topology42peer; no tail timing. Fix Python/C++ parity assumptions; capture still verifies actual producer/consumer coverage. Resident01 retires due this cold-capture error; retain dense result.

- Resident02 corrected tail:51.409s, A4.7665565/IQR.041473, B4.766242/IQR.0391545ms, delta.0003145 <2IQR.082946, inconclusive/no credit; A drift.0141355, no hot capture; A/B token sequences exact. Actual topology73compute42nativepeer vs72/41+1external, no duplicate embedding.

- Resident02 dense+head-onlytail:54.061s, A4.7632555/IQR.03813825 vs B4.585261/IQR.0381885ms; delta.1779945 >2IQR.0762765, A drift.0013565. This is the actual combination, not .169565+.315 or summing separate runs. Below.4ms formal gate; no formal request. FP8 changes feedback tokens, four ranks agree and per-arm stable; full five-sample quality still pending. Summary RESIDENT_AB_SUMMARY.json.

- Baseline transport audit: resident01/02 A included unqualified FP32 peer gather from tail prototype; formal14afd2dc has HCCL FP32 head. Dense A/B remains a comparable precision-only delta on that intermediate parent, but cannot establish formal-parent total. Head-onlytail .0003145 is incremental beyond already-native external head, not whole tail vs formal. Restore default FP32 HCCL; PreparedGreedyTail selects explicit common native_fp32_gather=True, no TP-specific branch. Resident03 rechecks actual formal-parent tail/combination via same-process ABABAB; no profiler/formal request until .4ms.

- Resident03 actual formal-parent head restored: tail A4.777644/IQR.07595 vs B4.763707/IQR.0357255ms, delta.013937<2IQR.1519; first A median5.193459 vs later4.767/4.766735, drift.426725 demonstrates baseline slowdown. No tail gain/trace/second formal. Combination initial delta.1870405<2IQR.2798005 amid recorded foreign restart/weight loading.

- Stable foreign-memory preflight (eleven snapshots spanning10s) followed by same resident combined plan:31.7316s, A4.766884/IQR.041197 vs B4.593715/IQR.04260925; delta.173169>2IQR.082394, A drift.00069. Whole combination below.4ms gate. Precise pureFP8 formal-parent follow-up:88.3533s including cold new plan and loading waits, A4.766596/IQR.03914175 vs B4.600908/IQR.04155375; delta.165688>2IQR.0782835, A drift.002206,72compute41nativepeer+1externalHCCL eacharm. Both four-rank-equal, no hot preparation, per-arm tokens stable; FP8 token sequence differs, formal5-sample pending. Formal parent summary indexed in RESIDENT_AB_FORMAL_PARENT_SUMMARY.json; previous intermediate-parent results retained, not added.

- Loading gate hardened in future resident harness:10s/11snapshots, foreign-memory range>=16MiB includes driver pool resets. Candidate watchdog writes partial status and retires owned Elastic ranks at45min. Current resident03 retains frozen3snapshot gate plus explicit10s preflight for steady retry; no frozen model mutation. CPU1172passed58conditional skips; targeted resident/FP32 wire11passed includes new pool-reset contract.

- kernel-switch-audit-01: serving09四卡原始activity复核；rank0分类活动3165.492/token、null11912.587/token；0–2us空隙2141.778次/2.230560ms。包含编译图前后依赖与自定义核词法调用行，复用图全局逐层归属仍需调用点join。专家TPC1.767958、MME1.093235、重叠0.795505ms。静态审计不入收益台账。
- resident-ab-control-04/1790762369618084146-b1dd1db9: SRAM slicing关闭候选，denseFP8同精度ABABAB；5.239588→5.228827ms，delta0.010761<2IQR0.092805，token精确，无热编译。未证明实际kernel数减少，不能将开关值视为路径证据；不入收益台账，不运行正式。

- Correction to unsliced04: VALIDITY_REVIEW.json marks compiler A/B invalid. Saved recipe symbols prove all FP8/shared recipes unbundled(2decode), sliced(6decode) recipes all BF16/shared; runtime compiler setting did not isolate recipe cache, FP8 A captured after B reused B. Raw results retained; zero gain; cannot reject slicing direction. Next pass changes actual Tensor/Scalar IR inputs, verify post-graphs.

- integer/static05: no timed periods; cold compile rejected real HPU get_attr by strict FakeTensor mode. Scoped fake conversion fixed; eager get_attr required to retain host binding and avoid compiler deepcopy of HPU buffers. small01 lacked TP op registration; small02 caught wrong get_attr partition placement before model reload. small03 verifies corrected full coordinate→gather consumer chain, queued behind another task's whole-host lock. No gain credit.

- integer-small03: corrected eager resident-I32 binding passes18 HPU coordinate→gather consumer cases C1/C2/C6 with changed/reordered positions; no timing/gain. Resident06 queued for real16 baseline-first FP8 versus FP8+I32, ABABAB200 and exact-token gate. Foreign group2/6/7/3 weights stable at89.75GB; all-host request lock released before acquisition.

- Integer CPU suite attempt01: 1176passed/58skipped/4failed:3Engram timing checks loaded obsolete in-tree host binary,1legacy zero-argument backend mock. Attempt02 uses pinned service native library and preserves zero-argument default backend invocation:1180passed/58skipped. Resident06 frozen default-false keyword is behaviorally identical; candidate true unchanged. No model hotpatch.

- Integer resident06/static-int32-01: ABABAB200, same source/profile/cards/CPUs, distinct physical changed recipes. Real16 pooled A4.650864/IQR.6298385 versus B4.4950175/IQR.07539275; delta.1558465 below2IQR1.259677, A drift.8752735ms. Four-rank feedback tokens exact, no hot compilation,72compute/41peer unchanged. FirstSWA recipe77→57 physicalTPC/MMEnodes(constant12→2,cast13→3), compressed initial35→33. Foreign2/6/7/3 loaded then continued native warmup; archives retain preflight memory observations. No gain ledger/formal credit.45min direction budget reached, put aside/default off. Next cache-isolated slicing gate; no extra trace.

- compiler-cache-isolation-small02: graph-name hash from startup yields independent same-FP8 expert recipes,17→7TPC/MMEnodes. Initial and3changing-input/route consumers exact. Small ABABAB200 A.397175ms/IQR.02675275 versusB.4277925ms, delta−.0306175 belowgain gate.0535055ms: no real16/formal. Postgraphs prove sliced weight tiles inSRAM, whole unsliced W13/W2 inDRAM; preserve pipeline rather than disabling all slicing. Small01 failed in unrelated eager roll staging before consumer report; CPU-staged mutable inputs fix small02.

- SWA packed small01/02: existing native packed MLA replaces generic full256-row unpack forSWA-only layers. Small02 includes real KV pack+ring index_copy producer before attention.18exact consumers(TP4H16/TP2H32,C1/C2/C6,p0/255/16384) pass; small ABABAB2001.320894→.443324ms, delta.877570>2IQR.1334255ms. This eager-device-event scope is screening only, not model/ledger gain. CPU1181passed58skipped; reference/candidate flags isolated. Resident07 normal production runtime queued for real16 FP8 versusFP8+packedSWA; no formal. Foreign2/6/7/3 currently loading weights~59-63GB; timing waits for stable observations.

- SWA packed resident07/swa-packed01: same normal runtime andP06 parent; A4.599719/IQR.04488575→B4.489538/IQR.040464, delta.110181>2IQR.0897715; A drift.013044ms. Four-rank and cross-arm feedback tokens exact, nohot72compute41peer. FirstAttention recipe77→49TPC/MMEnodes. P07 eligible, defaultoff, noformal; compatible P06+P07 real16.275869ms, user×2.5 estimate.6896725ms; P07 only2layers, layer-aware conservative estimate.524401ms. No I32 overlap double-count.

- Attention norm/quant prototype: common Q-norm producer extended5120 and publishes BF16 normalized row plus existing dense FP8 operand; physical C1 chain9→3 nodes. Native single consumer small10 with32captured frames and per-interval sync: .0237108125→.01655834375ms, delta.007152469>2IQR.000878656. C1/C2/C6 toy numerical12cases bit-exact; model epsilon1e-20 not toy1e-6. Real16 resident08 ABABAB200:4.489892→4.324850, delta.165042>2IQR.068147, drift.0048105. Eachrank and eacharm stable; parent vs candidate24/233 feedbacktokens differ fromstep134: raw statusnumerical_contract_failed. No gain ledger/noformal; defaultoff. Actual workerCPUs pinned10/15/38/43 beforecoldarms andafterverified, initialauto0/5/28/33 mapping superseded byaffinity records. Foreigngroup2/6/7/3 idle duringmeasurement. RAMreload variant passed simulator but failed actual FP8/MME consumers; reverted to cached version. Instruction/form_fp_num and earlier unsynchronized/host-submission measurements are diagnostic only. CPU1184passed58skipped; native database contract pass.45min candidate budget closed.

## Expert logical-tail prototype — stopped

CPU compaction and simulator guards pass; no HPU timing, no real16, no formal. Final tail branch increases instruction count (W13 9394→22058; W2 9214→19742); no ledger credit. Default source restored. See [decision](EXPERT_TAIL_DECISION.json), archived patch and new sources.

## I32 second resident measurement — stop

Same resident08, same 0/1/4/5 and CPU10/15/38/43, ABABAB200. A4.594794/IQR.057877→B4.4919225/IQR.054007; delta.1028715 below2IQR.115754. All feedback tokens exact, A drift.012216. No credit; two unqualified real16 attempts, stop. See [decision](I32_RESIDENT_SECOND_DECISION.json).

## Corrected norm numerical reference; shared constants screened out

P08 hardware epsilon1e-20 C1/C2/C6: twelve cases bit-exact normalized/FP8/scale/MME against existing native norm and dense quant. CPU simulator24 cases exact. Raw real16 result retains generic-parent24/233 feedback changes; user accepts TP2 native numerical behavior, formal5semantic samples pending. Qualified pending delta.165042ms; no repeat timing, no formal request. Shared fixed metadata7→5 nodes passes9exact consumers, .022150516→.022288562ms no gain; stopped before real16 and source restored. Archived [decision](SHARED_STATIC_DECISION.json).

Kernel layer audit02 source-layer ownership labels superseded by03: cached names can retain first layer. [Audit03](kernel-layer-audit-03/SUMMARY.json) retains runtime compute slots and next reduction context, plus ambiguous source scopes; no source-layer exactness inferred from cached names.

## Engram gate/update fusion — component only, deferred

Small04 actual layer1 q/k/control weights, Engram→native control/RRMS+collapse.13→4 physical nodes, ABABAB200x32perinterval-sync .025506672→.016744953ms, delta.008761719>2IQR.001346156.12cases C1 exact, C2/C6 relative error pass. Two-layer isolated estimate .017523ms below recent real16 noise threshold. No real16/noledger/noformal. Default source restored, archived [decision](ENGRAM_UPDATE_DECISION.json), patch and new native sources. Earlier setup/build/recorder/layout errors retained, not performance evidence.

## Shared expert prequant reuse — rejected numerically

Toy complete FFN norm→shared expert consumer9cases: output L2 difference4.0–6.9%, exceeds declared5% bound. No timings, no16layer, noledger. Prototype archived and restored. Preserve both scale arithmetic in future fusion; [decision](SHARED_PREQUANT_DECISION.json).

## 2026-10-01 norm＋BF16 handoff＋Engram组合正式触发

resident09稳态A/B：4.4937795→4.2466885 ms，差值0.247091>2IQR0.169684；首次基线漂移未过门槛记录保留。替换纯P08，累计16层0.522960、机械整模预估1.307400 ms。正式/5样本待验收，启动kernel-fusion-serving-01，完整预热、容量1048576/8192/32。见NORM_HANDOFF_QUALIFICATION.json；CPU1186/58。

## kernel-fusion-serving-01：正式11.021114593 ms，未达10ms

完整56预热，16K→2784 EOS，5样本语义通过。组合预测1.3074、实测差值0.084777，不计预测收益。已测组合移出未端到端累计，保留LEDGER_PRE_FORMAL.md。64步trace已采，无热编译；首次collector缺少raw manifest误入CPU-only分支，未改写数据，补原始SDK/PID/clock硬链接manifest后analysis-02解析，不补采trace。

## 2026-10-01 kernel-first trace and screening

Four-rank serving02: 63 full cycles, 180 compute recipes and 100 native peer points; ready skew p50 3.414 us / P95 5.777 us. Rank0 template 3064 physical compute nodes/token (76.6/layer), below historical3384 but not40/layer. Source/call/dependency audit per layer/rank is kernel-switch-audit-04; cached scope ambiguity retained. Durable report: [REPORT](kernel-fusion-serving-02/REPORT.md). Formal11.021 remains unqualified for10ms; no new formal requests.

Mirror gather fusion: local nodes7→5, exact12scoring cases,14.894→28.068us; rejected before16. Prototype archived/restored, zero gain credit: [decision](index-mirror-gather-native-01/DECISION.json).

Compiler granularity: 32MiB physical17→17, no node reduction; replay observer setup error retained. Corrected native unsliced screen17→7,55.102→88.704us (>2IQR.378us regression), exact changing consumers. Stop before16: [decision](COMPILER_GRANULARITY_DECISION_20261001.json).

InverseRoPE→wo_a→wo_b: local9→8,28.965→27.121us,12exact C1/C2/C6 cases. Two real16 ABABAB: first baseline drift .822ms; second4.236337→4.220668ms,delta .015669<2IQR .114214. No qualified gain; prototype restored. Other cardsidle768MiB, worker CPU affinity restored to10/15/38/43 before any timing with thread-mask record. [decision](rope-woa-small-01/DECISION.json). Resident10 remains loaded for future independent candidates; no queued jobs.

## MHC SDK linear load — stop after two unqualified real16 runs

Control→gates 2→2 nodes, 15.957→13.794us; twelve exact hardware/simulator contracts. Steady real16 4.239385→4.192107ms, delta .047278 below2IQR .091217. First baseline drift retained. No ledger/formal credit; production prototype archived/restored. See [decision](mhc-linear-small-01/DECISION.json). Resident11 retired for quiet-host serving retest.

## Quiet-host serving retest — preparation

User replaces load1 gate with CPU PSI some avg10<1%, no foreign user task competition on10/15/38/43, and stable foreign-card weight allocation. Exact serving01 frozen sources/native library reused for serving03; same capacity/full warmup/sample. New external one-second monitor records CPU counters, per-thread switches/scheduler wait, PSI and desktop process CPU. Forecast multiplier now1.5, trigger real16 .8ms. No formal request yet.

## Host retest and isolation

Unchanged serving03 formal10.936414ms,2784EOS exact. Tail .835612ms,PSI during decode peaked2.67%; no quiet baseline. Host64-stepdiagnostic archived with explicit truncated status; no worker period>15ms in that ROI. Controller/worker helper overlap and desktop SMT allowance corrected. Short512 diagnosticengine9.993749ms; first4-token coalescence means onlypartialITL (.245706 excess),no credit. Corrected-isolation formal10.030447ms,2784EOS exact;2778 observable intervals yieldconditionalexcess .314448ms; stillnot10ms orjittergate. No new baseline.

ExternalGC probe unavailable: kernelBPFattachEINVAL, calibration didnotexecute, no GCabsence/causality inferred; registered own tracepoints cleaned. Source comparison foundHPUV4.1missesGPUworkerpost-warmupfreeze_gc_heap. Reuse shared utility, GC remainsenabled. CPU934passed112skipped. Serving04started for candidatevalidation; no microledgercredit.

### 2026-10-01 host复测与分页边界

- 原源码正式10.936414ms；隔离主核/SMT及控制线程后正式10.030447ms，均2784 EOS且token精确一致。后一轮可辨ITL正向超额0.314448ms，未建立安静新基线。
- GCfreeze正式10.734059ms，可辨ITL正向超额0.721990ms，撤回，无收益。
- page-boundary-trace-01为512token长度限制诊断，保留failed EOS门禁，不属于正式性能验收。P16640跳过early prefix，target提交5.000–5.138ms，下一步周期15.438–15.769ms；重复位置与128token分页一致。
- 共用BatchStageState增量append及prefix前安全发布候选待测；微基准收益累计仍0。

### 2026-10-01 paged append 生产路径核查

- serving05正式10.552049ms、2784自然EOS、token精确一致；2779个可辨ITL中位10.168295、P90 11.961750、P99 16.599511ms，>15占1.655272%、正向超额0.629729ms。
- Audit-only profile无generation，不是trace性能样本：四rank prefix_state_transitions=24、page_appends=0。BATCH_DECODE未开启，BatchStageState候选未命中，已归档并恢复batch源码。
- 生产PagedStageState每次页表增长原先会全设备同步并失效索引镜像；新候选保留旧映射/镜像、只写新增项，V2同owner可提前启动prefix。
- paged-append-component-02单模块0、CPU10，ABABAB×200，页表数值及读回顺序通过；未包含mirror重建和模型消费者，不计完整链/整模收益。CPU945/112。
- serving06沿用原compute/native/profile，桌面移离worker辅助物理核，单独记录OS环境；短分页门禁通过才进入正式。待验收收益仍0。

### 2026-10-01 formal goal achieved

- serving06/speed-02: formal9.933990731ms/token,2784naturalEOS,fullwarmup,5/5fixedsemantic samples passed. All outputs exact against accepted references.
- Observable ITL2780intervals: median9.911872,P90 10.314221,P99 12.199491ms,>15ms0.071942%,positiveexcess0.176862ms/token. Three initial intervals unobservable. PreflightPSI<1%;decodepeak2.06%recorded.
- Serving07 repeats earlier3qualityshapes before final dependency sample; same source snapshot and retained engine/runtime identity; fifth sample636tokens exact. Initial empty disk-full attempt preserved, original process.json truncated: recovery record explicitly states loss.
- Shared page append and auxiliary-core isolation qualify this final combination; no separate complete-chain credit for metadata-only microbenchmark. Pending gain0. PR52 remainsdraft.

- 2026-10-01 ordinary sampling correction: shared validation wrongly rejected nonzero temperature when DSpark was off. Candidate `42899946`, PR53; CPU 976/114; HPU inverse-CDF outputs agree with CPU under changing controls. `sampling-validation.json`; full independent serving qualification pending. Component timings are overhead observations, not gains.

### 2026-10-03 official combined sampling regression

- `official-512k-serving-02`: formal prefill9038.992651tps, decode18.535273ms;2128naturalEOS,14/14fact checks. No sampler fallback on shutdown audits.
- `official-512k-serving-03`: formal prefill8957.475039tps, decode17.674718ms;2128naturalEOS,14/14checks. Async resolver does not repair sustained regression. Both failed; PR56 remains draft and gateway stopped.
- `official-512k-path-diagnostic-02`: separate profiled request,63completecycles. Rank0 TPC/MMEunion7.354335ms, profiledcycle20.497663ms; worker_commit median17.7805ms. Native cached recipes lack exact graph metadata; functional kernel attribution unresolved. Prefill event enclosure now includes sample completion, checkpoint and all-rank acknowledgment. Checkpoint44–47ms.
- `nucleus-real16-controls-01/control/controls-order`: production pinned async upload, ABABAB200. Amedian6.837969ms,IQR1.078580ms; B6.678011ms. Difference0.159958ms <2IQR2.157161ms; rejected, production reordering withdrawn. Next minimal diagnostic separates control upload and native prefix/suffix entry. No ledger gain.

- `nucleus-real16-controls-phase-01`: productioncontrols vs prior synchronousfixture ABABAB200; median6.900444→6.426778ms below3.076552ms threshold. Upload0.45ms, prefixentry0.37ms. No effective gain and no full-serving repetition. Next gap: production model post-prefix view/signature access absent from fixture.
- `prefix-owner-handoff-01`: owner-only complete-ring/SWAconsumer checks pass but median5.377884ms difference below7.209410ms threshold. No independent gain. Concurrent weight loading is recorded in other fixture launch; not a quiet standalone baseline.
- `prefix-owner-codec-02`: same40layer/46ring consumer plus compiled existing SWA reader;54.309955→8.748552ms,45.561403ms >8.100912ms threshold. Three changed-state trajectories/checkpoint/liveB1/B2 consistency pass. Per-request handoff, not decode gain; formal pending.
- `nucleus-real16-model-entry-01`: two lock-acquisition failures before device init preserved; previous owned fixture was still retiring. Third launch acquired leases and commenced. Adds actual model input views/signatures around native enqueue; no performance conclusion yet.

- `nucleus-real16-model-entry-01`: actual model metadata paths ABABAB200. A6.899591ms versus B6.968609ms; difference-0.069019ms, threshold1.635796ms. All4rank/crossarmfeedback exact. Rejected; normal model remains unchanged. Both entry guesses fail the micro gate. Rejected source retained; no gain ledger entry.
- Retired diagnostic graph files losslessly gzip-compressed with original SHA256 in `official-512k-path-diagnostic-02/graphs/compression-manifest.json`; no original graph content lost.
- Prefix codec component promoted to branch `efbbd824`; next formal combines that qualified transition with lightweight existing worker-audit aggregates for upload waits and native prefix submission. No profiler or trace is enabled. Source/runtime defaults, capacities, sampling distribution and replay ABI unchanged.

- `official-512k-serving-04`:6820431e,completewarmup and one official-seed16K→2128EOS request;8860.171914tps/17.584649ms,14/14facts. Both gates failed. Audit points to parameter update4.93–5.07ms, not prefixenqueue. Settings/runtime have no workspace/evidence dependency;599 installedfiles match HEAD. No raw capture.
- `sampling-control-numpy-01`: pinned-host scalar preparation replaced by a NumPy view; old/new full-vocabulary samplerconsumer outputs agree over600draws perarm. ABABAB200 wallmedian1.286731→1.131647ms,0.155084ms >0.145101ms threshold, narrowly. Candidate isolatedfill0.058ms/copy0.123ms/event0.055ms. This does not explain full-stage5ms and has not passed the16layer gate. Next `nucleus-real16-numpy-01` uses production control updates and separate reference/candidate classes on0/1/4/5,CPU10/15/38/43.

- `nucleus-real16-numpy-01`: sameformalmodules0/1/4/5,CPU10/15/38/43. ABABAB200 A6.245492ms B6.144109ms;0.101384ms <0.964327ms threshold. Rejected and NumPy production change withdrawn; reference/candidate source retained. Upload in16layerfixture~0.43ms versusformal~5ms. Existingfixture cannot explain the measured full-stage dependency.
- `nucleus-real40-controls-diagnostic-01`: diagnostic component broadened to40layers/512K to reproduce the observed full-stage control update; no formal request and no speed qualification. Original Torch parameter writes retained, adding fill/copy/event subdivisions. Lock acquisition failed beforedeviceinit whilepreviousownedfixture retired; laterlaunch acquiredleases andbegan.

- `nucleus-real40-controls-diagnostic-01`: failed before timing, manual fixture lacked the shared decoderhalo method and start/stop fields. Static dependencies repaired for02; failure preserved.
- `nucleus-real40-controls-diagnostic-02`:40layers/512K ABABAB200 original controls versus timer-only code:14.217461→14.201767ms (no gain). Controlupdate~0.49ms, not representative of the observed serving wait. No performance qualification; stopped.
- `sampling-control-initialization-02`: service host.to initialization and component GPUliteral initialization give essentially identical producer/sampler costs and exact600draws. Rejected as explanation.
- Next minimal serving diagnostic retains original parameter writes/transfer/event protocol and records decode-only wall/threadCPU subdivisions forfill/copy/event. Warmup and first-prefill counters separated. No further unmodified formalEOS request authorized or planned; one80token diagnostic after startup is needed because complete component cannot reproduce the serving-only wait.

- `serving-control-subphases-01`: INVALID. Instrumentation phase loop overwrote the sampling cache key;236decodeuploads/80tokens. No production-path timing inference. `3ee391fb` uses phase_key and adds repeated nonzero-ordinal cache-hit regression.
- `serving-control-subphases-02`: replacement short80token diagnostic on fixed maintained source3ee391fb. Original controls protocol retained; four-rank ready/shutdown deltas separate startup from hot decode. Queued after full warmup; not formal EOS qualification.

- `serving-control-subphases-02`: corrected protocol,79hotupdates/79commits perrank after ready subtraction. H2Dcopy4.090–4.479mswall/0.061–0.066msCPU, fill0.191–0.197ms. Pinned-source eagercopy unconditionally joins pipeline in installed Bridge source. No EOS qualification.
- `sampling-control-pageable-01`:600fixedu exact perarm;1.296448→1.215031ms,0.081417<0.427450ms threshold. No gain; queue dependencies absent. Nextreal16 candidate changes onlyhostmemorytype, normal controls/event protocol unchanged.

- `nucleus-real16-pageable-01`:6.302911→6.233618ms,0.069293<2.980360ms threshold. No gain and no formalEOS. Originalcontrols0.47ms here; serviceH2D4.09–4.48ms remainsunrepresented. Next inspect existing servingCPU/native trace for queued dependency, not more scalar-fill variants.

- `serving-control-subphases-03`: candidate-only80token actual-service diagnostic:a9c733c5,pageable owned-snapshot controls;14CPUtests+600fixedu exact. Real16noisygatefailed;notformalqualification. Distinguish blockedcopy elimination from transferredprefixwait, usingpairedready/shutdowncounters against02. No newformalEOS.

- `sampling-control-owned-01`: BridgeHPUEvent.record alsojoins eagerthreads; pageableinputs alreadyowned-snapshottedbeforeenqueue. B1sourcecandidate skipsfrontendEventforpageableonly;pinnedfallbackunchanged.1.221963→1.072465ms,0.149498<0.465413noisegate. No gaincredit. B2queued64steps,reorder,controlframes/tokens exact. Actualservice03event counterspending.

- `serving-control-subphases-03`:79hotupdates perrank;copy0.032–0.046ms butfrontendEvent4.314–4.447ms. Total80tokenrequest3.295svs02 3.290s. Waitrelocationconfirmed,notgain. Nextreal16owned-snapshotcandidate removes unnecessaryfrontendEvent;sampledistribution/GPUpointersunchanged.

- User stopped current sampling-fence direction. `nucleus-real16-owned-01` interrupted;ownprocessgroupSIGTERM,no gaincredit. Release rollback must keep efbbd824/prefill/cache fixes and official sampler warmup, restore generic sampling. NewdeviceRNG/once-per-requestcontrols direction onlyafterrelease,onaseparatePR.

- `release-rollback-01`: source5a787dbe, requested sampling reverts complete; efbbd824/prefill/cache fixes and official sampler startup warmup preserved. CPU977/112. One full-warmup officialT1/P.95/seed42 uncached16K→2128naturalEOS request: prefill8914.621782tps/decode10.599712ms,14/14facts. Decode recovered; >=9000prefill release gate FAILED by0.95%. PR56 draft/main unchanged/public gateway off; no unchanged repeat. See OUTCOME.md.

- Publication explicitly authorized despite the prefills gate miss: PR56 merged main ea017d60, tagdeepseek-v41-tp4-official-20261003. Installed598tracked plugin/toolfiles exactly matchmain/runtime unchangedfromformal; publichealth/models200/authmissing401/chatoutput normal. Gatewayactive; existingkeyunchanged. Independent release proofs and accepted performance stored in installed validation directory and release-rollback-01.

- `published-prefill-audit-01`: user1K/21KTTFT10.841/19.066s confirmed server-side prefill slowdown; newly completed ~42K prompt prefill128.173511s, queue53.875us. Source admission shows checkpoint small-block/search prewarm gaps and >32K→524288search fallback; eager fallback with cache disabled, live compile/lowering threads active. No new generation/profiler/runtime mutation during user benchmark.16K8914.6tps qualification does not cover arbitrary lengths. See OUTCOME.md/static-geometry.json/metrics.prom. No gain.

- `visible-index-chain-01`: packed capacity524288 vs65536, Full→Reindex→packed-rowconsumer, all4boundary outputs exact; no effective gain. TPC source independently limits scored rows by positions, so prior interpretation of full-capacity dotproduct work is superseded. `visible-index-mme-chain-01`: same-process ABABAB200 at41983position; ratio1 saving4.5770465ms vs0.0779845threshold, ratio2 saving2.967156ms vs0.0657045threshold; all32K/42K/62K/64K selections and row reads exact. Component only, not formal model credit. `visible-index-real16-01` failed argparse before weight/device initialization: copied launcher contained revoked --sampling-phase-timings.02 removes it, parameters41984context/512Kcapacity and modules2/3/6/7 with64/69/92/97CPUs; awaits real16gate.

- `visible-index-real16-02`: actual41984prefill/native decode warmup succeeded, but module entry imported resident helper as a bare script module, raising ModuleNotFoundError before A/B. Evidence retained;03 fixes both module/script import modes. No timing/gain credit.

- `visible-index-real16-03`: observed6.505572→5.055611ms, delta1.449961>0.118629threshold and allfeedback exact, but INVALID for42K qualification: chain still used literal16384+index despite42Kprefill/reset. Rawresult retained with QUALIFICATION_CORRECTION.json. No gainledger credit, servingcontroller stopped before formal request. Fix actual loopposition and add context-contract regression before real16-04; candidate backend continues its existing warmup.

- `visible-index-real16-04`: corrected actual chain positions use context41984+index; six CPU context regressions passed. Every arm checks first device position and records it per rank;03 remains invalid. Signed source9056543d adds durable measurement checks. Same other-group cards/CPU; resident load in progress, ABABAB200 queued. No gain credit yet.

- `visible-index-real16-04`: interrupted during weight load by logged SIGTERM at07:05:36, source unresolved; no operator/compiler exception and no timing. All fixture processes subsequently retired. Preserved OUTCOME.json, no gain credit. User asked whether another session is cleaning this group.

- `visible-index-real16-05`: user confirms continued use of2/3/6/7 after04SIGTERM; temporaryforeign module2lease retired. Fourlocks reacquired, same corrected9056543dcontext; queuedABABAB200. `visible-prefix-serving-02` formal gate points ONLY to05 with fourdevicepositions41984; length/cache diagnostics queued after formal semantics. Public gateway remains off.

- `visible-index-real16-05`: VALID actual41984positions onfourranks;10.0259465→5.0995845ms,4.926362>0.1233685noisegate, all200tokens exact acrosssixperiods/fourranks. Native41communicationpoints/72compute_calls eacharm; nohotcompile. Ownloaderpause215sauto-thawed, finalperiodwaitedfornootherloading;04/03notcounted. Residentretired. `visible-prefix-serving-02` now uses独立unit1cat-tp4-visible-prefix-validation after conventionalunitexternalclientSIGKILL; normalinstallation9056543dparity/512K/C16384/cache/officialsampling unchanged. Installed-engineCPU23/0.

- `visible-prefix-serving-02`: third startup interruption, independentunitclientSTOP07:38:17 duringlong-searchprefillwarmup; first48stepscompleted, nooperatorerror/noscoredrequest. Formal/diagnosticcontrollers cancelled BEFORE sending requests. SavedSTARTUP_INTERRUPTION_03.json+journal; publicgatewayOFF, PR57draft. User coordination requested after three externallyinterrupted startups; no furthermodelreload untilclear. Validreal16-05 evidence is retained.

- `visible-prefix-serving-03`: user resumed servingqualification afterexternalstop. All8cardsidle/PSI0; project/tmp/evidence leases0/1/4/5 heldbeforeindependentnormalservice startup. Source78ebead9runtime+tools byte-equalinstalled; docs-only advancefrom9056543d. ReusesVALIDreal16-05 gate, oneofficial16KEOSthen1Kcold/hit21K/42K/62Kdiagnostics. Earlierinterruptedcontrollersremaincancelled; no baselinerepeat.

- `visible-prefix-serving-03`: no formalrequest; LLVM ENOSPC duringpre-KVC256, followedbypthread_join/segfault. Ownedservice+pendingcontrollersretired. Losslesslyarchivedownhistoricalgraphsondiskwithtarbytecompare/SHA256 (GRAPH_ARCHIVE.json+storage-archive.json); tracesnotdeleted. `visible-prefix-serving-04`: source297d1995 addsnormal-launcher recreationofconfiguredcompilerTMPDIR; portableCPU3passed. MachineprofileTMPDIR usesownRAMscratch; math/sampling/cache/defaultflagsunchanged. Rootheadroomrestored, allsource+toolsparitychecked; reusevalidreal16-05, oneformalEOS+length/cache checksqueued. Archiveworkmustfinishbeforeformalmeasurement.

- `visible-prefix-serving-04`: complete startup; one official-seed 16K→2128 EOS formal, prefill8478.213991tps/decode10.522140ms,14/14facts pass; prior release prefill8914.621782tps/decode10.599712ms. No repeat formal. 1K official cache cold0/hit896; colddecode37.396ms vs hit10.278ms. Same-seed sequences diverge at token8, both reasoning coherent; exact-token assertion exceeds user semantic contract, deterministic state diagnostic pending. Missing long cases now queued behind measured other-group weight growth; per-request arrivals and load snapshots retained. Public gateway off, PR57 draft.

- `visible-prefix-serving-04` completed missing length diagnostics: 21504prefill4.080056s/decode10.861744ms;41983prefill52.733681s/decode12.540521ms;62463prefill75.946818s/decode12.659461ms. Official T1/.95/42,80tokens lengthlimited, not EOS qualification; competing loader reached stability before request but CPU PSI rose during timing. Byte reader clientTTFT invalid (large initial prompt IDs processed byte-by-byte); engine histograms retained. Separate4096-byte greedy state probe: first72tokens exact then wording differs; cold/hit coherent, cached896, TTFT.938/.384s. Do not promote greedy numbers or claim exact state equality. Source static finds 32K index admission/shared-key cap and finite MLA search guard. Candidate bounded64K/short handoff warmup now CPU/component phase; no new formal.

- `prefill64k-index-chain-01`: failed before measurement, inherited worker CPU exclusion; explicit taskset restored reserved CPU admission for02. No HPU timing or gain.02 adds competing-load gate and four-rank synthetic Q exchange→Full→Reindex→actual MLA consumer,ABABAB200.

- `prefill64k-index-chain-02`: no correctness or timings, fixture MLA scale float rejected by native Tensor schema;03 uses production FP32 scale tensor. No source-candidate regression inferred.

- `prefill64k-index-chain-03`: native capability rejection before correctness/timing: candidate-gather Meta and host glue both bounded F32 source columns to32768. CPU-only reuse tests could not expose it; TPC addressing uses runtime int32 source_rows. `prefill64k-native-build-01`: two bound checks extended to65536, all other native source hashes match release; rebuilt with60availableCPUs in RAM scratch, kernel database CTest passed. Initial CPU profile conflicts preserved in /tmp logs; task-local generated symlinks now select one matching candidate native package across legacy V4/V4.1 collection, no shared artifact overwritten. Full CPU suite underway.

- `prefill64k-native-build-01`: full CPU1013passed/111skipped after fixing duplicate HostRows registration via canonical extension import; retained binary/ABI validation, no interface/signature change. Two failed profile collections and two pybind test failures preserved in /tmp logs. CTest1/1. `prefill64k-index-chain-04`: first retry with matched expanded64K native Meta+host package, complete consumer gate; no full service restart yet.

- `prefill64k-index-chain-04` superseded before any completed period: used legacy sparse BF16 MLA instead of serving FlashInfer and smaller packed capacity. Preserve48exact selection/legacy-output checks, no production-chain credit.05 corrects consumer to compiled_flash_prefill_mla with640 columns/causal SWA and actual512K packed cache/page-table geometry. Untimed20-step progress writes added. Candidate/native package unchanged; no service repetition.

- `prefill64k-index-chain-05`:48exact checks with actual serving consumer, but timing NOT serving-comparable:46–50lazy runtime threads each inherited only mainCPU10/15/38/43. Raw A200~1.67s and partial B~.286s preserved, not gain/ledger evidence.06 rebinds helpers after correctness warmup and after collective barrier, archives every TID affinity. Competing group log confirms warmup, not weight load; temporary workspace growth alone no longer blocks timing whenPSI<1. MLA32K/64Kadmission CPU101passed, separate HPU validation pending.

- `prefill64k-index-chain-06`: completed ABABAB200 actual Full→Reindex→FlashInfer consumer; four-rank max wall 1356.455→110.669ms; 48 exact checks; component-qualified, long-prefill serving pending. Other-group startup recorded.

- `prefill64k-mla-chain-01`: completed 4Kqueries/64Ksearch ABABAB200, max-rank wall7.564→6.010ms;12exactoutputs. Gate passed. Newly admitted16Kshape is pending.

- `prefill64k-mla-chain-02`:16Kqueries/64Ksearch actual MLA exchange ABABAB200;23.228→16.706ms,12exactchecks, gate1.228ms passed. Combined source serving pending.

- `visible-prefix-serving-05`:combined installedb7553683, fullwarmup started10:43; componentgates passed, official16KEOS and arbitrarylength/cache diagnostic queued. GatewayOFF, PR57draft.

- `visible-prefix-serving-05` COMPLETE:normalinstalledb755, fullwarmup, official8851.110tps/10.565244ms/2128EOS/14facts/exactprior04; goals9.5k/10msNOTpassed.1KcoldTTFT1.765s/hit1.148s;42K9.104sprefill/12.540msdecode;62K11.803s/12.738ms. Public restored, PR57draft, mainunchanged. Remainingmirror/directscore32Kbound is static hypothesis only.

- `api-recovery-enospc-01`:supervisor failed11:47:20 at periodic process.json ENOSPC; systemd stoppedmodel.13GB unusedownrecipecache movedto tmpfs, rootfree23GB.8e92e25d fixes optionalmetadata failure and proxy503;7CPUtests/ruffpass. NormalRestart=on-failure service started12:27; unchanged inference source b755, availabilitysmoke queued, no formalrepeat.

- `api-recovery-enospc-01` RESTORED13:08:34:full warmup, publicmodels200 and32token official-sampling generation passed; normalbackendRestart=on-failure, NRestarts0. No performance request repeated.

- `long-context-fallback-audit-01`:READONLY user longbenchmark/source audit.>64Kratio1 and>128Kratio2 sharedindex guards reject; MLA>64K rejects; decodevisiblepruning>32Kdisabled, future Fullrows scanned in2048tiles. Live stack actualexecution/sync, no observedPythoncompiler atinstant; first82Kcold extra unresolved. No benchmark/restart/gain credit. Publicservice remainsup.

- `prefill-windowed-index-chain-01`:qualified actual Full→Reindex→MLA long source-window producer/consumer;48exactoutputs,ABABAB200. No16Kcredit. `prefill-paged-mla-chain-01`:12exact outputs, complete paged load/decode/query-owner MLA/reverse exchange qualified; `prefill-paged-mla-contract-02`:largestquerybucket/ratio2 correctness passed.
- `long-index-mirror-chain-06`:native model-entry actual q/w exchange/cachewrites/Full/Reindex/MLA chain passed both ratios; earlierhost-entry/cold attempts retained without credit. `long-index-mirror-sequence-01/02`:changingposition/state,1024exactcomparisons eachgeometry, correctness-only.
- `long-index-mirror-real16-01/02`:superseded numerical comparison, oldfixture>32K aliasednullpage. `long-index-mirror-real16-diagnostic-01/02`:tool failures before validdata, no model evidence. `long-index-mirror-real16-diagnostic-03`:notlaunched, unnecessary afterphysicalpage rootresolved. `long-index-mirror-real16-03`:corrected distinct333pages, actual16layers+head/feedback,4.890051→4.297652ms,delta.592399>2IQR.165892,allperiod/rank/crossarmtokens exact. No16Kextrapolation.
- `long-prefix-serving-01`:combined independently installed31e519fd; fullwarmup official uncached16K→2128EOS8028.980763tps/10.568225ms,14facts/alltokens exactb755. Strict9.5k/10goalsFAILED,prefillregression unresolved.82945/144385/30000180token official diagnostics completed,13.169/21.310/64.861sprefill and11.424/12.329/14.504msdecode.1Kcache896hit observed,tokenexactnessunproven. Publiconline,PR57draft/mainunchanged. Near512K524161diagnosticcompleted:noOOM/restart,3830.849tps/17.318101ms,postCPU PSI9.39%;80tokenreasoningmisstatestail,noqualitypass.500003EOSqualityFAILED:cold/hit499968all419tokensexact,expectedmarkerabsent;acceptedinputtailverified. Rootunresolved,noarbitrarylongqualitypass,norepeatformal.

- `decode-main-baseline-01`:user-authorized one mergedmain16Kdecode trace, sourcee6f0a36c independently installedexact277Pythonfiles, nativehashesmatch, fullwarmupactive. Publicgatewaystopped;0/1/4/5 private lease, CPU10/15/38/43. Existing unprofiled10.568225ms reused, no new formalbaseline. Own7retiredgraphdirectories losslesslycompressed andtarcompare/SHA256verified, rootfree38GB. Step2handoffI32residentarm added and15CPUtests pass; source audit flagslegacyfp4writer1024rowbound and orderedmutationdependency. No componentgain yet.

- decode campaign preparation: raw profiler launch options corrected before any acquisition; interrupted warmup dumps archived losslessly. Current main full warmup complete; sole capture waits SSD>=30GiB. Static factories and I32 remain default-off;71 CPU checks and native database pass. New FP4 pack+roundtrip and I32 mirror clamp/gather share core implementations; HPU producer/consumer and real16 gates pending, gain0. Whole-expert unslicing reuses COMPILER_GRANULARITY_DECISION_20261001 component rejection (17→7 kernels but55.10→88.70us); no repeat of invalid cache-isolated old trials.

### Decode main follow-up: expert / mHC / Attention small-chain screens

Sequential W2, including 44MiB SRAM allowance, preserved exact outputs but retained17 physical nodes and had no >2IQR saving. No real16 or formal run. [decision](decode-expert-sequential-01/STATE.json).

Packed mHC projection/RRMS input preserved C1/C2/C6/C128 gate bits; C1 compiler already used logical views, so physical2→2 and no small-chain gain. Default off. [decision](decode-mhc-packed-01/STATE.json).

Existing wide woA epilogue preserved WoB consumer bits at C1/C2/C6 but no small-chain gain. No repeated inverse-RoPE trials. [decision](decode-woa-wide-01/STATE.json).

- `decode-sampling-static-serving-02`: invalid formal, first decode failed because a two-column token/certificate tensor violated the established native single-integer D2H ABI; one output token, no EOS or performance credit. Preserved result/logs. Replacement uses scalar token*2+coverage, actual selected device tensor remains separate.
- `decode-sampling-scalar-abi-01`: four ranks passed real native scalar copy at vocabulary boundaries and covered/uncovered certificates; background four-rank full-vocabulary fallback uses the same uniform and repairs the device token; correctness only, no speed claim.76 CPU sampling/completion/static-coordinate tests passed.
- `decode-sampling-static-serving-03`: independent installed0ec9eb39; diagnostics/graph dumps OFF; all four workers passed actual scalar completion/full fallback startup validation. Full warmup then one official16K->EOS qualification pending. Formal CPU/card monitoring controller ready.
- `decode-main-baseline-01/DEVICE_ENTRY_ALIGNMENT.json`: reanalysis only: matched127 token frames device entry skew median1864.952us, exit skew1.116us, first recipe~32.8us across cards; span spread1864.187us. This localizes skew to entry/first rendezvous without proving CPU causality.
- `decode-host-entry-01`: SSD-backed preparation only, not launched; committed CPU/wait/switch scopes and explicit phase capture; no full Torch CPU profiler, no graph exports.

- `decode-sampling-static-serving-03` COMPLETE: fullwarmup, official16K uncached→2382naturalEOS;14facts passed; formal16.533458ms, prefill6201.971tps. Four ranks bounded2382/fallback1; no speed qualification/default promotion. `decode-sampling-gil-diagnostic-03`:503GIL samples,187nonblocking sampling errors; mainRPCpoll dominates, no hardware/latency proof. CPU-only producer/certificate/output handoff A~.0268ms vsB~.0236ms does not reproduce the6ms regression, so no polling-yield change made. Next: actual four-card CPU/device jointtrace.

- `decode-expert-access-01`: independent access axes/prefetch16 preserve bytes and consumer outputs, but physical18→23,6batchGEMM→12GEMM, SRAM→DRAM. Rejected before16timing; no gain credit.
- `decode-position-resident-01` quiet retry: A5.576310/B5.211879ms, saving.364431<2IQR.546232; exact tokens/33states/14hidden rows; no qualified gain.232 copied-position GC recipes expose a missed hot-compilation gate.
- `decode-position-serving-01`: installed559a79be, native unchanged; sampler/static+device next-position and full filtered repair warmup, graph/profilerdiagnosticsOFF; known regression repair preflight pending.

- `decode-mhc-norm-01`: shared gates+FFN norm/quant newTPC, C1/C2/C6 five outputs exact; physicalTPC2→1 through router MME; device37.790→40.926us, host53.164→55.042us. No small-chain gain/no16trial. Archived source, removed unused operator from normal build.

- `decode-position-serving-01` COMPLETE: independent559a79be fullwarmup; official16K uncached→2382naturalEOS; TPOT11.392437ms versus accepted10.568225(+.824212), prior regressed16.533458; no baseline/default promotion. All returnedtokenIDs equal the prior reviewed14-fact output. New remaining gap must be diagnosed before any repeat formal.

- decode-position-trace-01: 31 complete hardware cycles; stage entrance median 295 us, end skew 2 us, rank0 handoff mean 1.835 ms. Formal remains 11.392 ms. Two point30 MME attributions unresolved; see REPORT.md.
- decode-input-feedback-01: shared private-root feedback source 74ec9e5b; 92 CPU checks pass; exact real-16/AB gates pending; no gain credit/default promotion.

- decode-input-feedback-01 stopped before candidate creation: resident hand-built reference omitted device_next_position; no HPU candidate/timing result. Corrected cold arm option, cache reused by decode-input-feedback-02.

- decode-input-feedback-02: four-rank token, mutable-state and 14 hidden-row gates pass. First A period cache-file growth invalidated timing; one late scalar/range/clamp recipe preserved. No actual compile-event counter then; no performance conclusion. Tool fixed to warm full trajectories, record compiler events, retain invalid periods and gate CPU PSI. Retry03 reuses01 recipe cache.

- decode-input-feedback-03: complete warm ABABAB 200; 5.218984→5.150237 ms/16; .068747 < .3330545 significance threshold. Exact four-rank tokens/33 states/14 hidden rows; no hot compile. Default stays OFF, no ledger/formal. Resident retained for future independent candidates.

- `decode-device-loop-trace-02`: e8588c95 integrates one-step-ahead ordinary decode, both mapped Engram inputs, private device position/input feedback, alternating immutable draw frames, asynchronous certification and same-position full repair. CPU lifecycle tests pass; startup native repaired-state proof, 160-token service trace and formal result remain pending. No gain credit. `decode-device-loop-trace-01` failed before weights were allocated; allocation-only diagnostics isolate the unsafe rank-placeholder cache path, and the retry uses one safe SSD cache directory.

- `decode-device-loop-trace-02`: device handoff gate passed, overall gain unqualified. Missing TMPDIR in effective command-line settings caused 18,602 recorded TPC fuser output failures; compiled graph lost fusion. Launcher fixed at `75bc1ab2`; no formal request run on this graph.
- `decode-device-loop-trace-03`: full warmup in progress with corrected effective settings, history mirror at `72e1312e`, fresh recipe cache; same native ABI and device closed loop.

- `decode-device-loop-trace-03`: startup stopped before model allocation: long compiler TMPDIR also inherited by ZMQ IPC, exceeded 107-character socket path limit. No measurement.
- `decode-device-loop-trace-04`: same source, short SSD compiler TMPDIR; full startup warming.
- `decode-device-loop-serving-01`: prepared candidate-only formal request; all profiler/trace switches removed; gated on corrected service trace. Not launched.

- `decode-device-loop-trace-04`: fusion restored; startup rollback equality failed on all workers. No formal request or trace. `dc7df85f` replaces numeric equality with exact byte equality, retaining named fatal state mismatches and rejecting nonfinite hidden outputs.
- `decode-device-loop-trace-05`: candidate correctness retry; same device graph, complete warmup, fresh process, corrected SSD scratch, existing compiled recipes reused.

- `decode-device-loop-trace-05`: strict byte check narrowed startup failure to shared.candidate_pool on every rank; hidden and other tested state identical. No request run.
- `decode-device-loop-trace-06`: same device hot path; startup Torch snapshot copies fully drained and compared on CPU, with changed-byte diagnostics. Full warmup retry, source `80dd64fb`.

- 2026-10-04 `decode-device-loop-trace-06`：启动正确性失败，只有 candidate_pool 的 159 字节不同（首偏移 128，在第 0 行内）；隐藏状态和其余已检查状态逐位相同。取消单项 trace／正式请求，不计收益，转为组件复现。
- `decode-micro-resident-01`：真实 16 层常驻工具约 4 分钟加载完成；候选编译前遇到已安装 bridge 缺少依赖策略 API，无计时。固定启动器前置接口和指纹检查；`decode-micro-resident-02` 改用此前已构建的 `decode-serial-policy-01` bridge，其余执行条件不变。
- `decode-control-linear-01`：汇编复核后确认与已有 `mhc-linear-convert-isa-01`／`mhc-linear-small-01` 同一候选；撤回重复原型，不运行硬件、不计收益。

- `decode-micro-resident-02`：ABABAB200，三轮设备差值 −0.260184／−0.288589／−0.098148 ms；五层组真实链状态和 hidden 精确（原记录 8+6 步），记录调用 73→41，joint_info[0]仍949。该方案没有证明物理节点融合且三轮都慢，保持关闭、零台账收益、零正式请求。
- `decode-micro-resident-03`：新增共用设备闭环的生产微链（采样帧→双 Engram→原生16层，证书异步交接，同随机数回退）；父版本 `handoff_sampling_bounded`，五个连续真实输入，再ABABAB200。准备中，不能计收益。

- `decode-micro-resident-03` completed: five consecutive real hidden outputs/state exact. Parent fixture returned the final packed certificate as raw token; fixed fixture decoding, no model arithmetic changed. No timing.
- `decode-micro-resident-04`: token/hidden/mutable state exact on four ranks. Recorded whole-period HPU averages cannot qualify the requested per-step median; retained raw intervals, no gain credit. `decode-micro-resident-05` adds per-step device events.
- `decode-micro-resident-05`: own offline edit changed shared replay.py during cold preparation; source guard rejected before timing. No gain. Fixed launcher freezes ordinary execution source on SSD; workspace remains unchanged and available for development.
- `decode-micro-resident-06`: frozen source, same-process ABABAB200, five-input exact gate, per-step HPU-event medians, device loop flag OFF outside candidate. In progress, no gain credit.
- `decode-ordered-peer-sum-01`: default-off ordered BF16 peer-reduction kernel prototype. Same rank order, FP32 accumulation, one final BF16 boundary; keeps peer communication unchanged. Native TPC compilation and CPU reference checks pass; full-chain performance pending. Hypothesis: remove repeated casts/intermediate sums, not fewer communication points.

- `decode-micro-resident-06` completed: ABABAB200, five-input exact token/hidden/33-state gate on all four ranks; device paired medians save 1.012987 ms/token, three consistent directions, no hot compilation. Micro gate passed only; formal pending. Synthetic chain fallback rate 48/232 in each arm, matching tokens; full-request fallback frequency remains to be measured.

- `decode-expert-scaled-lut-01`: offline ISA/coverage rejection. Safe synthetic shuffle-group input is exact with 5896→4744 simulator instruction packets after clearing inactive unpack bits and choosing the correct shuffle bank. Real checkpoint W13/W2 samples (layers 0/14/39, experts 0/1/42) have 0/225 eligible N256 blocks. No HPU micro, no new serving restart, no gain credit; retain source and reason instead of pursuing an unsupported uniform-scale assumption.

- `decode-device-loop-serving-02`: formal full warmup / official T1, top_p.95, seed42 / uncached16K→2382 naturalEOS: TPOT9.997343ms. Token IDs and output byte-exact to parent11.392437ms (14 facts). Improvement1.395094ms from parent and0.570882ms from original10.568225ms; pending micro batch accepted and closed. Companion trace failed before SDK start because missing HABANA_PROFILE_WRITE_HLTV. New launcher rejects that configuration before loading. `decode-device-loop-trace-07` retries only the companion trace with both raw-profiler initialization flags, same decoder source/native libraries; no repeat formal request.

- `decode-micro-resident-07`: queued default-off ordered peer arithmetic fusion against the already qualified device-closed-loop parent. Five real hidden/state inputs and native ABABAB200; no measurement or gain credit yet. Fixed launcher now waits for owned HPU teardown, freezes both baseline/candidate factories, and validates both source fingerprints. Companion trace uses idle profiler initialization before its diagnostic request; no repeat formal run.

- `decode-device-loop-trace-07`: profiler start HTTP200, then VllmWorker-3 exited during the first prefill (zero emitted tokens); executor retired peers. No raw metadata or new speed result. Shim API/HwTrace version warning recorded as unresolved correlation, not a proved cause. Formal9.997343 baseline remains valid. Owned allocations released without resets; no repeated service load, next native micro now loading.

2026-10-04 decode-micro-resident-08: terminated before correctness/timing in initial sampler setup; memory-pressure evidence retained; zero gain credit. decode-peer-post-collapse-01: default-off fused ordered peer sum/residual consumer under development; no measurement yet.

2026-10-04 decode-peer-post-collapse-01: native component five exact checks, ABABAB200 three savings8.950/12.500/5.773us per boundary; teardown fixed and untimed retirement-check exited0. Forecast only40attention boundaries=0.358ms, noMoE or formal credit; default off, real16integration pending. Meta35pass12skip, sharedCPU36pass, kernel database pass.

2026-10-04 decode-expert-affine-address-01: both affine-counter and constant-address paired variants byte exact offline, but5252→5572packets (+6.09%); noHPU/model loading, zero gain credit. Stop this address-rewrite direction; parent nibble arithmetic and SRAM pairing retained.

2026-10-04 `decode-c1-token-wide-sat-01/02/03` (SSD)：复用 DSpark SAT 的 C1 数量验收。01 工具声明1个通信点而实际4个，计时前失败，已改为实际数量。02 五输入四卡逐位一致，MoE物理节点19→28，四层三轮差0.000229/0.000435/0.000818ms，数量门槛失败，不记收益。03 手写六路W13输出布局和六路W2映射，节点19→24，四层三轮差−0.047896/−0.047654/−0.047748ms；五输入逐位一致，SRAM复核发现整六路W2解码输出落到DRAM，W13的4个N块仍在SRAM、W2降为6个GEMM、Silu仍6个。两次失败后暂停SAT C1，开关默认0。所有正式数字沿用9.997343，不测整模。物理节点审计工具已加入，可复用。

2026-10-04 `decode-attention-prologue-01` (SSD)：下一模块手写KV norm/RoPE/量化/环形cache和decoded镜像发布TPC；直接使用VLM中已舍入BF16，避免读回刚写的KV输出。TPC编译通过，尚未做生产微基准/物理数量验收，不记收益。

2026-10-04 KV publish handwritten fusion: archived attention-prologue-03/04 numerical fixes; 05 and 06 exact five inputs/four ranks. 06 final consumer nodes 20→14, but AB rounds [0.005690000000000028, 0.01693550000000002, -0.0005055000000000059] ms/layer: inconsistent direction, no gain-ledger credit, default off. Pause isolated variants; fold into a larger compatible Attention module only.

2026-10-04 mHC MME plus handwritten RRMS/gates/norm: 01 single-group timing noisy, 02 repeated 16 native groups exposes consistent regression [-0.015280875, -0.015333718750000003, -0.015293062499999996] ms/FFN phase. Physical 5→4; norm/quant/router exact, gates max error 7.75e-7. No gain entry. 03 register reduction rejects exact-gates fixture2 and is reverted. Pause this partition: norm/router must remain independent of control MME; LUT cache warning alone did not explain it.

2026-10-04 CSA publish05: reference native KV codecs asserted (04 generic reference invalid); 5 inputs/all states plus scores/MLA/WO exact across 4 ranks. Physical consumer 63→27; AB savings [0.052113343749999985, 0.0703136875, 0.06127037499999999] ms/source. Ratio2 sources2/8/14: estimate 0.18381112499999996 ms/token; no ratio1 or formal credit.

2026-10-04 CSA publish07 supersedes05: production static compiler, packed writer, hot off, true mirror reader; 5 inputs/all states and consumers exact/four ranks. Physical 44→17; AB [0.0468718125, 0.0508414375, 0.053027531249999996] ms/source, all positive. Ratio2 applicable sources2/8/14 estimate 0.1525243125 ms/token, no R1/formal credit. No double-count05.

- 2026-10-05 reference study: pinned DeepGEMM/Ascend/TileKernels/FlashMLA/DeepSelect/SGLang and open PR42245; source-to-Gaudi notes at `evidence/REFERENCE_NOTES.md`. Shared-coordinate entry is default-off; producer C1/C2/C6 fixtures exact, complete-stage timing unavailable after a legacy fixture contract mismatch and host-memory exhaustion. No gain for those attempts.
- Candidate-block coordinate producer: actual query-weight projection → row preparation → stock gather/MME, five fixtures bit-exact, native three rounds positive; 26→20 physical nodes, median 2.317531 µs per logical scoring chain. Micro-only estimate0.060256ms/token using26 calls, flagdefault0. See SSD `decode-candidate-coordinate-chain-01/DECISION.json`; candidate-only physical recovery in `decode-candidate-coordinate-physical-01`, no repeated timings. Whole-output metadata variant under evaluation; no second gain credited.

- Qualification correction: candidate coordinate small-shape screen is not a production-shape gain. Withdraw 0.060256 ms/token extrapolation; pending qualified total remains 0.8516655625. Full 262144-row mirror / 2048-block chain validation is required. Whole-output small screen 20→19 nodes but three negative pairs, off.

- 2026-10-05 accepted batch03 9.564282 ms/token; six entrypoint defaults committed in23551208. Attention output tiny-fusion archive committed in0dd878fb. New active MoE accumulator0ms.
- MoE structural screens on SSD: pipeline3-02 21→17 complete nodes,11 routed,5×4 exact, slower13.144us/layer; prefetch-01 preserves17 nodes and changes final order toW13/W13/W2/W2, slower3.501us/layer; plain-w13-01 21→15 complete /9 routed,one39.32MB SRAM W13,5×4 exact,slower1.448us/layer. All off, no gain credit or formal run.
- MoE dictionary-01 invalid native stack normalization; dictionary-02 physical15 but first exactness input fails. Isolated decoder simulator confirms unsafe group-dictionary assumptions with byte SHUFFLE and unpacked high bits. Runtime prototype removed; source/ISA/probes archived. No gain credit.
- MoE sram-handoff-01 manual tiny RMW intermediates cause decoded weights to spill; rejected before native replay/timing. Next source-guided test retains the baseline producer bundle and changes only noncommon pipeline slice count4→2 during candidate cold compile.

- 2026-10-05 MoE two-slice control: `decode-moe-two-slice-02` weights SRAM, 21→17 nodes, exact, **slower 2.356 us/layer**, disabled.
- 2026-10-05 exact manual SAT loop expansion: `decode-moe-unroll-chain-01`, five × four ranks exact, native ABABAB saves 0.310 us/layer; superseded by next combined candidate.
- 2026-10-05 one-route SiLU + expanded decoder + two slices: `decode-moe-streamed-01`, 21→17, five × four ranks exact, three positive rounds, saves 0.760 us/layer; forecast 0.030397 ms/token, not formal. W13 outputs and activations still DRAM.
- Next: `decode-moe-aligned-01` sets W13 producer granularity to one complete expert (1280 columns), so two slices can align at 3840/3840 instead of 4096/3584. Exact compiled SRAM handoff must be checked before timing.

- `decode-moe-aligned-01`: route-granular access achieves 3840/3840 slices and exact five-input results, but W13 activation still DRAM and slower 0.841 us/layer. Zero credit. `decode-moe-aligned-flat-01` removes the flatten→route reshape boundary with a flat-input exact SiLU, requiring actual up-projection SRAM output before timing.

- `decode-moe-aligned-flat-01`: flat SiLU still no W13 activation SRAM, rejected before timing.
- `decode-moe-full-sram-01`: explicit 59.4 MB scratch exceeds actual 47.5 MiB pool, rejected compilation before timing. Next `decode-moe-full-sram-reuse-01` shares dead weight storage with explicit completion edges and a 40 MiB section cap.

- `decode-moe-full-sram-reuse-01`: audit invalid (shared W2 misidentified as routed W13 by stored axis); actual graph SRAM valid, no timing.
- `decode-moe-full-sram-reuse-02`: actual 21→15, full routed intermediates SRAM, five × four exact; slower 14.16 us/layer. RMW finalizer adds shared-expert completion to bundle entry, so zero credit.
- `decode-moe-pipeline-sram-01`: move terminal products/scales outside RMW, use two independent three-route branches. Compiler rejects lost SRAM placement on activation reshape; no timing. Version 02 preserves explicit section offsets across logical aliases.

- `decode-moe-router-four-slice-01`: production router restored; retained compiler slicing policy; SRAM W13→SiLU exact; micro only.
- `decode-moe-dual-quant-01`: combined FFN dual-quant + streamed SAT, five exact fixtures, three positive native A/B rounds; supersedes previous estimate, defaults off.
- `decode-moe-router-pipeline-01`: earlier router-inclusive two-slice result archived under new policy.
- `decode-moe-pipeline-sram-03`: explicit RMW exact but slower; no credit.
- `decode-moe-aligned-scale-02`: scalar scale alone did not remove fixed-ID norm diamond; no timing/credit.

- `decode-mhc-unpack-post-01`: missing native dispatch discovered before timing; fixed and covered by database test.
- `decode-mhc-unpack-post-02`: exact but two-half loads add load-use dependencies; superseded after ISA review.
- `decode-mhc-linear-post-01`: native linear conversion + exact deferred gates, five inputs exact; three positive rounds; replaces earlier mHC estimate.
- `decode-mla-reuse-vector-01` / `decode-mla-publish-vector-01`: exact 128-value codecs through Q/KV→MLA→WO→peer/mHC; consistent positive native A/B, unchanged logical stages; defaults off.

- `decode-woa-single-amax-01`: ISA-only precursor; backend unroll pragma ignored, no timing.
- `decode-woa-single-amax-02`: cached BF16 row, one amax, one scale load per group, frontend unroll; five inputs/four ranks exact, three positive native rounds; default off.


## 2026-10-05 roofline pass

- `evidence/ROOFLINE.md`: historical named trace vs current official baseline kept separate; priorities A→E.
- `decode-peer-prune-01`: single-box AG empty scale-out stream pruning;5×4 exact,3 rounds slower~.10µs/point; zero credit.
- `decode-pcie-capability-01`: owned2/3, DMA-BUF export/import and16KB data correctness passed; not a timing result.
- `decode-pcie-oneshot-01/02`: old benchmark's two-input contract disagreed with the current one-input TPC GUID; no timing.
- `decode-pcie-oneshot-03`: repaired private standalone probe, TP2 10KB exact; host-submitted batch20.86–20.89µs, not a native TP4 gain.
- `decode-pcie-native-04` and05–08: compiler crash; later gdb pins the actual null call to missing `GetSuggestedManipulation` in the old TPC library. Initial runtime-mix attribution withdrawn. Private wrapper adds the hook;09 compiles. Zero performance credit.
- `decode-mhc-official-tolerance-01`: official-equation tolerance passed, native chain slower13.89µs; no credit.
- `decode-mhc-shared-rrms-01`: one RRMS statistic removes redundant scans but remains slower13.97µs; disproves scans as the sole cause. MME resource contention with WO remains the next attribution target, not a numerical rejection.
- `decode-mhc-parallel-k-01`: TPC eight independent K accumulators plus deferred post; upstream tolerance passed5×4,3 positive rounds; replaces linear-post ledger item.
- `decode-roofline-audit-01`: exact compiler SRAM overlap examples; actual expert ISA109 bundles/33loads/32stores/96decode ops. Two-op dictionary direction stopped; no gain assigned to static analysis.

- `decode-pcie-native-10/11`: port to actual runtime APIs;10 rejected an empty collective plan;11 compiled and captured. No latency credit.
- `decode-pcie-native-push-12/13`: invalid native timing; stale stream bookkeeping missed native completion; epoch check rejected80/535. Zero credit.
- `decode-pcie-native-push-14`: explicit SCAL completion wait,535 epochs and10KB BF16 TP2 sum pass;~9.19µs completed primitive. Not TP4 nor production-chain qualification.
- `decode-mhc-split-k-01`: official accuracy5×4 and55 CPU checks pass;3 rounds slower3.89µs/boundary. Patch archived, production restored; zero credit.
- `decode-pcie-native-nrank-15`: zero-target completion wait threw after capture; exceptional path missed imported DMA-BUF unmapping. Modules0/1/4/5 retain three driver context references each; ordinary reset cannot finish. No foreign processes stopped. Reboot coordination pending. nrank16 was stopped while waiting, never launched hardware. See SSD incident record.

- `decode-ffn-bf16-quant-01`: simulator5 fixtures exact,3791→3439 instructions. Native graph classification expected the old expert GUID, so no timing; corrected to quantizer GUIDs for the identical expert parent.
- `decode-ffn-bf16-quant-02`: complete correctness/placement passed; pre-timing monitor failed on unrelated resetting-card N/A telemetry. Owned group retired. Monitoring now preserves unknown rows and requires all owned modules to be present;28 CPU tests pass.
- `decode-ffn-bf16-quant-03`: user-authorized2/3/6/7, five×four exact,3 positive rounds; incremental forecast.001143438ms, default off.
- User recovery decision: continue on usable cards. No reboot authorized or performed. Cyclic DMA-BUF test quarantined; all affected0/1/4/5 remain excluded.

- `decode-dense-weight-stream-01`: five x four exact, SRAM producer confirmed; three-fragment copy+MME slower ~5.21 us/boundary.
- `decode-dense-weight-stream-02`: compile option 1 rejected by the tool allowlist; no timing.
- `decode-dense-weight-stream-03`: one fitting 10MB SRAM operand, five x four exact; three rounds slower ~3.01 us/boundary. Second performance failure; stop direct weight-copy approach, zero credit.
- `decode-peer-overlap-control-01`: native topology guard rejected one-point overlap groups before timing.
- `decode-peer-overlap-control-02`: eight-point groups, five x four exact; ~0.0365 us/boundary prototype improvement, not production integrated and not credited.
- `decode-index-gain-replica-01`: five x four exact query/gain/score bytes, three positive native rounds. Eight-layer forecast .03070325ms, generic TP C1 integration default off.

- `decode-mhc-swizzled-control-01`: five x four exact, three positive native rounds (~.138us/boundary), same number of logical stages. Cold common-path C1 weight packing default off; .005530156ms forecast for 40 qualified boundaries.
- `decode-mhc-positive-recip-01/02`: static-only reciprocal trials; uncontracted loop did not shorten critical instructions. No hardware timing/credit. The 25-value specialized post contract removes the unused RMS branch and its invariant setup; checked in `decode-mhc-rrms-post-01` first.

- `decode-mhc-rrms-post-01`: specialized 25-value controller layout, five x four exact. Three minuscule positive differences (2.5–7.3ns/boundary), literal forecast recorded without a material-gain claim.
- `decode-mhc-early-gates-01`: both arms use the qualified parallel controller; returning gates to the WO producer is slower 1.255–1.268us/boundary. Keep deferred post. Zero credit.
- `decode-mhc-positive-recip-02`: FMA Newton variant still has an equally long serial Sinkhorn loop. Static rejection; no hardware run. Archived, removed unqualified numerical variant from maintained source.

- `decode-mla-static-softmax-01`: five x four exact, compiler producer still19 nodes and same SRAM locations. Full 640-row unroll is slower .219–.312us/layer in3 native rounds; sparse local traffic reduction did not shorten the full chain. Archived patch; no maintained default and zero gain credit.

- `decode-dense-cold-transpose-01`: five x four exact; WO [N,K]→[K,N] with native MME transpose flag, no hot copy; three rounds slower .104–.114us/boundary. No credit.
- `decode-qkv-cold-transpose-01`: same cold layout change on QKV [1792,5120], full Q/KV→MLA→WO→peer consumer. Five x four exact; round differences −.000037645/+.000019777/−.000042418ms, inconsistent, no credit. Stop cold-layout transpose for these shapes.
- Common replay integration: index-gain replica reduces the StageVariant collective count only for C1; C2/C6 retain two query/gain points. 100 CPU checks cover topology, resource teardown, native input/tail and overlap. Fresh native database and50 native metadata checks pass.
- `decode-merged-parallel-control-01`: queued real16 native official-sampling comparison for existing MERGE_LOCAL_SEGMENTS with the qualified parallel controller/deferred gates in both arms. This earlier direction was interrupted before measurement; formal batch03 logs prove8 extra mHC partitions per4-layer group. No forecast credit yet.

- `decode-merged-parallel-control-01`: host bundle preflight repaired before opening devices. Untimed real16 preparation then reached15.1GiB host headroom and the launcher stopped its own process group. All2/3/6/7 workers retired, devices returned768MiB. No timing or gain credit. Repartitioning itself is still unmeasured.
- Resident input lifecycle fix: serial A/B plans reused weights but each engine constructed another privately pinned Engram table map. `tools/deepseek_v41_device_chain.py` now reuses one input producer per host/device/token bucket, with independent replay plans and sampling frames. Prior consumer must drain/retire before reuse.35 CPU ownership/rollback/resident tests pass. `decode-merged-parallel-control-02` reuses completed native binaries and cold recipes; no model change or gain credited to the fixture repair.

- `decode-merged-parallel-control-02`: five input states x four ranks exact,33 mutable tensors exact, full feedback exact and no timed compilation. Same-process real16 device medians4.186109→4.374776ms; paired savings−.196708/−.175141/−.202181ms. Compute recipes73→41, peer points42→42. Fewer boundaries remove32 independent control/communication overlap windows; keep MERGE_LOCAL_SEGMENTS off, zero gain credit.

- `decode-dense-dma-prefetch-01`: real DmaMemcpy10MiB→SRAM→MME proved in all four final graphs; five inputs x four exact; native complete-boundary savings−.009744/−.009750/−.009746ms. DMA variant is slower even with the intended memory placement. Archive source and remove unused native registration; keep the reusable DMA/SRAM proof check. Zero gain, no formal request.

- PCIe acyclic star01/02 (SSD): no reciprocal imports. Bounded polling error and612 changing epochs cleanly release2/3. Same-process native primitive A18.1661us→B16.0945us; still above6us even atTP2, do not extend toTP4 or count as model gain. Nrank16 remains quarantined. Maintained CPU import-DAG/epoch guards:19 checks.

- `decode-mhc-producer-fusion-01`: default-off scheduling hypothesis, moves pure parallel control into the PRE-peer producer, unlike rejected post-peer merge. Requires ready inputs and nonaliasing state.50 CPU scheduling/import/DMA checks pass; native real16 A/B queued on available2/3/6/7 with separate recorded plans and shared drained Engram roots. No gain credit before hardware result.

- Producer-fusion01 failed before timing: initial graph guard rejected resident constants/read-only inter-partition nodes.27 CPU checks now cover those nodes, real split composition and mutable aliases. Retry02 retains every peer point and rejects late peer-dependent inputs; no gain credit.

- `decode-kv-hardware-codec-01`: offline Gaudi2 simulator, all256 E4M3FN encodings. Native conversion alone has15 mismatches; finite-top-exponent/negative-zero correction removes all.79→74 VLIW instructions, unchanged5 loads/7 stores. No hardware latency, no ledger credit, not a new default.

- Producer-fusion02 exposed interleaved Bridge placeholder ABI error before timing. CPU reproducer confirms old order fails. Fixed03:5 states×4 ranks and feedback exact, no hot compile;73→41 compute calls,42 peer points unchanged. Device3-round regressions0.142305/0.150671/0.130405ms; A4.188502→B4.3295135ms. Archive source, stop both consumer/producerside merging directions, remove unqualified pass from runtime. Ledger unchanged.

- Producer-fusion03 compiler diagnosis: all11 observed merged physical graphs place control after their last MME (Exec_idx). Fewer recipes did not establish the intended earlier overlap. This is static ordering evidence, not a measured stall attribution. Require compile-only earlier placement before another device trial; no new trace.

- `decode-host-shared-peer-01`: no device imports; one shared mapped-host mailbox, bounded native polling. First212 changing epochs pass (one pair20.286/12.600us), then a later period times out on ready/publication flags (codes2/7). Timing unqualified, no TP4 expansion/no gain. All owned processes exited;2/3 returned768MiB without reset. Archive for protocol diagnosis.
