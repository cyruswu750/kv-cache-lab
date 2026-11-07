# M4-B protocol: trace evaluation of C0 / C1 / C2

> **About this copy.** This is a maintained copy of the M4-B protocol. After both evaluations ran,
> its status notes and workflow wording were edited; no experimental rule, parameter or check was
> changed. The exact text fixed before measurement, whose SHA-256 the M4_EVAL_V1 and M4_EVAL_V2
> freeze manifests record, is the version at commit
> [`b858d9e`](https://github.com/cyruswu750/kv-cache-lab/blob/b858d9e9f1ff78bd157b2af80720b54ae85cda84/docs/M4B_PROTOCOL.md). Results and their limits are in `docs/results.md` (M4).
>
> **Provenance.**
> - **Fixed before any GPU timing:** the design choices of decision D33 and every parameter below
>   (workload, seeds, configurations, budgets, R, statistics, comparisons, validity checks, deadline
>   and stopping rules). Every number below, apart from the dated notes in this box, is either a
>   deterministic offline trace statistic computed on the CPU from the generator alone, or a
>   parameter value fixed then.
> - **GPU correctness gate (§9), 2026-11-04:** attempt 1 failed on a defect in the wrapper's pytest
>   temporary directories, since fixed; attempt 2 passed its tests, but three coverage gaps against
>   §9 were found afterwards (C1 and C2 counters not fully reconciled with the models, restored bytes
>   not compared directly with the C1 entries, per-request memory diagnostics not retained);
>   attempt 3 ran the strengthened gate and passed: 46 tests, 311 s of the 900 s cap. Detailed gate
>   evidence is kept outside the repository.
> - **Timing pilot (D34):** one timing-only session on the calibration traces plus the warmup, with
>   no evaluation seed, solely to inform the §9.5 runtime-feasibility decision. It took 189 s;
>   ten-session projections were 1,891–1,935 s, below the 2,160 s (80%) threshold. It selected no
>   policy, configuration, evaluation seed, success criterion, statistical rule, session count or
>   deadline; one session measures no run-to-run variability and does not guarantee that ten
>   sessions complete in time. Its evidence is kept outside the repository.
> - **Freeze (D35):** the single pilot was accepted as the planning basis, with its documented
>   uncertainty (one session; an approximate calibration-to-evaluation workload mapping; unmeasured
>   overhead between sessions). 2,700 s stayed the hard deadline, with no additional sessions,
>   retries, extensions or parameter changes. The freeze records this document's bytes, and the
>   manifest digest was recorded outside the repository. No gate or pilot time informed any design
>   choice.
> - **Execution:** M4_EVAL_V1 ran once under this protocol and is invalid under its whole-tier
>   warm-disk rule (no performance result). M4_EVAL_V2 ran once under
>   `docs/M4_EVAL_V2_PROTOCOL.md`, which amends only the warm-disk preparation.

Context: `docs/design.md` §12 (M4-A) and §13 (M4-B preparation), decisions D32–D35. The
reviewed M4-A code is unchanged. C1-PA is a separate class, and the plain-LRU C1 is kept for the
offline diagnostic.

## 1. Question
On four synthetic request traces, sequential and batch 1, how does each tested configuration change
the trace-mean synchronous first-token latency (D22)?

The configurations, each with explicitly accounted and **unequal** resources:
- tiered always-restore (C2), with either restore backend;
- no cache (C0);
- a bounded GPU-only prefix cache (C1).

This is a workload-level microbenchmark of restoration. It is **not** service TTFT, throughput,
concurrency or production traffic.

## 2. What is fixed
| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, bf16, SDPA, as in M2/M3 |
| Chunk | C = 256 tokens; one chunk payload = 7,340,032 B; one chunk file = `chunk_format.file_size(ns)` |
| Machine | the M3 machine and stack. The freeze records the stack, and any difference is invalid evidence |
| Request semantics | M4-A replay: one prefill per request (restore + suffix prefill + admission), greedy argmax; `steps = 1` in timed sessions |
| First-token rule | L ≤ N − 1 at chunk boundaries (`kvlab.boundaries`), for every configuration |
| Identity | `identity.chain_keys` under the model namespace, for C1 and C2 |
| C2 engine | `AlwaysRestoreEngine` + `TieredKVCache`, unchanged |
| Correctness contracts | D10 and D12 unchanged. No configuration is compared bitwise with another where the splits differ |

## 3. Traces (synthetic; generator `kvlab/traces.py`, unchanged)
**Seeds:**
- calibration seeds 41001–41004: used only by the correctness gate (§9) and implementation
  tests. The one exception is the D34 timing-only feasibility pilot, which selected nothing;
- evaluation seeds 42001–42004: used only by the evaluation sessions.

Both seed sets come from the same generator with the same parameters. Token range: the default
[10, 150000), ordinary vocabulary ids.

| Trace | Generator and parameters |
|---|---|
| **T0** no-reuse control | `no_reuse(n_requests=48, min_len=800, max_len=1900)`. The length range is chosen to match T3 |
| **T1** shared system prompt | `shared_prefix(n_requests=64, prefix_len=1536, suffix_min=16, suffix_max=200)`. The prefix is 6 chunks; a suffix never completes a 7th chunk |
| **T2** multi-turn | `multi_turn(n_conversations=8, n_turns=6, opening_len=384, user_min=32, user_max=160, answer_min=64, answer_max=320)`, turn-major round-robin |
| **T3** document reuse | `doc_reuse(n_requests=64, n_docs=12, instruction_len=256, doc_min=384, doc_max=1536, question_min=16, question_max=96, zipf_s=1.0)` |

**Offline statistics of the evaluation traces** (C = 256; `trace_stats`). They are checked by the
committed `tests/test_m4_spec_offline.py`, and were first computed with a local preparation script
that is not part of the repository:

| Trace | seed | SHA-256 (prefix) | requests | prompt tokens | prompt length min / median / max | oracle reusable tokens (ratio) | requests with reuse | unique chunks W (MiB) | largest request (complete chunks) |
|---|---|---|---|---|---|---|---|---|---|
| T0 | 42001 | `f04ff29d4817…` | 48 | 63,538 | 853 / 1,239.5 / 1,892 | 0 (0%) | 0 | 222 (1,554) | 7 |
| T1 | 42002 | `5f9da2bf6661…` | 64 | 105,096 | 1,553 / 1,645.5 / 1,733 | 96,768 (92.1%) | 63 | 6 (42) | 6 |
| T2 | 42003 | `05191908301a…` | 48 | 57,337 | 418 / 1,187.5 / 2,191 | 36,096 (63.0%) | 40 | 56 (392) | 8 |
| T3 | 42004 | `1c30ff4fbfcb…` | 64 | 90,692 | 802 / 1,366.5 / 1,864 | 69,120 (76.2%) | 63 | 58 (406) | 7 |

The full digests are frozen with the protocol. The oracle bounds every configuration: no
configuration can restore more than the oracle-reusable tokens.

## 4. Configurations and backends
All configurations run in every session, on every trace, with fresh state per trace (§6).

| ID | Configuration | Persistent KV between requests |
|---|---|---|
| **C0** | `NoCache`: full prefill; no keys, lookup or admission | none |
| **C1-PA(W)** | GPU-only cache with the **prefix-aware** policy (§5), capacity W chunks: the GPU-resident ceiling. On T1–T3 it never evicts, so its cache decisions equal C1-LRU(W)'s (implementation overhead and latency may still differ). On T0, which uses T3's budgets, capacity 58 is below T0's 222 unique chunks: it evicts 222 − 58 = 164 times | ≤ W × 7,340,032 B on the GPU |
| **C1-PA(W/2)** | the same, with capacity ⌈W/2⌉ chunks (ρ ≈ 2). This is an **artificial cap** | ≤ ⌈W/2⌉ × 7,340,032 B on the GPU |
| **C2-base** | tiered always-restore, `restore_backend="baseline"` (the project default), host H = ⌈W/2⌉ slots, disk D (below) | none on the GPU; pinned host slab; disk |
| **C2-batch** | the same tiers, `restore_backend="batched"`, `restore_staging_chunks=4`. This backend is opt-in and explicitly selected; the default is unchanged | as C2-base, plus transient staging |

**Backends:**
- Both C2 backends are carried, because the default-backend decision is open.
- M3_CONFIRM_V1 confirmed the batched backend only for its own microbenchmark scope.
- M4 does not change the default.

**Not timed:** C1-LRU(W/2), the reviewed M4-A plain LRU, is reported **offline only**, as
key-level restored tokens and dead entries (§5). It is an offline diagnostic, not a timed arm
(decision D33). The gate (§9) still checks its correctness on the GPU.

**Budgets per trace** (capacities in chunks; bytes = chunks × 7,340,032):

| Trace | W | C1-PA(W) | C1-PA(W/2) | C2 host slots H (slab = (H + 1) × payload) | C2 disk D |
|---|---|---|---|---|---|
| T0 | (uses T3's) | 58 | 29 | 29 | 60 files |
| T1 | 6 | 6 | 3 | 3 | 8 files |
| T2 | 56 | 56 | 28 | 28 | 58 files |
| T3 | 58 | 58 | 29 | 29 | 60 files |

**Budget details:**
- **Disk:** D = (W + 2) × `file_size`. For T1–T3 this holds the whole working set, so C2 never
  loses a chunk. Disk evictions must be 0, and that is checked.
- **T0:** it uses T3's budgets. The configurations stay fixed while the workload changes. T0
  is expected to evict in C1, and to spill and evict on disk in C2. That overhead is part of
  the negative control. The expected eviction counts are predicted by the key-level models
  and checked (§8).
- **The C2 host LRU spills to disk** and does not drop. With D ≥ W **and successful
  admission, spill and read I/O**, C2's restore length on T1–T3 must equal the oracle, request by
  request (a check, §8). Disk capacity alone does not guarantee this: a failed write is
  recorded and skipped by design. The fault counters are therefore validated as well (§8).

## 5. C1 policy: why prefix-aware, and what plain LRU does
**Findings on the reviewed C1** (`kvlab/gpu_prefix_cache.py`, plain chunk LRU):
- **Method:** CPU, deterministic, tiny bf16 Qwen2, C = 4, using the unchanged class.
  - The findings come from a local preparation probe, not part of the repository.
  - The committed `tests/test_prefix_aware_cache.py` reproduces the oversized-request and
    shared-root cases, for plain LRU and for C1-PA.
  - Each finding was cross-checked with the strict reference and with `lru_restore_lengths`.

1. **Oversized request** (more complete chunks than the capacity):
   - Admission evicts the request's own leading chunks in order, and only the last `capacity`
     chunks remain.
   - The same 5-chunk prompt at capacity 3, repeated, restores 0 tokens every time; the cache
     holds A2–A4.
   - No leading chunk remains reusable.
2. **Retained later chunks:**
   - They are unreachable under the longest-leading-run lookup: no lookup can reach them while
     an ancestor is missing.
   - They become useful only after some request recomputes and re-admits the missing ancestors.
   - They still count against G, so `retained_bytes()` is correct but includes dead bytes.
3. **Admission × eviction × hits:**
   - Hits are touched root-first, so a request's shared root is the least recent of its chunks.
   - If the request's new chunks overflow the capacity, they evict that root while keeping
     descendants.
   - Example (capacity 4): X = s0 s1 x2 x3, then Z = s0 s1, then Y = s0 s1 y2 y3 y4. Y evicts s0,
     so the next Z restores 0 although s1, y2… are cached.
   - With every request ≤ capacity, a request's own chunks are its most recent entries and are
     not evicted during its admission. Present chunks are not refreshed by storing (as for the
     host tier).
   - Continuation (suffix prefill, decode) never touches the cache. This is unchanged and
     tested.
4. **Storage accounting:**
   - Each entry owns exactly one chunk-sized contiguous storage. `retained_bytes()` equals
     entries × payload, and no view retains request storage.
   - On a hit, the request KV aliases neither the entries nor the gather scratch: the D4
     constructor makes its own copy.
   - Transiently, three L-token payloads therefore coexist: the entries, the gather scratch
     (`restore_scratch_bytes`) and the constructor copy.
   - The constructor copy is not a separate reported term. `request_kv_bytes` reports only the
     N-token tensor after the suffix prefill.
   - C2 has the same D4 copy. Per-request device peaks must therefore be measured on the GPU
     (§8); they cannot be derived from the reported terms.

**Offline effect on the evaluation traces** (restored tokens; key-level models, CPU):

| Trace | capacity | plain LRU | prefix-aware | oracle | LRU dead entries, max |
|---|---|---|---|---|---|
| T1 | 3 | **0** | 48,384 | 96,768 | 3 |
| T2 | 28 | 8,192 | 8,448 | 36,096 | 6 |
| T3 | 29 | 63,744 | 65,536 | 69,120 | 5 |
| T0 | 29 | 0 | 0 | 0 | 6 |

- On these traces, at the computed capacities (W, ⌈W/2⌉, ⌈W/4⌉), prefix-aware is never below plain LRU on
  any request. This is an observation on these traces, not a general property.
- At capacity W, on T1–T3, both equal the oracle.
- **T1, plain LRU:** every request is oversized at capacity 3, so plain LRU is degenerate.
  It would turn the GPU-only baseline into a straw man.
- **T2:** round-robin access defeats both recency policies at ρ ≈ 2. This is a property of the
  trace, and it is reported as such.

**Proposed C1 policy, "prefix-aware LRU" (C1-PA):** a new class, so that the reviewed
`GPUPrefixCache` is not silently changed (§11).
- **Lookup, hit touches and store order:** exactly as C1-LRU. A present chunk is skipped
  without refresh.
- **Admission:**
  - an absent chunk is admitted only if its parent is cached (prefix closure);
  - when full, the policy evicts the least recently used **leaf**, that is, an entry with no
    cached child, that is not one of the current request's chunks;
  - if no such leaf exists, it stops admitting the rest of this request
    (`write_skips["prefix_capacity"]`).
- **Invariants, checked after every request:**
  - the cached set is prefix-closed, so every cached chunk is reachable;
  - retained bytes ≤ G;
  - restore lengths ≤ oracle;
  - an oversized request keeps its first `capacity` chunks.
- **Rationale:** this is the standard structure of radix/prefix caches (leaf-first, LRU among
  leaves). It strengthens the GPU-only baseline. Its choice rests on the CPU mechanics above,
  **not** on any GPU measurement.
- **Asymmetry, disclosed:** C2's host and disk tiers keep plain chunk LRU, unchanged.
  - C1-PA avoids the demonstrated LRU self-eviction pathology.
  - Offline, on these traces, it restores at least as many tokens as plain LRU.
  - That does not guarantee lower C1 latency: prefix maintenance and different
    restore/recompute splits have their own costs. No direction of latency bias is claimed.

**Scope note.** Prefix-aware eviction had originally been left out of M4. This protocol re-admits
it for C1 only, because of findings 1–3 (decision D33). C2's policies are not changed.

## 6. Sessions, order and state
- **Session:** one fresh process. It runs all four evaluation traces, in a seeded random trace
  order.
- **Session count:** R = 10 sessions, with session seeds 44001–44010. This is exactly one batch.
- **Per trace:** fresh instances of every configuration. Specifically:
  - a new C1 cache;
  - a new `TieredKVCache` with a new host store and a new, empty disk directory under
    `.kvcache/m4/<session>/<trace>/<config>/`.

  Disk directories are deleted after that trace's checks (untimed). No state crosses traces or
  sessions.
- **Interleaving:** for each request in trace order, the five configurations run back to back,
  in a seeded random order per request. Each configuration advances its own state only.
  - This pairs the configurations in time, as M3 interleaved the backends. Random order
    reduces order imbalance, but it does not guarantee identical regime exposure.
  - Shared coupling, which interleaving does not remove:
    - both C1 caches stay allocated on the device while the other configurations run;
    - all configurations share the CUDA caching allocator, the page cache and thermal state.
  - Every latency conclusion is therefore **conditional on this interleaved, co-resident
    experiment**.
- **Process warmup** (untimed, excluded, kept as evidence):
  - **Warmup traces:** after model load, four warmup traces are replayed through every
    configuration, interleaved as above. They use the §3 parameters except for the counts:
    - `no_reuse(43001, n_requests=4)`;
    - `shared_prefix(43002, n_requests=4)`;
    - `multi_turn(43003, n_conversations=2, n_turns=3)`, which gives 6 requests;
    - `doc_reuse(43004, n_requests=6, n_docs=2)`.
  - **Capacities:** the §4 rule applied to each warmup trace's own W.
  - **Instances:** fresh ones for each warmup trace, all thrown away afterwards.
  - **Required coverage**, verified offline with the key-level models before the freeze, and
    rechecked by the validator from the warmup records (exit 3 if missing). Across the warmup:
    - each cached configuration has ≥ 1 hit and ≥ 1 admission;
    - C1-PA(W/2) has ≥ 1 eviction or prefix-capacity refusal;
    - each C2 backend has ≥ 1 host hit, ≥ 1 disk hit and ≥ 1 spill.
- **Within-trace warmup:** none, W = 0. Cold-cache requests are part of the workload and are
  measured. The secondary "steady-state" metric excludes the first quarter of each trace.
- **Warm disk only, enforced and verified** (the M2/M3 procedure, `kvlab/perf_harness.py`
  `page_cache_before`):
  - before every C2 request, untimed, the runner computes the `mincore` resident fraction of
    every chunk file in that instance's disk tier;
  - if any fraction is below 1.0, it warms those files, records `prewarmed_from_fraction`, and
    re-measures;
  - `resident_fraction_before` must be 1.0 for every C2 request (validator; otherwise exit 3);
  - prewarm events are counted and reported.

  The page cache is extra host DRAM, not counted in H or D.

## 7. Timed region and metrics
**Timed region** (D22, unchanged definition):
- **Start:** `torch.cuda.synchronize()` immediately before the configuration's `prefill()`
  call.
- **End:** when the first token id is a host integer. The argmax runs on the device, and its
  `.item()` is the end. The full logits are copied to the host only after the end, for the
  checks. The timed path must not use `greedy_continue()`, which copies the logits to the CPU
  before its argmax.
- **Inside the region:**
  - C0: the full prefill;
  - C1: lookup, gather, the D4 constructor copy, suffix prefill and admission (device copies);
  - C2: restore, suffix prefill and admission (D2H, spills, disk writes and evictions).
- **Outside the region:**
  - building the token list;
  - before the start: `reset_peak_memory_stats()`, `memory_allocated()` (A0), and the persistent
    occupancy of every configuration (both C1 caches' retained bytes);
  - after the end: `max_memory_allocated()` (P);
  - resource readouts, records and decode steps;
  - all checks.
- **No synchronized phase instrumentation** runs in timed sessions.

**Primary metric:**
- For each session s, trace and configuration: the trace-mean first-token latency over all
  requests of the trace.
- For each comparison A − B: D_s = mean_A,s − mean_B,s. With per-request pairing, this equals
  the trace-mean of the per-request paired differences.

**Primary comparisons** (15, fixed in advance; d < 0 means the first-named configuration is
faster):
- T1, T2 and T3, for each C2 backend b ∈ {base, batch}:
  - C2-b − C0;
  - C2-b − C1-PA(W/2).

  That gives 12 comparisons.
- T0: C2-base − C0, C2-batch − C0 and C1-PA(W/2) − C0. That gives 3 comparisons.

**Secondary (reported, not part of the verdict family):**
- C2-b − C1-PA(W), against the GPU-resident ceiling;
- C2-batch − C2-base;
- C1-PA(W/2) − C0 on T1–T3;
- the steady-state versions of all comparisons;
- per-session medians and p90, then the median and range across sessions;
- per-request paired d, with M3's stratified block bootstrap (strata = sessions; b ∈ {1, 2, 4,
  8}), as a sensitivity analysis only;
- restored and recomputed tokens;
- hits by tier;
- **logical copy traffic** (bytes moved per request, counted from the code paths; not profiled,
  and not storage). Here n_r is the number of restored chunks (L = n_r × C), n_s the number of
  admitted chunks and P the chunk payload:
  - C1: gather D2D = n_r × P; D4 constructor D2D = n_r × P; admission D2D = n_s × P;
  - C2, both backends:
    - H2D = n_r × P;
    - D4 constructor D2D = n_r × P;
    - admission D2H = n_s × P;
    - disk file bytes read, for disk hits, into the host staging buffer;
    - disk file bytes written, for host spills;
  - C2-batch only: placement D2D = n_r × P, from staging to destination. Baseline has no
    explicit placement copy. Any per-copy temporaries PyTorch makes inside a copy are not
    counted;
- **allocated or live storage** (bytes held, not moved), reported separately from the
  traffic:
  - persistent: C1 retained entries (`retained_bytes()`); C2's pinned host slab, host logical
    occupancy, and disk used and peak;
  - transient per request: C1's gather scratch (n_r × P); C2's device destination
    (`device_destination_bytes`, the planned chunks × P) and batched staging
    (`device_scratch_bytes`, min(n, 4) × P);
  - the request KV after prefill (N × per-token bytes);
  - page-cache observations: the mincore resident fraction (§6). It is a point-in-time
    observation, not an accounted budget;
- disk evictions;
- **device memory per request:**
  - the process-global peak P, labelled as such. It includes both C1 caches and every
    co-resident structure;
  - **the incremental process-allocation peak above A0, P − A0.** It covers everything the
    process allocated during the request, including persistent cache growth: C1 entries
    admitted inside the timed region stay allocated. It is therefore not purely temporary
    overhead. The persistent growth during the request is recorded separately:
    - C1 retained bytes after the request minus before;
    - for C0 and C2 it is 0 on the device;
  - A0 and every configuration's persistent occupancy, recorded alongside;
  - P is a whole-process figure. It is never reported as a per-configuration GPU peak.
- **slow samples** (descriptive only):
  - a request sample is slow if its latency exceeds 1.3 × the minimum of the same (trace,
    request id, configuration) over the 10 sessions. The work is identical across sessions,
    because the decisions are deterministic;
  - regimes are maximal chronological runs of slow samples within a session, per
    configuration;
  - M2.1's fixed-case minimum is not used, because trace requests differ in length and cache
    state.

**Not reported:** p99 and tail latency. ~50–64 requests per trace cannot support them.

## 8. Statistics and correctness
**Unit of analysis: the session.**
- Requests within a trace are neither independent nor exchangeable:
  - their lengths and cache states differ by design;
  - each request's state depends on all earlier requests;
  - slow-mode regimes cluster in time.
- The estimand is conditional on the fixed evaluation trace (one seed per trace kind): "for this
  trace, the mean per-request latency difference". It is **not** a population of traces.
- Fresh processes are the replicates. The session-level D_s already averages over the
  request structure.
- **Independence is an assumption, not established.** The sessions run back to back in one
  batch, and machine, driver or thermal regimes can persist across process boundaries. Every
  interval and verdict below is conditional on treating the D_s as independent.

**Verdict per primary comparison:**
- **"Faster" or "slower"** requires both of the following:
  1. **stability (unanimity):** all R = 10 D_s have the same sign. This is an observed
     stability rule, as M3's R2 was. No probability bound is attached to it: a mean-zero
     effect does not imply equally likely signs;
  2. **mean test:** the two-sided t-interval for the mean of D_s, at level 1 − 0.05/15
     (Bonferroni over the 15 primary comparisons, 9 degrees of freedom), excludes 0 in the same
     direction. It assumes independent, approximately normal D_s. With R = 10 that assumption
     cannot be checked meaningfully, and it is stated with every verdict.
- **Otherwise: "inconclusive".** There is no equivalence claim.
- **Effect:** the mean of D_s, with its 95% t-interval (descriptive, same assumptions), the
  Bonferroni-level interval, the median and the range.
- **Ratios:** the ratio of trace-means is reported separately, and never mixed with D.

**Departure from M3, stated:**
- **Not reused for the verdict:** M3's verdict used block-bootstrap intervals over
  *repetitions of one fixed (L, S)*, which are exchangeable within a stratum, together with the
  span rule R3. Trace requests do not have that property. Resampling them would answer a
  question about a population of request positions that the fixed trace does not sample.
- **Reused:**
  - the freeze/manifest, the fail-closed validator and exit codes 0 / 3 / 4;
  - the deadline and the stopping rule (D31);
  - interleaving;
  - the "all sessions agree" stability idea (R2);
  - the block bootstrap, as a sensitivity analysis only.
- **No R3 analog:** interleaving per request plus 10 fresh-process sessions replaces it.
  Regime durations are still reported.

**Evidence validity, recomputed by the validator.** An unintended infrastructure or cache
fault makes the performance evidence invalid (exit 3, no verdicts), on **every** trace,
including T0. Expected capacity behaviour is not a fault: C1 evictions, C1-PA prefix-capacity
refusals, C2 host spills and T0's disk evictions are expected, but their counts must equal the
key-level predictions. Required:
1. **C2 faults:**
   - `spill_failures` is empty;
   - `write_skips` is empty;
   - every miss reason is `absent` (no CRC, identity, I/O or staging read failure);
   - undeletable bytes, temp bytes and unlink/evict `OSError`s are all 0 after every request.
2. **C2 placement:** per request, host hits, disk hits, spills and disk evictions equal a
   key-level model of the two LRU tiers (spill on host eviction, no promotion, D15). That
   gives 0 disk evictions on T1–T3, and T0's count as predicted.
3. **C1 faults:**
   - `write_skips_no_capacity` is 0;
   - prefix-capacity refusals and evictions equal the key-level C1-PA model.
4. **Page cache:** `resident_fraction_before` = 1.0 for every C2 request (§6).
5. **Process:** complete records (D31 semantics), and no caught OOM.
   - **Exits explained:** every session's exit must be explained by its own recomputed result.
     The runner exits 4 exactly when it recorded a strict mismatch, and 0 otherwise. An
     uncaught exception exits 1 and writes no report. Any other exit, or an exit that disagrees
     with the recomputed comparison, is invalid evidence (exit 3), even when the batch record
     has a consistent shape.
6. **Schema:** every field of every record is type- and range-checked before it is used:
   non-negative integers below 2^62, finite times, hex digests, and the exact traffic keys.
   Malformed evidence is invalid (exit 3). It is never a crash or a correctness failure.
   - Records that fail the schema never reach a semantic check or the coverage aggregation.
   - A last-resort, fail-closed backstop reports any record the validator still cannot
     interpret as malformed (exit 3, with a report). It can never make evidence valid.
7. **Reconciliation:** cumulative counters must equal values recomputed from the models. These
   cover:
   - hits by tier, stored chunks, and H2D, D2H and disk bytes;
   - lookups, full recomputes and restored requests;
   - misses and the `absent` miss reason, counted per request whenever the restored prefix is
     shorter than the eligible one;
   - host evictions, which equal the spills in fault-free execution;
   - host and disk occupancy against the entries, the slab size, registration (CUDA), the
   restore backend and restore storage, and C1 retained/peak bytes.
   - When the spec requires them (`M4_EVAL_V1`): a CUDA device and the expected chunk payload.
   - One model namespace across sessions.

**Correctness, recomputed by the validator** (in complete, valid evidence, any failure → exit
4; malformed or missing → exit 3):
1. **Strict outputs:**
   - each configuration's first-token logits and token for every request must be bitwise equal
     to a `ResidentReference` replay of **that configuration's own** restore lengths and
     stored chunks (D12);
   - C0 is compared with the empty-cache reference;
   - this runs untimed, right after each trace's timed replay.
2. **Decisions:**
   - C1-PA restore lengths and stored chunks must equal the key-level model;
   - C2's restore length must equal the oracle on T1–T3 (given the validity checks above);
   - every L must satisfy the first-token rule.
3. **Ownership and budgets:**
   - C1: retained bytes = entries × payload ≤ G after every request; prefix closure holds;
   - C2: host and disk budgets are never exceeded; disk evictions are 0 on T1–T3.
4. **Forward calls:** exactly one prefill forward per request (`steps = 1`).
5. **Cross-configuration outputs** are **not** compared bitwise (D10). Agreement of first
   tokens is reported descriptively only.

## 9. Bounded GPU correctness gate (M4-B1, before any timed session)
**Run once, on calibration seeds only, with no timing analysed. Cap: 15 minutes of GPU wall time,
enforced as one monotonic deadline by the bounded, fail-fast wrapper `scripts/m4_gate.py`.**
- **Order:** the wrapper runs the three suites below in order, each with `pytest -x`.
- **Stops:** it stops at the first failing suite (nothing further runs), at the deadline, or on a
  stop signal. On a deadline or a signal, the running suite's process group is terminated and
  reaped, and a FAIL record is written in every case.
- **Ownership:** the wrapper owns the suite's process group, not only its leader. After every
  suite, the group is checked and any remaining descendant is terminated. A suite that leaves
  processes running is a FAIL (`leftover_processes`). The caller's signal mask is restored.
- **Record:** `gate_record.json` holds the commands, exit codes and per-suite and total elapsed
  times.

1. **Existing suites:** `pytest -m "gpu and m1"` (both backends) and `pytest -m "gpu and m3"`.
   This checks C2's paths at the frozen source.
2. **New `tests/gpu/test_m4_replay.py`** (§11), on the real model:
   - **What runs:**
     - the first 12 requests of each calibration trace;
     - the CPU probe's oversized and shared-root sequences, scaled to C = 256.
   - **Configurations:** C0, C1-LRU(W/2), C1-PA(W), C1-PA(W/2), C2-base and C2-batch. The §4 budget rule
     is applied to each truncated trace's own W.
   - **Checks, with `steps = 8`:**
     - strict bitwise equality with `ResidentReference` of each configuration's own decisions;
     - **restored KV bytes (D12):** on every C1 hit, the restored K/V bytes equal the stored
       entry bytes and the reference's stored chunks. C2 keeps its existing byte checks;
       - the entry comparison is made at the instant `restore()` returns, before the suffix
         prefill and admission;
       - it covers exact lengths, chunk order, recomputed identities, and no storage shared
         with an entry;
     - decisions, tier placement and counters equal to the key-level models, with all fault
       counters 0:
       - for C1: every counter, at the restore boundary and after admission;
       - for C2: every engine counter, host and disk evictions, and occupancy in entries and
         bytes, after every request;
       - expected values come only from the models, the trace and the chunk format;
     - prefix closure;
     - retained bytes = entries × payload ≤ G;
     - no overlap between CUDA storage ranges of the request KV and the C1 entries;
     - C2 budgets.
   - **Recorded as diagnostics only:** per-request `max_memory_allocated`, to quantify the
     transient D4/gather copies.
     - **Where:** written as structured JSON to `<gate output>/m4_diagnostics/<case>.json`,
       outside pytest's temporary directories.
     - **Contents:** A0, P and P − A0 as process-level allocator figures (never a
       per-configuration peak), the logical storage terms, observed and expected counters, units
       and observation boundaries.
     - A missing or incomplete file fails the gate.
3. **Pass criterion:** every item passes, with no CUDA error.
4. **On failure:** the gate stops. It is reported as FAIL, and no performance session runs. A
   fix is reviewed, and the gate is rerun into a new output directory; the failed run is kept.
5. **Time projection:**
   - The gate records the wall time of its replay as a **feasibility diagnostic**, not as
     performance evidence.
   - If it projects the evaluation beyond 80% of the deadline (§10), the batch does not start
     until a recorded decision resolves it. R, the traces and the deadline are never adjusted
     automatically.
   - For M4_EVAL_V1, the gate's own wall time was supplemented by the one D34 timing pilot; the
     decision on that basis is D35. Nothing was adjusted.

The M0 strict suite (known `nat_long` FAIL) is not part of the gate. It stays as recorded.

## 10. Execution budget, stopping and failure handling
- **Attempts:** exactly one evaluation batch of R = 10 sessions. It is launched only after the
  gate passes and the protocol is frozen.
- **Deadline:** 2,700 s monotonic for the batch, rechecked before every launch. The planning
  estimate is ≈ 3 min per session, which includes untimed reference checks; it is not a
  measurement.
- **Stopping rule (as D31):**
  - any nonzero session exit stops further launches;
  - there are no retries, replacements, extensions or added sessions.
- **Outcomes:**
  - valid, complete evidence → exit 0, verdicts reported, including all negative and
    inconclusive results;
  - a correctness failure in complete evidence → exit 4, with no performance analysis;
  - missing, inconsistent or timed-out evidence, or a caught OOM → exit 3, with no statement
    either way.
- **Rerun:** a rerun needs a new spec name, a documented reason and a new freeze. It never
  replaces or re-analyses part of a recorded batch.
- **Exclusivity:** GPU work runs exclusively, never concurrently with any other GPU job.
  - WSL2's `nvidia-smi` cannot list Windows host processes, so exclusivity is recorded as
    "no competing process observable", not as proven.
  - The preflight records GPU memory and utilization, and the runner records them during the
    batch, as in M3.

## 11. Implementation
**Implemented and tested on CPU** (tiny random bf16 Qwen2, C = 4, and key-level checks at
C = 256; `docs/correctness-matrix.md`, M4-B section):
1. **`kvlab/prefix_aware_cache.py`:** `PrefixAwareGPUPrefixCache`, a subclass of the unchanged
   `GPUPrefixCache`.
   - Admission: prefix closure, LRU among unprotected leaves, and a stop when no victim exists.
   - Protection: the request's own chunks are protected for the duration of `store()` (the
     lifecycle is documented in the module).
   - Invariants are rechecked after every store.
2. **`kvlab/trace_models.py`:** independent key-level models, written with different data
   structures from the classes:
   - `pa_model` (C1-PA);
   - `lru_model` (C1-LRU, with dead entries; offline diagnostic only);
   - `tiered_model` (C2: host LRU with spill, disk LRU, no promotion);
   - `oracle_lengths`.
3. **`kvlab/m4_spec.py`:** the spec `M4_EVAL_V1`, holding:
   - traces and seeds, with their roles;
   - the budget rule, the configurations and the 15-comparison family;
   - R = 10 sessions and the 2,700 s deadline;
   - the seeded trace order and per-request configuration order;
   - the exact session commands.
4. **`kvlab/m4_bench.py`:** the session driver (§6–§7), covering:
   - the warmup, the interleaved timed replay and the untimed strict check;
   - per request: residency, logical copy traffic, storage terms, allocator observations
     (CUDA only), and C1 growth and fault counters.

   Any exception propagates. **`scripts/m4_measure.py`** runs one GPU session; formal sessions run
   only inside the batch. The D34 pilot used a private local mirror of it with an isolated pilot
   spec.
5. **`kvlab/m4_validate.py`:** the fail-closed validator (§8, §10).
   - It runs a schema pass before any semantic check, reconciles counters and storage, and
     reconciles each session's exit with its recomputed strict result.
   - It reuses `kvlab.m3_confirm.batch_problems` for the deadline, stopping and contradiction
     rules.
6. **`kvlab/m4_stats.py` and `kvlab/m4_analysis.py`:**
   - session-level comparisons: unanimity plus the Bonferroni-level t-interval (closed-form
     t distribution, checked against numeric integration);
   - secondary results, the steady state, the M3 block-bootstrap sensitivity analysis, and
     resources.

   **`scripts/m4_analyze.py`:** exit codes 0 / 3 / 4. Without a freeze, its output is labelled
   UNFROZEN.
7. **`kvlab/m4_freeze.py` and `scripts/m4_eval.py`:** freeze, verify and run.
   - `run` reuses `kvlab.m3_confirm.run_batch` unchanged.
   - The freeze refuses a protocol whose first line says DRAFT.
8. **`tests/gpu/test_m4_replay.py` and `scripts/m4_gate.py`:** the gate (§9) and its bounded,
   fail-fast wrapper.
   - The wrapper is tested on CPU with dummy suites (pass, fail-fast, deadline) and with real
     pytest children.
   - The gate's check helpers and per-case driver are tested on CPU with a tiny model,
     including targeted counter and byte mismatches (`tests/test_m4_gate_checks_cpu.py`).
   - GPU executions are listed in the provenance note at the top.

**The role of calibration (fixed):**
- The calibration seeds feed only the GPU correctness gate and the CPU tests. The one exception
  is the D34 feasibility pilot.
- No timing from them is analysed, except in that pilot, which was used solely for the §9.5
  runtime-feasibility decision.
- Nothing measured on them, or on the evaluation sessions, selects a policy, metric, trace,
  budget, threshold, comparison or session count.

**Before the evaluation:** a passing gate run of the current gate version (attempt 3), then the
protocol freeze (D35) from the commit containing this protocol, then the one batch. All three
took place; see the provenance note at the top.

**Not implemented, by decision:** runtime restore-length policy, cost-model refit,
asynchronous restoration, and any additional benchmark arm.

## 12. Claims and their limits
**Allowed form:**
> On synthetic trace Tk (seed 4200k), at a GPU KV allowance of G bytes for C1, C2-b with H
> pinned-host bytes plus D disk bytes (warm page cache) changed the trace-mean synchronous
> first-token latency by X ms [interval] relative to C1-PA(G), in all 10 sessions.

**Excluded claims:**
- "the same memory budget";
- service TTFT, throughput, concurrency or production benefit;
- other traces, models or GPUs;
- cold NVMe (D26);
- p99 or tail latency;
- "better than GPU caching" in general.

**Labels:**
- "conditional on an interleaved, co-resident experiment": both C1 caches stay resident while
  every configuration runs;
- "artificial GPU cap" for every G below the device's free memory;
- "C1 is this project's GPU-only baseline", not vLLM or SGLang;
- C1-LRU is offline only.

**Resource table:** it is reported per configuration and never summed:
- persistent GPU KV;
- request KV;
- restore scratch;
- the measured per-request device peak;
- the pinned slab;
- host logical occupancy;
- disk used and peak;
- the page cache, disclosed as uncounted.

**Unchanged:** D4, D10, the M0 `nat_long` strict FAIL, VAL-005 and VAL-006, M3's
`supports_adoption = false`, and the `baseline` default.

## 13. Out of scope, and what would justify it later
**Runtime restore-length policy** (restoring less than L* and recomputing more). Building one
would need:
1. evidence from M4 itself that the restore path is a material share of trace latency on the
   chosen backend;
2. a per-request headroom estimate on **calibration** traces: the established saving of an
   oracle choice over always-restore, with an interval excluding 0;
3. a headroom of at least a threshold, for example 5% of the trace-mean, fixed before that
   calibration;
4. a separate, declared experiment.

**Cost-model refit:** the M2 model describes the baseline backend and stays as recorded. A refit
is justified only by a declared policy experiment, only on the backend that the policy would use, and
only with fresh calibration sessions, never with M4 evaluation data.
