# M5 next-stage protocol — DRAFT (not frozen, not approved)

**Status, 2026-11-05: DRAFT.** It states requirements for a future candidate-measurement harness,
calibration and held-out evaluation of the M5a policy (`docs/M5_DESIGN.md`). **GPU correctness,
calibration and evaluation are NOT RUN.** No harness implements this yet. Case lists, repetition counts,
seeds and the time cap are still open, and so is the exact uncertainty method (§8). They must be fixed,
reviewed and frozen before any evaluation measurement. Each GPU stage needs its own authorization.

## 1. Questions
For held-out request/state cases on this machine and stack (batch 1, synchronous first-token latency,
warm page cache):
1. How well does a model fitted on calibration cases predict each candidate's mean latency?
2. How often does `CostAware` choose the best measured candidate, and what is its mean regret against that
   candidate? The same is asked of `AlwaysRestore` and `Recompute`.
3. What do planning and selection cost, and what end-to-end latency does the policy actually deliver?

No speedup, interior optimum, tail benefit or accuracy threshold is required for success. A degenerate
result (always-restore is always best, or never distinguishably worse) is a valid outcome.

## 2. Stages (each separately authorized)
- **A. GPU correctness gate** (strict, no timing) on the real model, for both backends:
  - forced candidates, compared byte for byte with the source and against `ResidentReference` replays of the
    actual length and the actual stored chunks;
  - policy runs through `run()` and `prefill_request()`;
  - forward-call counts;
  - truncation under an injected invalid chunk.
- **B. Timing-only pilot**, if needed to set the budget. Its data is never used for fitting or evaluation.
- **C. Calibration:** fresh sessions on calibration cases only, then `fit_restore_cost`. The artifact and
  the predictions for every declared evaluation case are frozen (with digests) before any evaluation
  sample exists.
- **D. Protocol freeze, then one held-out evaluation batch**, on disjoint cases in fresh sessions.
- **E. Analysis** by a fail-closed validator and a predeclared analysis (§8–9).

## 3. Cases
- **A case** is one complete request/state:
  - the prompt (N tokens);
  - the indexed prefix: chunks, tiers and order;
  - the host and disk LRU orders and slot mapping;
  - the capacities;
  - the admitted-chunk count m, from states where admission does not spill;
  - the backend and staging group;
  - the provenance of the chunk bytes.
- **Candidates:** within a case, every legal candidate `0, C, …, k·C` is measured, each through
  `FixedChunks`. Every case lies inside the declared domain.
- **Bounded:** k ≤ 16, so at most 17 candidates, and a small number of cases under a declared time cap.
- **The split is by complete case**, never by individual L. Calibration and evaluation ids are declared
  before calibration. `fit_restore_cost` refuses evaluation ids.
- **Coverage, in both splits:**
  - short prefixes (1–2 chunks), where M2.1/M3 saw recompute competitive in the mean;
  - long suffixes (S ≫ C, up to several thousand tokens);
  - host-only and disk-only prefixes;
  - mixed tiers in both orders (disk-then-host arises naturally from LRU; host-then-disk is constructed);
  - several values of N, and m including 0 and positive values (identifiability, `docs/M5_DESIGN.md` §5).
- **No cherry-picking:** cases are chosen by rule before any measurement, not to make a policy win.

## 4. State preparation and reset (benchmark-only code)
- **Identical starting state.** Before every candidate sample of a case, the state must be identical:
  - chunk content and provenance;
  - tier placement and slot mapping;
  - host and disk LRU orders;
  - capacities and free slots.

  A state fingerprint is recorded and checked before each sample; a mismatch invalidates the evidence.
- **Benchmark-only mechanics.** Preparation and reset use benchmark code and public store operations, as
  `kvlab/perf_harness.py` `Bench.prepare` does:
  - copy chunk bytes from one source computation;
  - insert in the declared order;
  - re-establish recency by leasing host keys, and re-putting existing disk keys, in the declared order.

  Production stores are not changed to simplify measurement.
- **What does not reset state:**
  - constructing a `DiskStore` deletes the existing files of its directory; it is not a state-restoration
    mechanism;
  - admission cleanup (discarding newly stored chunks, as in `Bench._reset_admissions`) does not restore LRU
    order: restored host chunks were leased and disk hits moved to the end. The order must be re-established
    and verified separately.
- **Reference provenance.** The chunk bytes come from one recorded computation. Strict checks seed
  `ResidentReference` with the same provenance and replay the actual restore length and the actual stored
  chunks, outside timed samples.

## 5. Page cache
Before every sample of a case, whatever the candidate, one common declared preparation runs:
- every disk file of the maximal feasible prefix (the union of all candidates' reads) is warmed and
  verified with mincore;
- the outcome is recorded;
- it never depends on the cost model's choice.

This is a warm-page-cache experiment. No cold-device claim is made (D26), and page-cache memory is not
counted.

## 6. Timing
- **Per sample (D22 boundary):** synchronize, `t0`, `engine.prefill_request(prompt)` with the attached
  policy, `argmax(...).item()`, `t1`. The region therefore includes:
  - ingest and online planning;
  - selection and validation;
  - restore and suffix or full prefill;
  - admission.

  The collector is disabled inside the region.
- **Candidate runs** use `FixedChunks(k)`.
- **Policy runs** use the real `CostAware`, `AlwaysRestore` and `Recompute` engines on the same prepared
  states. They measure the latency each policy actually delivers, including its own planning and
  prediction.
- **Outside timed samples:** diagnostic instrumentation (phase marks, CUDA events, profiler) runs in
  separate samples, and correctness checks run after the timed region.
- **Overhead diagnostic:** repeated calls of `plan_restore` and `policy.choose` on the prepared state. They
  are side-effect free, which the fingerprint verifies. These are diagnostic timings, kept separate from any
  validated latency claim.
- **Recompute is not C0.** `Recompute` and the L = 0 candidate keep chain keys, lookup-free bookkeeping and
  admission. A no-cache baseline may be reported as context, never mixed into candidate or policy metrics.

## 7. Order and records
- **One repetition** of a case measures every candidate and every policy run once, in a seeded random order.
- **Records** are chronological. Each holds the case id, the repetition, its position in the repetition, the
  wall time, the policy, the selected and actual lengths, the sources, the outcomes of the state fingerprint
  and the residency check, and the latency.
- **Pairing** is by repetition.
- **Fresh data only:**
  - calibration and evaluation are new sessions;
  - M4 data has already been observed and analyzed;
  - M2/M3 data were measured under other protocols and backends, so they inform case selection only.

## 8. Metrics (to be fixed at freeze)
**Means.** μ̂(c, L) is the mean over the repetitions of case c for candidate L.

**Prediction error:**
- the definition: `predict_ms` (with m) minus μ̂(c, L), for every evaluation candidate;
- reported as signed and absolute errors in ms and relative, by regime (tier mix, suffix length).

**Best measured candidate:** L̂*(c) = argmin over L of μ̂(c, L), with exact ties going to the larger L.
- It is a **noisy point estimate**: the best among measured candidates, not a global optimum, and never
  the fastest single timing sample.

**Ambiguous cases:**
- **Definition:** a case is ambiguous when the paired difference between L̂*(c) and the runner-up, or the
  policy's choice, has an interval containing 0.
- **Method:** a dependence-aware resampling that resamples whole repetitions jointly across the case's
  candidates, so the pairing is kept.
- **Reporting:** ambiguous cases are counted and listed. Agreement is reported over all cases and over
  unambiguous cases.

**Agreement:** the fraction of evaluation cases where the policy's choice equals L̂*(c).

**Mean regret:** the mean over cases of μ̂(c, chosen) − μ̂(c, L̂*), in ms and relative.
- It is reported for `CostAware`, `AlwaysRestore` (the longest feasible prefix) and `Recompute` (L = 0)
  on the same cases.
- **Bias correction:** the minimum of noisy means is biased low, so naive regret is biased high and
  agreement biased in both directions. A split-sample estimate is reported next to the naive one: choose
  L̂* on one half of the repetitions and evaluate on the other, then swap.

**Uncertainty:**
- intervals keep the pairing and the time dependence (block resampling over chronological repetitions),
  and treat cases as units;
- the existing binary `perf_stats.selection_penalty` and `policy_totals` do not support multiple candidates
  and are not reused as if they did.

**Overhead and delivered latency:**
- the planning and selection time (median, p99, µs; diagnostic);
- each policy run's mean latency per case, and its consistency with μ̂(c, the chosen L).

## 9. Validity (fail-closed)
Evidence is invalid on any of:
- a state-fingerprint mismatch;
- residency below 1.0 after the declared preparation;
- a strict mismatch;
- an admission spill (the controlled-state assumption);
- an unintended miss or truncation in a candidate run;
- missing or out-of-order records;
- a model or prediction digest that does not match the frozen record;
- a batch-control problem (deadline, stop, retries).

Out-of-domain fallbacks are counted and reported, never dropped. Invalid evidence gets no performance
conclusion.

## 10. Out of scope
Trace-level replay with the policy, matched-resource or p95/p99 comparisons, cold NVMe, admission policy,
pipelining, concurrency, vLLM and a default-backend change.
