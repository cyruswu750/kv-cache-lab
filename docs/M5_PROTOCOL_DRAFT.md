# M5 calibration and evaluation protocol — executed once (stages A–F, 2026-11-05)

**Retrospective status and results update.** This header, the stage table (§2), the observed counts (§3) and
§11 were updated after stage F. The file keeps its historical name.

**What is unchanged:**
- the cases, prompts, seeds, schedules, state reset, timing boundaries, validity and failure rules, fit and
  statistics (§3–§10), as they were declared before measurement;
- the bound specification digests (calibration `ab1a506e…`, evaluation `5e6ecdc4…`).

**Approval:** this document was never frozen or approved as a whole, the way the M4 protocols were. Each stage
was authorized separately. Before each formal batch, its execution conditions were declared, and before the
evaluation batch they were recorded together with digests of the frozen inputs (the specification, cost model,
predictions, source identity and analysis implementation). That record is kept privately and is not part of the
repository. **This documentation update was written after the measurements and did not exist before them.**

**Status, by stage** (each executed once, with no retry or replacement):
- **A. GPU correctness gate** (source `9785c33`): PASS, 47 tests (M1 23, M3 17, M5 7). It covers correctness
  only.
- **B. Diagnostic pilot** (calibration session s01 only): it completed all 1,134 samples and passed the
  session-level validator.
  - **It is diagnostic only**: excluded from fitting and from every formal result, and its timings are not
    performance results.
  - **Its data:** never reused.
- **C. Formal calibration batch `M5_CAL_V1`** (source `a3c6341`): succeeded, with full validation passed. 3
  sessions, 3,402 samples, ≈ 773.7 s.
- **D. Fit and frozen predictions `M5_FIT_V1`** (CPU): exit 0.
  - **Model** `kvlab-restore-cost-v1`, digest `f9fe6543…`.
  - **Predictions:** digest `e0d5c335…`, for all 47 evaluation candidates, frozen before stage E.
- **E. Held-out evaluation batch `M5_EVAL_V1`** (source `a3c6341`): succeeded, with full validation passed. 4
  sessions, 3,408 samples, ≈ 660.1 s.
- **F. Analysis** (CPU): exit 0.

**Executable code:** byte-identical across stages A–F. Between A (`9785c33`) and C (`a3c6341`), only
documentation and the ignore rule of §11 item 6 changed.

**The outcome** (details, intervals and limits: `docs/results.md`, M5):
- on all 8 evaluation cases, CostAware chose the same restore length as AlwaysRestore;
- an incremental benefit of CostAware over AlwaysRestore was **not demonstrated**.

**Raw evidence:** the output directories of §2 are published byte for byte under `results/m5/`, with
`results/m5/SHA256SUMS` (decision D44). The offline reproduction is described in `docs/results.md`, M5.

Design: `docs/M5_DESIGN.md`. Decisions: D40–D43.

## 1. Questions
On the declared grid, on this machine and stack (batch 1, synchronous first-token latency, a warm page cache):
1. How well does a model fitted on calibration cases predict each held-out candidate's mean latency?
2. How good is each policy's choice? Measured against the FixedChunks latencies of its selected candidate:
   CostAware, AlwaysRestore and Recompute.
3. What latency does each policy deliver, its own planning and prediction included?

No speedup, interior optimum, tail benefit or agreement threshold is required. A degenerate or unfavourable
result is a valid outcome.

## 2. Stages, commands and deadlines (each executed once)

| Stage | Command (from the repository root, `HF_HOME=<weights> HF_HUB_OFFLINE=1`) | Deadline | Executed |
|---|---|---|---|
| A. GPU correctness gate | `.venv/bin/python scripts/m5_gate.py --out-dir local/m5_gate_<ts>` | 900 s (M1 + M3 + M5 GPU suites) | once: PASS, 106.2 s |
| B. Calibration-only feasibility pilot | the unchanged s01 `session` command of `M5_CAL_V1`, run through `run_batch` in a separate ignored run directory; never used for fitting or evaluation | 900 s outer budget | once: session s01, 262.4 s; diagnostic only |
| C. Calibration batch | `.venv/bin/python scripts/m5_run.py batch --split calibration --run-dir results/m5/M5_CAL_V1_run` | 2,400 s | once: succeeded, 773.7 s; valid |
| D. Fit and freeze (CPU) | `.venv/bin/python scripts/m5_fit.py --cal-run-dir results/m5/M5_CAL_V1_run --out-dir results/m5/M5_FIT_V1` | — | once: exit 0 |
| E. Evaluation batch | `.venv/bin/python scripts/m5_run.py batch --split evaluation --run-dir results/m5/M5_EVAL_V1_run --cost-model results/m5/M5_FIT_V1/cost_model.json --predictions results/m5/M5_FIT_V1/predictions.json` | 2,400 s | once: succeeded, 660.1 s; valid |
| F. Analysis (CPU) | `.venv/bin/python scripts/m5_analyze.py --eval-run-dir results/m5/M5_EVAL_V1_run --fit-dir results/m5/M5_FIT_V1 --out-dir results/m5/M5_EVAL_V1_analysis` | — | once: exit 0 |

**`batch` refuses to start** unless:
- the source tree is clean: an empty `git status --porcelain`, in which the git-ignored output directories of
  this table do not appear (§11 item 6);
- the run directory is new;
- for evaluation, the predictions' digest and their bindings to the artifact and to both spec digests hold.

**What it then does:**
- writes `spec.json` and `batch_plan.json`;
- runs the planned sessions through `kvlab.m3_confirm.run_batch`: one deadline from the first launch, a stop
  on the first nonzero exit, no retries;
- the record's `freeze_manifest_sha256` field holds the batch-plan digest (a name inherited from M3). No
  M3/M4 freeze machinery is used or repeated.

## 3. The specification (`kvlab/m5_spec.py`; `scripts/m5_run.py plan --split …` prints it and its digest)
**Configuration:**
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct@989aa79…`, bf16, SDPA, chunk C = 256.
- **Backend:** the explicitly selected performance configuration is **`batched` with a staging group of 4**.
  Both backends are covered by the correctness gate. The default backend stays `baseline`.
- **Prompts:** seeded uniform token ids in [10, 150000), with one seed per case; calibration and evaluation
  seeds differ.

**Cases.** A case declares the state of every complete prompt chunk (host, disk or absent). In this grid it is
an indexed prefix followed by absent chunks. Derived from that declaration:
- **LRU orders:** the host and disk LRU orders are the index order;
- **capacities:** host slots = host chunks + admissions (no admission can spill); disk files = disk chunks;
- **admissions:** exactly the absent complete chunks;
- **candidates:** every feasible candidate;
- **provenance:** one full prefill of `prompt[:P]`, where P covers every indexed chunk.

| Calibration | N | Prefix | Cand. | m | Evaluation | N | Prefix | Cand. | m |
|---|---:|---|---:|---:|---|---:|---|---:|---:|
| cal-01 | 300 | H | 2 | 0 | ev-01 | 400 | D | 2 | 0 |
| cal-02 | 600 | DD | 3 | 0 | ev-02 | 700 | HH | 3 | 0 |
| cal-03 | 1100 | HDH | 4 | 1 | ev-03 | 1250 | DHD | 4 | 1 |
| cal-04 | 1500 | DH | 3 | 3 | ev-04 | 1700 | HD | 3 | 4 |
| cal-05 | 2100 | HHDD | 5 | 4 | ev-05 | 2400 | HHHDD | 6 | 4 |
| cal-06 | 900 | — | 1 | 3 | ev-06 | 3000 | D×4 H×6 | 11 | 1 |
| cal-07 | 1800 | D×5 | 6 | 2 | ev-07 | 3850 | H×10 D×3 | 14 | 2 |
| cal-08 | 3500 | H×12 | 13 | 1 | ev-08 | 1000 | DDD | 4 | 0 |
| cal-09 | 3700 | D×12 | 13 | 2 | | | | | |
| cal-10 | 4000 | D×6 H×8 | 15 | 1 | | | | | |
| cal-11 | 2800 | H×5 D×5 | 11 | 0 | | | | | |
| cal-12 | 1300 | DHHH | 5 | 1 | | | | | |

**Coverage:**
- short prefixes (1–2 chunks);
- long suffixes (S up to 3,744 tokens; S > C for the L = C candidate of every case with N > 2C);
- host-only, disk-only and mixed prefixes in both orders;
- 12 distinct N values, and m from 0 to 4.

**Design check (CPU, before any measurement).** The 81 calibration rows have rank 9 of 9 and a condition
number of 28.4, with the fit's column scaling and tolerance. A synthetic fit on this design records exactly the
declared domain. That domain is N ∈ [300, 4000], S ∈ [44, 3744] for L > 0, at most 14 restore chunks (12 host,
12 disk) and m ∈ [0, 4]. Every evaluation case is inside both the decision and the prediction domain. A case
outside would be a build error: it is never dropped or repaired after timings.

**Holdout.**
- **Separation:** calibration and evaluation differ by configuration, not by id. A configuration signature
  excludes the id and the prompt seed, and a duplicate configuration or prompt across the split is refused.
- **What it tests:** generalization to new request/state configurations (new N, tier layouts and admission
  counts) with new token content, inside the calibrated box, on the same machine, model, stack, backend and
  group.
- **What it does not test:** other workloads or content distributions, other machines, cold storage,
  concurrency or trace-level cache dynamics.

**Fit weighting.** `uniform` (declared before calibration): one row per (case, candidate) mean, so a case with
k+1 candidates contributes k+1 rows. cal-10 is 15 of 81 rows; cal-06 is 1. The aggregate results weight
**evaluation cases equally**.

**Schedule:**
- **Calibration:** 3 sessions, each running 2 warmup and 12 measured repetitions of every candidate (FixedChunks
  only, no policy runs, because no calibrated model exists yet).
- **Evaluation:** 4 sessions, each running 2 warmup and 10 measured repetitions of every candidate plus
  CostAware, AlwaysRestore and Recompute.
- **Order:** the case order is seeded per session, and the run order is seeded per (case, phase, repetition),
  with seeds derived by SHA-256.

**Counts and estimates:**

| Batch | Runs per repetition | Samples per session | Measured, total | Samples, total | Estimated batch time |
|---|---:|---:|---:|---:|---:|
| Calibration | 81 | 1,134 | 2,916 | 3,402 | ≈ 1,390 s |
| Evaluation | 71 | 852 | 2,840 | 3,408 | ≈ 1,410 s |

The estimates assume 0.35 s per sample, 3 s per case setup and 30 s of model load per session.

**Observed** (batch record, first launch to last exit; added retrospectively):

| Batch | Elapsed | Samples |
|---|---:|---|
| calibration | 773.7 s | 3,402 = 486 warmup + 2,916 measured |
| evaluation | 660.1 s | 3,408 = 568 warmup + 2,840 measured |

Both were within the 2,400 s deadline. The estimates above were never validated, and the deadline was the only
binding limit.

**Pilot observation.** The stage-B session s01 took 262.4 s from process launch to exit, about 0.22 s per sample
overall. Three sessions extrapolate to about 790 s. That is an estimate from a single session, not a confidence
interval: session-to-session and thermal variation are unknown from one session.

**Calibration conditions (declared before the formal batch):**
- **Start:** no unrelated GPU workload; the current device settings (no clock locking, power or fan changes, and
  no cooling intervals between sessions).
- **Sessions:** the three run consecutively through the existing batch runner, with the unchanged spec and the
  2,400 s deadline.
- **Telemetry:** an external, read-only `nvidia-smi` monitor covers the whole batch, session transitions included.
  It runs outside the benchmark processes and the timed region, and keeps timestamps for alignment. Its records
  are stored outside the formal run directory. Telemetry is context only; it is never a sample-exclusion rule.
- **Samples:** slow and throttled samples are all kept. No retries, replacement sessions or filtering.
- **Pilot data:** never reused.

**Evaluation conditions** (recorded before the evaluation batch, as described in the header):
- the same device, monitor, sample and failure rules as calibration, and the strict drop_pagecache rule (§11
  item 2);
- the four sessions run consecutively, with no cooling intervals and no evaluation pilot;
- the ≈ 1,410 s estimate is unvalidated, and the 2,400 s deadline is authoritative;
- the frozen choices were kept as they were. The model and grid were not changed after the predictions were
  frozen, even though those predictions made CostAware choose the maximal length on every case.

**Resources per case (planning):**
- a pinned host slab of at most 13 chunks (≈ 95 MB);
- an unpinned source bank of at most 14 chunks;
- at most 12 disk files (≈ 88 MB);
- resident reference chunks on the GPU (≤ 14 × 7.3 MB);
- the restore destination and staging buffer.

## 4. State preparation and reset (`kvlab/m5_state.py`)
**Bank.** One bank per case and session (`ChunkBank`): a full prefill of `prompt[:P]`, copied by the M1 store
path. It is the single provenance of the host slots, the disk files and the `ResidentReference`, which is seeded
from the bank's bytes and never recomputed at another length.

**Stores:** new stores in a new directory. Host chunks are inserted in LRU order (slots 0, 1, …) and disk files
are written in LRU order.

**Reset after every sample:**
1. discard the admissions in reverse allocation order (which restores the free list);
2. lease the host chunks in the declared order (host recency);
3. `put` the existing disk keys in the declared order (disk recency).

Admission cleanup alone is not a reset (a test shows the LRU difference). Reopening a `DiskStore` deletes files
and is never used. Counters are never reset or overwritten; samples record deltas.

**Fingerprint (before every sample).** It is compared with the setup observation; the spec-derivable part was
compared with `m5_spec.expected_structure` at setup. It covers:
- membership, LRU orders, host slots, the free list and capacities;
- no pins, no busy staging buffer, no temp or undeletable bytes;
- the directory listing, and each file's size, inode and mtime.

It proves an identical index, recency and slot state, and that the observed file metadata (inode, size,
mtime) is unchanged. It is not a byte-level proof and does not read payloads.

**Content.** The SHA-256 of every host slot and disk file is checked against the bank at setup and at the end
of the case. Every sample also byte-checks what it restored (§6).

## 5. Page cache
Before every sample, whatever the policy, `kvlab.warm_disk.prepare` runs over the disk files of the maximal
feasible prefix (the union of every candidate's reads):
- one observation pass, at most one prewarm per file, one final pass, no retry;
- then `complete` after the timed region.

**Required:** outcome `verified` (or `empty` without disk files), and `drop_pagecache` unchanged from before the
preparation to after the request.

**Limitations:** sequential, unpinned point observations (`docs/M4_EVAL_V2_PROTOCOL.md` limits apply). No
global cache drop or WSL tuning. No cold-device claim.

## 6. Samples and timing (`kvlab/m5_bench.py`)
**Order per sample:**
1. structure check;
2. attach the policy;
3. counters;
4. page-cache preparation;
5. **timed region:** synchronize, `t0 = perf_counter_ns()`, `engine.prefill_request(prompt)`,
   `int(argmax(logits))`, `t1`, with the collector disabled.

The timed region includes ingest, chain keys, planning, selection, validation, the decision record, restore,
prefill, admission and the first token. It contains no hooks, phase marks, extra synchronization, fingerprint
scans or reference work.

**After the timer (untimed):**
- the warm-disk completion and the counter deltas;
- the strict check of the sample's **actual** split:
  - the restored bytes against the bank;
  - the first-token logits, bitwise, against the bank-seeded reference of the same split (computed once per
    split and session);
- the release of outputs and the reset;
- the garbage collector, once per repetition.

**Runs:**
- **Candidate runs** use FixedChunks through the engine.
- **Policy runs** use the real CostAware (the frozen artifact), AlwaysRestore and Recompute.

**One prefill:** exactly one prefill per request is verified by the CPU tests and the GPU gate, not by a hook
inside timed samples.

**Failure handling:** any state, strict or programming error stops the session. The session file is still
written (status `failed`, with the error), and the exit is nonzero.

## 7. Records and identity
Every sample records:
- its identity (case, phase, repetition, position, run) and sequence number;
- the policy, the selected and actual lengths, the sources, the stored chunks and the miss reasons;
- the decision (reason, plan, objectives);
- the timing (t0, t1, latency, wall time);
- the fingerprint, the page-cache record, the counter delta and the strict check.

Every session records:
- the spec digest and the session seed;
- the namespace (fields and digest);
- the source identity, packages, model weight blobs and device;
- for evaluation, the cost-model and prediction digests.

## 8. Validity (`kvlab/m5_validate.py`, fail-closed)
**Recomputed from the spec and the recorded namespace:**
- **Files and batch:**
  - the expected files;
  - `spec.json`, the batch plan and the batch record, each of which must be a JSON object;
  - the batch plan (commands, bindings);
  - `m3_confirm.batch_problems` (launch order, exits, deadline, recorded session-file digests);
  - a fully successful batch: status `succeeded`, no stop, and every session exiting 0 by itself.
- **Sessions and cases:**
  - every session's status, spec digest and seed;
  - device and pinning (for real evidence);
  - a complete identity: a clean source tree with its head, diff digest and file hashes, the model weight
    files, the package versions and, for GPU evidence, the GPU;
  - the case order;
  - per case, the prepared structure, the fingerprints, the bank coverage and the content checks.
- **Samples:**
  - the exact sample identity sequence, warmups included;
  - the plan, the selected and actual lengths, the sources, the declared admissions, no miss and the decision
    reason;
  - CostAware's selection and objectives equal to the frozen predictions;
  - the exact counter deltas (hits, bytes, admissions; no spill, write skip or disk write);
  - integer timing;
  - the page-cache record: the files, the recomputed outcome and drop_pagecache, and its schema and bounds
    (size, page count, resident pages, prewarm status and bytes); the passes in order, the final pass before t0,
    and a gap equal to t0 minus that final pass;
  - the strict results.
- **Across sessions and files:**
  - equality across sessions of the namespace, source, weights, packages and device;
  - the predictions recomputed from the bound cost model for this spec, which must be identical, with their
    digest, their bindings and their creation before the evaluation launch (the same check runs before
    launch);
  - the artifact's provenance and case declarations.

**Exit codes:** any problem is exit 3 (invalid; no fit, no analysis). A strict mismatch in otherwise complete
evidence is exit 4. Injected faults belong to the correctness gate, never to performance cases. A truncated
performance sample is invalid evidence (tested).

## 9. Fit and predictions (`kvlab/m5_fit.py`)
**Fit:**
- **Inputs:** validated calibration evidence only. Each candidate's mean is the arithmetic mean of its measured
  (never warmup) FixedChunks latencies over all sessions.
- **Call:** `fit_restore_cost` with the evaluation ids declared, the `uniform` weighting, and the sessions'
  namespace, backend and group.
- **Provenance:** the calibration and evaluation spec digests, every evidence file's digest and the source
  identity.

**Predictions, frozen before any evaluation sample** (from the model as reloaded from disk). Per evaluation case:
- the CostAware decision on the case's plan;
- `predict_ms` for every candidate with the actual m. An out-of-domain prediction is recorded as unavailable,
  with its reason, never fabricated.

The predictions' digest covers the artifact digest and both spec digests.

## 10. Statistics (`kvlab/m5_analysis.py`)
**Means.** μ̂(c, L) is the mean of the measured candidate latencies of case c.

**Best measured candidate.** L̂* is the argmin over candidates, with exact ties going to the larger L. It is
never the fastest single sample.

**A. Candidate selection quality** (FixedChunks latencies of the selected candidate):
- **Agreement:** L_P = L̂*. A noisy point estimate.
- **Naive regret:** μ̂(L_P) − μ̂(L̂*). Descriptive and biased high.
- **Cross-evaluated difference:**
  - **definition:** choose the reference on partition A (sessions s01, s03) and evaluate both it and L_P on B
    (s02, s04); swap; combine with weights ½ and ½;
  - **reading:** signed, and negative values are kept. It is not unbiased regret against a true oracle;
  - **dependence:** sessions are fresh processes run back to back on one machine, so dependence between them
    is not excluded.

**B. Delivered policy latency.** The mean of the policy runs, and the paired per-repetition differences
CostAware − AlwaysRestore, CostAware − Recompute and AlwaysRestore − Recompute. This is the primary practical
report, whatever its direction.

**C. Prediction error.** `predict_ms` − μ̂ in ms and relative. Every candidate row has equal weight.

**Ambiguity (per case):**
- **Procedure:** a predeclared simultaneous procedure over the family of all pairwise candidate differences,
  using studentized-max bootstrap intervals at α = 0.05 with pairing preserved.
- **Possibly-best set:** the candidates that no other candidate beats with an interval entirely below zero.
  A case is ambiguous if the set has more than one member.
- **Scope:** approximate coverage within one case. There is no control across cases, and a winner is never
  first selected and then tested.

**Uncertainty:**
- **Resampling unit:** whole repetitions (pairing across candidates and policies is preserved).
- **Blocks:** circular moving blocks of 3 within each (case, session) stream; blocks never cross sessions.
- **Sessions:** fixed strata, which assumes sessions are representative and independent.
- **Cases:** a fixed grid (no case resampling, no claim beyond it).
- **Settings:** 2,000 replicates and 95% percentile intervals.
- **Adequacy:** "inadequate" (no interval) when a session has fewer than 2 blocks. Cross-evaluated intervals
  are also suppressed when a partition half has fewer than 2 sessions.
- **No filtering:** slow samples are kept.

**Limits on claims.** Planning-overhead diagnostics stay separate from latency claims. There is no p95, TTFT,
matched-memory, cold-NVMe or production claim.

## 11. Decisions before execution, and their outcome
The items below were open before execution. Each one now records how it was settled, added retrospectively.
1. **The grid, counts and deadlines** above. Whether to run the stage-B pilot first, given that the per-sample
   time is an estimate.
   - **Outcome:** the stage-B pilot ran first. The grid, counts and deadlines were then used unchanged.
2. **The drop_pagecache rule:**
   - **The rule:** a cache drop inside any sample's window, from its preparation to the end of its request,
     invalidates the batch.
   - **The risk:** WSL page-cache drops (increments of the `drop_pagecache` counter) occurred during earlier
     M4 work. Their future incidence during a batch is unknown. None occurred during the stage-B pilot.
   - **The alternative:** a per-sample rule with a predeclared replacement procedure.
   - **Status:** the strict rule is kept for calibration; no alternative is implemented.
   - **Outcome:** the strict rule was kept for both formal batches, and neither batch had a drop. The counter was
     constant within each batch: 54 throughout calibration and 55 throughout evaluation. It increased by one
     between consecutive batches, outside every sample window.
3. **GPU monitor:** the session code records none. For calibration, the external read-only monitor described in
   §3 is used as context.
   - **Outcome:** the same external monitor covered both formal batches. Its records are context only, and no
     sample was excluded or weighted by them.
4. **Evidence location** (results/m5/) and its archival. The four declared output directories are git-ignored
   during execution (item 6). Archiving them into Git later is a separate, explicit decision.
   - **Outcome:** all four directories are published byte for byte (decision D44). They were force-added by
     explicit path, and the ignore rules stay in place for new files.
5. **Sessions:** the number of sessions (3 and 4) limits interval precision, and whether that is acceptable.
   - **Outcome:** 3 and 4 sessions were used. The limited precision, and the dependence between back-to-back
     sessions, remain limitations of the results.
6. **Generated output directories and source cleanliness** (blocker found 2026-11-05 before launch; resolved by
   ignore rules; the batch was not run before the fix):
   - **The conflict:** `scripts/m5_run.py batch` writes `spec.json` and `batch_plan.json` into the formal run
     directory under `results/m5/` before the sessions start. When that directory was untracked and not ignored,
     every session would have recorded a non-empty `git status --porcelain`, and the validator requires an empty
     one. The evidence would have been rejected and no fit would have been possible.
   - **The resolution:** `.gitignore` ignores exactly the four declared output directories of stages C–F:
     `/results/m5/M5_CAL_V1_run/`, `/results/m5/M5_FIT_V1/`, `/results/m5/M5_EVAL_V1_run/` and
     `/results/m5/M5_EVAL_V1_analysis/`. Nothing else under `results/` is ignored by this rule, and no executable
     file lies inside these directories.
   - **What is unchanged:**
     - every source-cleanliness check: `batch`'s pre-launch check and the validator still require an empty
       `git status --porcelain`, so any tracked modification or any other non-ignored untracked file still
       blocks a launch and invalidates a session;
     - the source hashes;
     - the evidence-integrity checks: the expected file set of the run directory, the spec and plan bindings, and
       the batch record;
     - the benchmark, specification, runner, cost model and tests.

     This is an ignore and documentation change only, so the GPU correctness gate (stage A, which passed on the
     same executable code) was not repeated.
   - **Outcome:** this ignore commit (`a3c6341`) is the source of stages C–F. Every formal session recorded an
     empty `git status --porcelain`.

## 12. Out of scope
Trace-level replay with the policy, matched-resource or tail comparisons, cold NVMe, admission policy,
pipelining, concurrency, vLLM, CI and a default-backend change.
