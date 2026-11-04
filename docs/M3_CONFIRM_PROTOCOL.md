# M3_CONFIRM_V1 protocol: preregistered confirmation of the batched restore backend (frozen 2026-11-04T01:22-08:00, before any measurement)

> **Status: DRAFT, pending premeasurement freeze.** No sample of this experiment has been measured.
> Before freezing, the reviewer replaces the first line's `(DRAFT pending premeasurement freeze)`
> with `(frozen <date>, before any measurement)`. `scripts/m3_confirm.py freeze` refuses while the
> first line contains `DRAFT`. After the freeze, this file must not change: the launch and the
> analysis both check its bytes against the freeze manifest.

This is a bounded confirmation experiment for the M3 batched restore backend (`docs/M3_DESIGN.md`,
decisions D30). The M3 evaluation (`docs/M3_PROTOCOL.md`, frozen and unchanged) found every
case faster with the batched backend. Its pre-registered adoption rule was still not met,
because (1024, 1) was inconclusive under the duration rule R3: the measured span was 73 s,
below 10 × 8.2 s (the longest slow regime). This experiment re-tests (1024, 1) and one fresh
case, with a longer measured stretch per case, under the unchanged M3 rules. The new evidence is
analysed on its own. It is never pooled with the M3 samples, and it does not overwrite the M3
result (`supports_adoption = false`).

## 1. Question
For each case and tier (host, warm disk), is the plain end-to-end first-token latency lower
with the batched backend than with the baseline backend, in both the mean and the median of the
paired difference, under rules R1–R3?

## 2. What is fixed (identical to M3, `docs/M3_PROTOCOL.md` §2)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, bf16, SDPA,
  C = 256. Staging group G = 4 (`restore_staging_chunks`).
- **Machine and stack:** the M3 machine and stack, required exactly (§8).
- **Timed region:** the unchanged M2 timed region, `Bench.measure`.
- **State:** store A holds the prefix in pinned host slots; store B holds it as warm files
  (page-cache resident, verified per sample).
- **Runner:** `scripts/m3_measure.py` and `kvlab/m3_bench.py`, unchanged. The same nine
  path/backend/mode combinations and the same ordering semantics:
  - `full` (plain), kept as context;
  - host and disk × baseline and batched, plain (primary);
  - the same four in the synchronized-phase mode, for restore-phase timing (secondary).
- **Not done:** new profiling, backend tuning, cost-model refitting or any runtime change. The
  default backend stays `baseline`; `batched` stays opt-in. D4 (constructor copy) and D10
  (full-recompute comparisons are report-only) are unchanged. The known M0 `nat_long` strict
  FAIL and the VAL-005 and VAL-006 qualifications stand as recorded.

## 3. Cases
| case (L, S) | chunks | role |
|---|---|---|
| (1024, 1) | 4 | the M3 case that was inconclusive under R3 |
| (1280, 96) | 5 | fresh: never measured in M2 or M3, and not used for development. One full staging group plus one chunk. N = 1376, so no new complete chunk is admitted |

## 4. Sampling
- **Sessions:** five fresh processes, run one after another: c1–c5, with seeds 32001–32005.
- **Per session:**
  - The runner shuffles the two cases once (seeded) and repeats that order four times. Each
    case therefore gets four separated blocks, and the cases alternate.
  - Each block runs 2 warmup repetitions (kept, flagged, never used in a statistic), then 24
    measured repetitions. `gc.collect()` runs before each repetition.
  - Each repetition runs the nine combinations in a fresh seeded random order.
- **Process warmup:** at the start of each session, every combination runs twice at both cases.
  This is the runner's `sorted({min(cases), max(cases)})`, which here is both cases.
- **Untimed checks (once per case per session, before that case's first block):**
  - M2 strict correctness for each backend;
  - the A/B check per tier;
  - restore-only device memory for each tier × backend.
- **No sample is filtered.** Slow samples stay.

The runner draws all randomness from one `random.Random(seed)`, so the whole sample sequence of
a session is determined by its seed. The validator replays it exactly. The replay reproduces all
6,480 sample identities of the five M3 evaluation sessions.

## 5. Exact evidence
| item | count | derivation |
|---|---|---|
| measured samples | 8,640 | 5 sessions × 2 cases × 4 blocks × 24 reps × 9 combinations |
| block-warmup samples | 720 | 5 × 2 × 4 × 2 × 9 |
| process-warmup samples | 180 | 5 × 2 cases × 9 combinations × 2 |
| strict correctness checks | 20 | 5 × 2 cases × 2 backends |
| A/B checks | 20 | 5 × 2 cases × 2 tiers |
| restore-memory records | 10 | 5 × 2 cases, each with both tiers × both backends |
| pairs per (case, tier, mode) | 480 | 5 × 4 × 24, in 20 strata (session × block) of 24 |

No profiler session or profiler output belongs to this experiment.

## 6. Statistics (the M3 definitions, unchanged; `kvlab/m3_stats.py`, `scripts/m3_analyze.py`)
- **Pairs:** within a repetition, d = batched − baseline (ms), for the same tier and mode.
- **Bootstrap:** stratified circular moving-block bootstrap within each contiguous stratum
  (session × block, chronological). Block lengths b ∈ {1, 2, 4, 8}; 4,000 resamples; 95%
  percentile intervals.
- **Deterministic seeds:**
  - verdict intervals: L·1000 + S + 7919·b;
  - speedup intervals: L·1000 + S + b;
  - context intervals: L + b.
- **Slow sample:** e2e > 1.3 × the minimum of its (case, tier, backend, mode), over this
  experiment's valid samples only. A regime is a maximal chronological run of slow samples.
- **Verdict per (case, tier, statistic):** "batched faster" or "batched slower" only if all
  three rules hold; otherwise inconclusive.
  - **R1:** the interval excludes 0 in the same direction for every b.
  - **R2:** the point estimate has that sign in all five sessions.
  - **R3:** the summed span of the case's strata is ≥ 10 × the longest slow regime in them,
    and the AR(1) effective sample size of d, per stratum and summed, is ≥ 10.
- **Primary:** plain e2e, the mean and the median of d. That gives 8 verdicts: 2 cases ×
  2 tiers × 2 statistics.
- **Secondary (reported separately; not part of the rule):** restore-phase verdicts from the
  synced samples.
- **Context and description (report only):**
  - restore path − `full` per backend;
  - the three speedup ratios;
  - per-session estimates;
  - slow shares;
  - copy counts and bytes;
  - device memory.
- **Not reported:** p99 or any tail claim.

## 7. Confirmation rule (fixed now)
`m3_confirm_v1_confirmed` is true only if all of the following hold:
1. The evidence is valid against spec `M3_CONFIRM_V1` (§8–§10): complete, consistent, bound
   to the freeze, and produced within the deadline.
2. All 20 strict correctness checks and all 20 A/B checks pass, recomputed from their details.
3. All 8 primary verdicts are "batched faster".

Outcomes:
- **Confirmed:** analysis exit 0, flag true.
- **Complete but not confirmed:** analysis exit 0, flag false. A verdict is inconclusive or
  slower. This is a final, valid result, and it is reported as such. It is not repeated or
  extended.
- **Correctness failure, consistently recorded, in complete evidence:** analysis exit 4. No
  performance analysis. Because of the stopping rule (§9), this is possible only if the failure
  is recorded in the last session, c5: every planned record must be present and consistent and
  the batch must have stayed within its deadline.
- **Invalid or incomplete evidence:** analysis exit 3. No performance analysis, and no statement
  either way. This is not an "inconclusive" result. It includes:
  - missing, malformed or inconsistent records;
  - a session that timed out, was killed or did not report;
  - a runner-recorded block failure (such as a caught OOM) or excluded samples;
  - sessions never launched because an earlier session exited nonzero. A correctness failure
    recorded in c1–c4 therefore ends here.

  Detailed correctness and A/B failures found in any readable session are always listed
  (`correctness_failures`, `observed_correctness_failures`), even when exit 3 takes precedence.

Scope:
- The flag is scoped to this experiment, this machine, this stack and these two cases.
- It does not change M3's `supports_adoption = false`, and it is not pooled with M3.
- Even if it is true, it does not by itself change the default backend. A default change
  needs a separate, reviewed decision in `docs/decisions.md`.

## 8. Premeasurement freeze (`scripts/m3_confirm.py freeze`, `kvlab/m3_confirm.py`)
The freeze happens once: after implementation and review, before any measurement. No commit is
required. It writes `results/m3_confirm/M3_CONFIRM_V1.freeze.json` and the sidecar
`M3_CONFIRM_V1.freeze.diff`, and it never overwrites them.

**The manifest records:**
- the base Git HEAD;
- the SHA-256 of `git diff HEAD --binary`, the file names it touches, and the diff itself
  (sidecar, hashed);
- the SHA-256 of every file a session records about itself (`kvlab/*.py`, `scripts/*.py`,
  `tests/**/*.py`, `pyproject.toml`), new untracked files included;
- the SHA-256 of every other untracked, non-ignored file outside `results/m3_confirm/`;
- the hashes of this protocol, `docs/M3_PROTOCOL.md`, `docs/M3_DESIGN.md` and
  `requirements.txt`;
- the list of analysis files;
- the exact session commands and environment, generated from the spec;
- the budget;
- the stack: Python, the package versions, the model weight blobs, the GPU name, the driver,
  the kernel and the CPU.

**It refuses:**
- while this file is marked DRAFT;
- if tracked evidence under `results/` differs from HEAD;
- if `kvlab/model.py` does not pin the required model, dtype and attention;
- if the stack differs from the M3 stack in the spec. The stack is never substituted silently;
  a different stack needs a new, separately reviewed spec;
- if a manifest or the run directory already exists.

The manifest does not hash itself. Its SHA-256 is printed at freeze time and should be
recorded outside the repository; the analysis can then require it with
`--expect-freeze-sha256`.

**The binding is checked in three places:**
- **at launch (`run`):** the working tree, the documents, the diff and the stack still equal
  the freeze. The launch environment has the M3 values of the variables the runner records,
  and `CUDA_VISIBLE_DEVICES` is unset. Otherwise nothing is launched;
- **per session (validator):** the recorded HEAD, diff hash, every file hash, the arguments
  (the frozen command), the package versions, the weights, the GPU and driver, the kernel and
  CPU, the model, the recorded process environment (the M3 values, including
  `HF_HUB_OFFLINE=1`), and a start time after the freeze and the batch launch;
- **at analysis:**
  - the validator, statistics and analysis files, and the documents, still have their frozen
    bytes. While any of them differs, the analysis rejects the evidence under this spec. This
    is a check on the analysis being run, not a permanent mark on the raw measurements:
    restoring the frozen bytes (for example from Git) reproduces the frozen analysis of the
    preserved session files. Adopting a revised analysis is a different action and needs a
    new, documented spec;
  - the batch record names the manifest by its SHA-256.

## 9. Execution, stopping rule and interruption (`scripts/m3_confirm.py run`)
- **Plan:** run all five sessions, in order, as fresh processes, one GPU process at a time.
- **No early stopping** on significance, and no outcome-dependent extension.
- **Deadline:** 1,800 s of monotonic time for the whole batch, counted from launching the
  first session process. It includes model loading, warmups, checks and measurement.
  - A session counts only if its exit is observed before the deadline. An exit first observed
    at or after it is a deadline overrun, even if the process ended slightly earlier.
  - Exits are observed when they happen: the wrapper waits on `SIGCHLD`, not on a polling
    interval.
  - The deadline is rechecked before every launch.
  - The wrapper's bookkeeping after the last exit (`finalized_s`) is not part of the budget;
    `elapsed_s` and every session's `ended_s` are, and the analysis checks the same quantities.
  - From the M3 session durations, the batch is expected to take about 15 minutes. That is an
    estimate, not a limit.
- **No automatic retries,** replacement sessions, extra repetitions or alternate seeds. A
  rerun is structurally refused, because the run directory and the manifest are never
  overwritten.
- **Stopping rule.** The batch stops, and nothing further is launched, after any of:
  - **any nonzero session exit**, whatever the session file says. The runner exits 1 after a
    failed correctness or A/B check, a caught OOM (it records the failed block and continues
    its own block loop), or excluded samples; any other code is a crash;
  - an exit 0 without a parseable session file;
  - the deadline;
  - a stop signal (SIGINT, SIGTERM, SIGHUP);
  - an error in the wrapper itself, which is recorded and then re-raised.

  The wrapper sees a session's problems only when that session exits. In particular, it cannot
  stop a session early when the runner catches an OOM inside it.
- **Execution status, separate from evidence.** `batch_record.json` records:
  - `execution_status`: `succeeded` only if every planned session was launched and exited 0
    before the deadline with a parseable session file; otherwise `failed`;
  - `stop`: the reason, the session and its actual exit code;
  - every session's exit code, timing, files, and the runner's own summary. The summary is
    shown to the operator, not trusted.

  `run` exits 0 only for `succeeded`, and 3 for every execution failure, including a nonzero
  exit of c5. The execution status never decides whether the evidence is complete; the analysis
  checks the actual evidence against the full spec (§7, §10).
- **Cleanup (SIGTERM, SIGINT, SIGHUP, deadline, or a wrapper error).**
  - **Signal handling:** the wrapper blocks these signals for the whole batch and receives them
    only synchronously: while waiting for a session, and before each launch. A signal can
    therefore not interrupt process creation, the interval between launch and the wait, or the
    cleanup itself. Signals arriving during the cleanup are only recorded (`signals_received`).
  - **The running session:** its own process group is terminated (SIGTERM, then SIGKILL after
    15 s; bounded at 30 s) and the session is reaped. Nothing outside that group is signalled.
  - **Scratch:** only that session's disk-tier scratch directory,
    `.kvcache/m3/M3_CONFIRM_V1_run/<session>/`, is removed.
  - **Record:** the record is finalized, and the previous signal mask is restored.

  A process stuck in uninterruptible sleep can outlive SIGKILL. That is recorded
  (`group_gone: false`), not waited for.
- **SIGKILL of the wrapper cannot be handled.** No final record is written (it stays
  `running`, which the analysis rejects), there is no escalation and no scratch cleanup. What is
  enforced:
  - the running session receives SIGTERM through its parent-death signal
    (`PR_SET_PDEATHSIG`);
  - a session whose wrapper died before that registration exits immediately;
  - the runner's GPU monitor has its own parent-death signal.

  A session that ignored SIGTERM would keep running; the unchanged runner does not.

**What survives an interruption.** The runner keeps all samples in memory and writes its
session JSON only at the end, so an interrupted session leaves no sample rows.

| item | finished session (any exit code) | terminated session | not launched |
|---|---|---|---|
| `session_<id>.json` | as written by the runner; SHA-256 in the batch record | normally absent. If one exists, it is kept and hashed | absent |
| `session_<id>.gpu_monitor.csv` | complete | whatever `nvidia-smi` had flushed | absent |
| `logs/session_<id>.log` | complete stdout and stderr | up to the last flushed line (one line per finished block) | absent |
| `batch_record.json` | written atomically after every state change | `execution_status: failed`, with the stop reason, session and time. If the wrapper itself is killed, it stays `running` | listed in `not_launched` |

## 10. Validation (spec `M3_CONFIRM_V1`, `kvlab/m3_validate.py`; analysis exit codes 0 / 3 / 4)
Everything in the M3 VAL-006 validation applies, with this spec's counts. In addition:
- **Exact sequence:** every sample, in file order, has the stage, case, block, repetition,
  order position and combination that the runner draws for that seed. Recorded wall times never
  go backwards.
- **Per-case records:** the correctness, A/B and memory records follow the first pass of the
  block order. There are exactly 10 restore-memory records, each covering both tiers × both
  backends as full restores.
- **Process warmups:** checked by multiplicity per (case, combination), using recorded fields
  only (no invented repetition IDs), and by their exact sequence.
- **Binding:** session provenance and the batch record are checked against the freeze manifest
  (§8). Sessions must be sequential, with distinct processes. Each exit code must agree with
  the runner's own success condition in its record.
- **Batch record:** these make the evidence invalid:
  - a record that is not finalized;
  - a stop other than a nonzero exit (deadline, signal, missing report or wrapper error);
  - planned sessions never launched;
  - a launch after a nonzero exit;
  - a killed session;
  - a session ending at or after the deadline;
  - a contradictory execution record. The execution status, stop, launched sequence, actual
    exit codes, session-file states and timing must describe a run the wrapper can produce,
    checked in both directions. Only two shapes are accepted:
    - `succeeded`: no stop, nothing unlaunched, every session exited 0 by itself with a
      session file;
    - a `nonzero_exit` stop: status `failed`, and the stop names the last launched session
      and its actual exit code. That session exited by itself; every earlier one exited 0
      with a session file.

    In both shapes, the session times are sequential, and `elapsed_s` is the last session's
    end. For example, a `nonzero_exit` stop while every recorded exit is 0, or a nonzero exit
    with no stop, is rejected (exit 3) before any performance analysis;
  - a recorded session-file state (presence, parseability, SHA-256, size) that differs from the
    actual file;
  - a time that is not a finite number. Parsing the batch record or the freeze manifest
    rejects the `NaN`, `Infinity` and `-Infinity` literals. Every compared time (batch
    launch, session start and end, elapsed time, freeze time, session start time and sample
    wall time) must also be a JSON number, not a boolean, that converts to a finite float.
    That later check rejects an overflowing literal such as `1e400`, which parses to infinity,
    and an integer beyond the float range.

  A nonzero exit by itself is not an evidence problem; the session's report is judged on its
  contents.
- **Closed file listing:** the run directory may contain only the batch record, the five
  session files, their monitor CSVs and their logs. Any other file is rejected, including
  `session_prof.json`, a `profile` section or `--profile`. The analysis writes to a separate
  directory.
- **Warmup timings** never enter a statistic.

## 11. Commands (NOT EXECUTED; run only after review and freeze)
First, once, after the review:
1. Replace the DRAFT status in this file's first line (see the note above).
2. Run `.venv/bin/python scripts/m3_confirm.py freeze`. It prints the manifest SHA-256; record
   it outside the repository.

Then run, in one Bash shell:
```bash
set -o pipefail
SHA=<the freeze manifest SHA-256>
TS=$(date +%Y%m%dT%H%M%S)
.venv/bin/python scripts/m3_confirm.py verify
.venv/bin/python scripts/m3_confirm.py run --expect-freeze-sha256 "$SHA" 2>&1 | tee "local/m3_confirm_run_$TS.log"
run_status=${PIPESTATUS[0]}
echo "wrapper exit status: $run_status"
.venv/bin/python scripts/m3_analyze.py --spec M3_CONFIRM_V1 --run-dir results/m3_confirm/M3_CONFIRM_V1_run \
    --out-dir results/m3_confirm/M3_CONFIRM_V1_analysis --expect-freeze-sha256 "$SHA" 2>&1 | tee "local/m3_confirm_analyze_$TS.log"
analyze_status=${PIPESTATUS[0]}
echo "analysis exit status: $analyze_status"
```
- **Exit statuses:** a pipeline's own status is `tee`'s unless `pipefail` is set. Even with it,
  only `${PIPESTATUS[0]}`, read on the very next line, is the wrapper's exact status. Any later
  command overwrites it.
- **Always run the analysis,** whatever `run_status` is. It classifies the evidence and lists
  recorded correctness failures in every case.
- **Confirmation:** reported only if `run_status` is 0, `analyze_status` is 0, and the analysis
  says `M3_CONFIRM_V1 confirmed: True`.

The `run` step executes, in order, for i = 1…5:
`HF_HUB_OFFLINE=1 .venv/bin/python scripts/m3_measure.py --out-dir results/m3_confirm/M3_CONFIRM_V1_run --session c$i --cases 1024:1,1280:96 --blocks 4 --warmup 2 --reps 24 --seed 3200$i --staging-chunks 4`

## 12. Limitations
- **Scope:** one machine (RTX 4080 Laptop, WSL2, driver 596.49), one model and stack, two
  cases, and warm disk only. No cold-NVMe or tail-latency claim.
- **Case selection:** (1024, 1) was chosen because M3 left it inconclusive, so this is a
  targeted confirmation, not a fresh random draw. (1280, 96) is fresh, but it was chosen from
  the same range as the M3 cases.
- **Unresolved slow mode:** its root cause is open (D29). Long slow regimes can make R3 fail
  again. A complete batch that fails R3 is a valid "not confirmed" result.
- **Self-attested freeze:** the freeze manifest and the batch record live in this repository.
  They bind the evidence to one frozen state, but they cannot prove that no one regenerated all
  of them; recording the manifest SHA-256 out of band at freeze time narrows that.
- **No in-flight samples:** the runner writes each session file only at its end, so an
  interrupted session contributes nothing.
- **Late failure detection:** a failure inside a session (a caught OOM, a failed check) is
  visible to the wrapper only when that session exits.
- **Signals the wrapper cannot handle:** it handles SIGINT, SIGTERM and SIGHUP; SIGKILL and
  other fatal signals end it without a final record (§9).
- **Comparisons:** the restore-phase timing uses the M2 synchronized mode, which adds
  synchronizations. `full` is context only (D10).
