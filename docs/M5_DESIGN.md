# M5a design: an opt-in restore-length policy and a versioned cost model

**Status, 2026-11-05.** This design note was written when the code was implemented and tested on CPU only (a
tiny random Qwen2, temporary stores and synthetic data), and nothing in it makes a decision-quality or
performance claim.

**Later the same day:** the GPU gate passed, and one calibration, fit and held-out evaluation ran and are valid
(`docs/M5_PROTOCOL_DRAFT.md`; results in `docs/results.md`, M5). On all 8 evaluation cases CostAware chose the
same length as AlwaysRestore, so no incremental benefit was demonstrated.

Decisions D40 and D41. The M5b preparation (harness, validation, fit, analysis and GPU gate) is in that protocol
and decisions D42–D43.

## 1. Scope
**New:**
- `kvlab/restore_policy.py`: a side-effect-free restore planner, the policies `AlwaysRestore`,
  `Recompute`, `FixedChunks` and `CostAware`, selection validation, and the per-request decision record;
- `kvlab/restore_cost.py`: a separately versioned latency model (`kvlab-restore-cost-v1`), with fitting,
  artifact validation, compatibility and domain checks, and prediction;
- an optional, keyword-only `policy` argument to `AlwaysRestoreEngine`, used by both `prefill_request()`
  and `run()`.

**Unchanged:**
- the chunk format, identity, tiers, eviction, no promotion, ownership, budgets and the restore
  mechanics of both backends (`TieredKVCache.restore()` is called as it is);
- the default backend, `baseline`;
- every M2/M3/M4 model, protocol, analysis file and piece of evidence. `kvlab/perf.py` and
  `kvlab/perf_stats.py` are not edited, and the M2 model is never loaded by M5a;
- D4, D10 and the known M0 strict failure.

## 2. The planner (`plan_restore`)
For prompt length N and chunk size C, the candidate restore lengths are `0, C, …, k·C`:
- **First-token rule:** k·C ≤ `max_restore_len(N, C)` (design §2): complete chunks only, and at least
  the last prompt token is recomputed. L = 0 is always a candidate.
- **Stop rules:** k stops at the first chunk that no tier indexes (`absent`), and at the first disk chunk
  when the host store has no staging buffer (`no_staging_buffer`), because `restore()` can never read it.
  Otherwise the stop is `eligible_limit`, or `no_eligible_chunk` when N ≤ C.
- **Indexed, not validated:** planning sees only index membership (`TieredKVCache.tier_of`). A planned
  chunk may still fail validation when `restore()` reads it. `RestorePlan.to_dict()` records this as
  `presence: indexed_not_validated`.
- **Configuration:** the backend is validated (`_check_backend`), so a configuration that `restore()`
  would reject fails before any choice.

**No side effects.** Planning does not:
- read, stat or validate a file;
- lease, touch or allocate an entry;
- change LRU order, counters, slots, staging, files or `last_restore`;
- create restore tensors or run the model.

The tests forbid file system calls, leases, staging, allocation, restore destinations and `restore()`
during planning, and compare the complete store state and file bytes before and after.

**The record (`RestorePlan`, frozen) holds:**
- the eligible length;
- the ordered tiers of the indexed prefix;
- the feasible candidate lengths and the stop reason;
- the namespace digest;
- the backend and its staging group;
- whether a staging buffer exists.

## 3. Policies
A policy maps a plan to a `PolicyDecision(selected_len, reason, objective_ms=None)`. The engine validates
the decision against the plan before any restore work (`validate_selection`). It must be a
`PolicyDecision` with a non-empty reason. The selection must be an integer (bool, float and numpy float
are rejected; numpy integers are accepted), non-negative, chunk-aligned and one of the plan's candidates.
Otherwise `PolicyError` is raised. Nothing is clamped.

| Policy | Choice | Deterministic edge behaviour |
|---|---|---|
| `AlwaysRestore` | the longest feasible prefix | no positive candidate: L = 0 (`no_feasible_restore`) |
| `Recompute` | L = 0 (`policy_recompute`) | — |
| `FixedChunks(k)` | exactly k·C | k·C not a candidate: `PolicyError` before restore; it never substitutes another length |
| `CostAware(model)` | argmin of the predicted objective over the candidates | see below |

**`CostAware` in order:**
1. **Compatibility:** the namespace digest, chunk size and backend must match the artifact, and so must
   the staging group for the batched backend. A mismatch raises `IncompatibleModelError`. It is a
   configuration error, never a fallback.
2. **No positive candidate:** L = 0, without consulting the model.
3. **Outside the declared domain:** the always-restore choice, with the reason
   `out_of_domain_always_restore:<field>`. This is the only fallback, and only `domain_violation()`
   triggers it.
4. **Otherwise:** the candidate with the smallest objective. Exact ties go to the larger L, so the policy
   departs from always-restore only on a strict predicted improvement. Each candidate's objective is
   recorded.

## 4. Engine integration
`AlwaysRestoreEngine(model, tiered, *, policy=None)`. `engine.policy` may be replaced between requests.

**policy=None** keeps the M1 operational flow unchanged: the same restore call and arguments, phase marks,
counters, callbacks, prefill, store and decode. The only new work is a `decision = None` assignment and a
`self.policy` check; the result types gain a `decision` field that defaults to `None`. A CPU differential test
compares it with an adapted copy (`self` → `eng`, a `SimpleNamespace` result) of the pre-M5a `prefill_request`
body. It compares results, logits, cache bytes, counters, phase marks, forward calls, callbacks, `restore()` key
counts, LRU orders and `last_restore`. Planning and validation are patched to fail during that test.

**allow_restore=False** never consults a policy. It is unchanged, `restore_not_attempted` included.

**With a policy:**
1. ingest (ids, chain keys);
2. `plan_restore`, then `policy.choose`, then `validate_selection` (phase mark `decide`);
3. if the selection is positive, `tiered.restore(token_ids, keys[:selected // C], …)`, so no chunk beyond
   the selected prefix is looked up, leased or read;
4. if the selection is 0, no restore call, and `tiered.last_restore` is set to `None`, so it cannot report
   an earlier request's restore;
5. exactly one prefill, from the **actual** verified length;
6. store/admission exactly as before (every complete prompt chunk that is not indexed);
7. `Prefilled.decision` and `RequestResult.decision` hold a frozen `RestoreDecision`: the plan, the policy
   name, `selected_len`, the reason, the objectives, `actual_len`, the actual sources and
   `restore_miss_reason`.

A validation failure inside the selected prefix (CRC, a missing file, a read `OSError`) keeps the verified
prefix that `restore()` returns, so `actual_len < selected_len` and `truncated` is true. There is no
re-plan and no second inference.

**Counters and miss reasons under a policy:**
- **Unchanged meaning:** `requests`, `restored_requests` and `full_recompute_requests` follow the actual L.
  Hits, bytes and admissions are the existing counters.
- `chunk_misses` and `miss_reasons` count only what `restore()` meets inside the selected prefix (validation
  or read failures).
- An absent chunk or a missing staging buffer that limits the candidates is recorded in the plan
  (`limit_reason`). It is not counted as a miss. The policy=None path counts it as before.
- A policy L = 0 is not a miss: `miss_reason` is `None` and the reason is in the decision.
- No counter field was added, so the historical `Counters.to_dict()` schema is unchanged.

**LRU side effect of a shorter selection.** Chunks between the selected length and the indexed prefix are
neither restored nor touched, and admission skips them because they are indexed. So a shorter selection
leaves different LRU recency than always-restore would. This is intended (only restored chunks are used),
but it means policies affect later requests' cache state. Trace-level effects are out of M5a's scope.

**Known pre-existing diagnostic limitation (unchanged):** with `allow_restore=False` (with or without a
policy attached), `tiered.last_restore` keeps the previous request's value, as before M5a.

## 5. The cost model (`kvlab/restore_cost.py`, schema `kvlab-restore-cost-v1`)
For a candidate L of a request with prompt length N:
- S = N − L is the number of recomputed tokens;
- h and d are the host and disk chunks in [0, L), in restore order;
- m is the number of admitted chunks: complete prompt chunks not indexed before the request.

Token features are in kilotokens; the result is in ms.

```
L = 0:  T = f_const + f_n·N + f_pairs·N²/2                                  + a·m
L > 0:  T = p_const + r_host·h + r_disk·d + p_s·S + p_pairs·S·(L + S/2)     + a·m
```

**Basis (9 coefficients) and identifiability:**
- **The pair features** count the query-key pairs of a causal forward: N²/2 for a full prefill, and
  S·L + S²/2 for a suffix. S² therefore enters only with the ratio the pair count fixes. Omitting a free S²
  coefficient is a chosen model constraint (attention work proportional to query-key pairs), not an
  identifiability limit: with varied suffix lengths, such a column would be identifiable (on the synthetic
  test design the 10-column matrix has full rank).
- **Separate path coefficients.** The full prefill (the causal SDPA path) and the suffix prefill (the
  explicit-mask path) have their own constants and pair coefficients. The M2 fit (baseline backend, S ≤ C)
  gave the two paths different fixed terms, about 13 and 17 ms (`docs/results.md`, M2).
- **Per-chunk tier costs** make mixed host/disk prefixes additive. A per-group placement term for the
  batched backend is not modelled; its residual will show up in the prediction error.
- **Ingest** (chain keys; about 0.13 ms per kt in the M2 fit) is not modelled separately. It is absorbed into the
  constants and the token terms.
- **The admission term a·m** is shared by both paths, because both run the same D2H copies.
- **Rank:** the fit requires full column rank of the column-scaled design (singular values above 1e-9 ×
  the largest). Row counts are not accepted as identifiability. This needs at least three distinct N
  among the L = 0 rows, both tiers, positive candidates over several suffix lengths, and a variable m
  (a constant m is collinear with the two path constants).

**Target and fit:**
- **Target:** the case mean of the synchronous first-token latency per candidate. Means, not medians,
  because the future metric is mean regret.
- **Method:** non-negative least squares over all candidate rows, by exact support enumeration on the
  column-scaled design. `perf.nnls_small` uses the same algorithm but is capped at 8 columns and is frozen.
- **Weighting:** explicit and recorded. `uniform` gives absolute ms errors; `inverse_mean` gives relative
  errors.

**Objective.** The policy minimizes T − a·m. Under the controlled-state assumption, m is identical for
every candidate of one request: the candidate does not change what is admitted, because indexed chunks are
never re-admitted. The term therefore cannot change the argmin. The assumption fails if admission spills
differently per candidate. The draft protocol requires states where admission does not spill. Admission
stays in the engine, in `predict_ms` and in every future measurement.

**Fitting entry point:** `fit_restore_cost(cases, *, calibration_case_ids, evaluation_case_ids,
compatibility, weighting, provenance)`.
- **A case** is one request/state: N, the tiers of the prepared indexed prefix, m and one measured mean
  per candidate. It must contain every legal candidate exactly once.
- **Refused, never filtered** (`FitError`):
  - a case declared for evaluation, an undeclared case, a duplicate or a missing declared case;
  - overlapping calibration and evaluation declarations;
  - non-finite, non-positive or boolean values, misaligned or illegal lengths, an impossible m;
  - unknown fields and non-canonical provenance.

**The artifact** is a canonical-JSON dict with:
- the schema;
- exact units, target and objective strings;
- the feature names;
- `compatibility`: the namespace digest, chunk size, backend and staging group (`null` for baseline);
- the coefficients;
- the fit method, weighting, rank, condition number and in-sample residuals;
- the domain;
- the calibration and evaluation case ids, and the digest of the normalized cases;
- free provenance;
- `digest`, a SHA-256 over everything else (domain-separated).

`RestoreCostModel(artifact)` validates everything once (exact keys, types, finite non-negative
coefficients, sorted disjoint ids, the digest) and caches the coefficients as floats. A decision is plain
arithmetic: nothing is re-serialized or re-hashed per request or candidate. `load_model` also rejects
NaN/Infinity tokens. `save_artifact` never overwrites. Malformed or tampered artifacts raise
`ArtifactError`.

**Domain:**
- **Contents:** per-field calibration ranges for N, the suffix length S, the restore chunks, the host
  chunks, the disk chunks and m.
- **Shape:** a box, not the calibration hull, so some in-box combinations were never measured.
- **Branch-specific bounds:** the S range is derived from, and applies to, positive candidates (L > 0) only.
  The L = 0 candidate (the full-prefill path) is bounded by the N range.
- **Decision domain:** the box projected onto the objective's inputs (N, and for each positive candidate S and
  the chunk counts). When a plan has a positive candidate, every feasible candidate is checked, and a violation
  makes `CostAware` use the documented always-restore fallback. A plan with no positive candidate is decided
  (L = 0, `no_feasible_restore`) before any domain check (§3). The m range is **not** checked for decisions:
  m does not enter the objective, a plan does not carry it, and under the controlled-state assumption it is the
  same for every candidate, so it cannot change the choice.
- **Prediction domain:** the full box, m included, with the same L > 0 restriction on S. `predict_ms` raises
  `OutOfDomainError` instead of extrapolating, and `ValueError` for impossible inputs (for example m greater
  than the complete chunks after L).

**No artifact ships.** Tests build artifacts from synthetic coefficients only (`tests/m5_toy.py`).

## 6. Invariants and edge cases
- The selected length is always one of the plan's candidates. The actual length is always a prefix of the
  selection, restored with full validation.
- Every request makes exactly one prefill and `steps − 1` decode calls, including after read and write
  failures.
- Leases and the staging slot are released and the budgets hold. Forced restores are checked strictly
  against a `ResidentReference` replay of the actual restore length and the actual stored chunks.
- No bitwise comparison is made across different restore lengths or against full prefill (D10).
- An incompatible model, an invalid policy output and an infeasible forced length all raise before any
  restore, prefill or counter change.

## 7. Limitations
- **Not run when this note was written:** no GPU execution, calibration or evaluation (they ran later;
  see the status note above). Whether any candidate other than the extremes was ever best was then unknown. The M2/M3 numbers suggest the choice may often be degenerate
  (always-restore), and that is an acceptable outcome.
- **Single process:** sequential requests, batch 1, synchronous restores; no async, pipelining,
  concurrency or vLLM.
- **No integration** into the M4 trace replay, and no admission policy.
- **Coarse domain:** a box; finer hull checks are deferred.
