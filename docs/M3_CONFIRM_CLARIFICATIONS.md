# M3_CONFIRM_V1: post-measurement clarifications

This is a dated record kept separately from the frozen protocol. It does not modify:
- `docs/M3_CONFIRM_PROTOCOL.md` (frozen, SHA-256 `78bfdedd…6530`);
- the freeze manifest (`results/m3_confirm/M3_CONFIRM_V1.freeze.json`, SHA-256 `dd58b894…8216`);
- any analysis file or analysis output.

It adds no acceptance criterion, removes none, and changes no verdict or flag.

## C1 (2026-11-04): R3's effective-sample-size condition is a threshold on the sum

**Ambiguous wording.** The frozen protocol's §6 (lines 94–95) reads:

> **R3:** the summed span of the case's strata is ≥ 10 × the longest slow regime in them,
> and the AR(1) effective sample size of d, per stratum and summed, is ≥ 10.

On its own, "per stratum and summed, is ≥ 10" could be read as two conditions: every stratum
≥ 10, and the sum ≥ 10.

**Operational rule, fixed before measurement:**
1. For each stratum (session × block, 24 pairs, chronological), compute the AR(1) effective
   sample size of d: n(1 − ρ₁)/(1 + ρ₁) for ρ₁ > 0, and n otherwise.
2. Sum these values over the case's 20 strata.
3. R3 requires that sum to be ≥ 10, together with the span condition.

There is no per-stratum minimum.

**Premeasurement sources for that rule:**
- **The protocol's own references.** `docs/M3_CONFIRM_PROTOCOL.md` §6 is titled "Statistics
  (the M3 definitions, unchanged; `kvlab/m3_stats.py`, `scripts/m3_analyze.py`)". Its
  introduction says the experiment runs "under the unchanged M3 rules". The reviewed
  pre-freeze draft has the same §6 text.
- **The M3 protocol.** `docs/M3_PROTOCOL.md` §6 (frozen before the M3 evaluation,
  SHA-256 `9d3edc57…901f`) says: "the AR(1) effective sample size of d (per stratum, summed)
  is ≥ 10".
- **The frozen implementation.** `kvlab/m3_stats.py` (SHA-256 `a8456244…c5fe`):
  - `effective_n` is documented as "Sum over strata of the AR(1) effective sample size of the
    paired differences";
  - `classify_case` evaluates `n_eff = effective_n(d_strata)` and
    `r3 = span_s >= 10 * longest_regime_s and n_eff >= 10`.
- **That file's recorded hash.** The M3_CONFIRM_V1 freeze manifest binds this hash before any
  confirmation sample, both in `files_sha256` and as an analysis file. The same hash is
  already recorded in the M3 evaluation's sessions s2–s5 and its profiler session
  (`results/m3/eval_20261103T173632/`, `source.files_sha256`); session s1 started before the
  file existed. The code was therefore unchanged from the M3 evaluation through the
  confirmation.
- **The analysis output.** It records one value per row,
  `cases[case].tiers[tier].e2e[stat].R3.ar1_effective_n`, which is the sum.

**Values under that rule** (`results/m3_confirm/M3_CONFIRM_V1_analysis/m3_analysis.json`):

| case | tier | summed AR(1) n_eff | R3 |
|---|---|---|---|
| (1024, 1) | host | 352.2 | pass |
| (1024, 1) | disk (warm) | 367.8 | pass |
| (1280, 96) | host | 368.7 | pass |
| (1280, 96) | disk (warm) | 360.5 | pass |

**Individual strata (descriptive only).** Six of the 80 (case, tier, stratum) values are below
10:
- (1024, 1) host: c1/block 7, 9.3;
- (1024, 1) disk: c5/block 5, 7.4;
- (1280, 96) host: c1/block 4, 8.8; c2/block 6, 6.2; c5/block 0, 7.7;
- (1280, 96) disk: c3/block 2, 6.9.

Two separate CPU recomputations from the saved evidence agree on these values:
- the independent result review, which used the session JSONs;
- a later check, which applied the frozen `kvlab/perf_stats.py` functions to
  `samples_chronological.csv`.

Both reproduce the summed values above.

**Consequence, stated openly.** A per-stratum minimum of 10 would have failed R3 in all four
(case, tier) combinations. That rule is not in the frozen protocol's operational definition,
and adopting it now would be a post-hoc criterion. It is not applied. The recorded result
stands: all 8 primary verdicts are "batched faster", and `m3_confirm_v1_confirmed = true`.
This flag does not change M3's `supports_adoption = false`. The default backend remains
`baseline`.
