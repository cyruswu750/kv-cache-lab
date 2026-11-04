"""M3 evidence validation, fail closed (docs/M3_PROTOCOL.md; BUGLOG VAL-006). CPU only.

`scripts/m3_analyze.py` must not infer what an experiment should contain from whichever files
happen to be in a run directory. `load_and_validate()` checks the directory against an explicit
specification (`SPECS`) before any statistic is computed, and returns one of three statuses:

- `invalid`: the evidence is incomplete or inconsistent, so no performance result may be drawn.
  Examples:
  - a planned session, case, block, repetition or path/backend/mode sample is missing,
    duplicated or outside the plan, so some pair is incomplete;
  - a retained warmup record (block warmup or process warmup) is missing, duplicated,
    substituted, unexpected, incorrectly flagged or malformed. Warmups are never used in any
    statistic, but they are part of the frozen procedure's evidence;
  - a measured sample fails the M2/M3 validity rules;
  - a correctness, A/B or restore-memory record, or one of its detailed fields, is missing or
    malformed;
  - a stored summary flag (`strict_pass`, `strict_pass_all`, `pass`, `correctness_all`,
    `n_measured`, `n_excluded`) disagrees with what the detailed records give;
  - the recorded configuration or source identity differs from the spec.
- `correctness_failed`: the evidence is complete and consistent, but a required detailed
  correctness or A/B check failed. Decision clause (1) fails, and no performance result is drawn.
- `valid`: complete, consistent, and every required check passed. The performance analysis may
  run. Its result can still be inconclusive; that is a valid outcome, not invalid evidence.

Acceptance is recomputed from the detailed checks against counts that come from the spec, so an
empty or partial set of checks can never pass. Stored summary flags are compared, never trusted.

No model weights and no CUDA are needed. Only the per-sample M2/M3 rules
(`kvlab.m3_bench.sample_problems`) are imported, lazily; that imports torch but does not
initialise CUDA or load the model.
"""

import json
import math
from collections import Counter
from pathlib import Path

from kvlab import perf

C = perf.C

# The frozen M3 evaluation (results/m3/eval_20261103T173632).
# Sources: docs/M3_PROTOCOL.md §3-§5 and §7-§8 (hash in
# results/m3/protocol_freeze_20261103T173632.sha256), and the recorded commands in
# results/m3/eval_20261103T173632/logs/commands.txt.
FROZEN_M3_EVAL = {
    "name": "m3_eval_20261103T173632",
    "source": "docs/M3_PROTOCOL.md (frozen 2026-11-03) + results/m3/eval_20261103T173632/logs/commands.txt",
    "git_head": "362ee40aeaaae204b2b1dc88398b46bc5826f410",
    "sessions": {"s1": 31001, "s2": 31002, "s3": 31003, "s4": 31004, "s5": 31005},  # session id -> seed
    "cases": ((256, 32), (1024, 1), (1536, 128), (3072, 256), (4096, 64)),
    "blocks": 2,    # blocks per case per session
    "warmup": 2,    # warmup repetitions per block (kept, flagged, never used)
    # process warmup (protocol §4; scripts/m3_measure.py): before block 0, every combination twice
    # at the smallest and the largest case, `sorted({min(cases), max(cases)})`. Its records carry
    # stage "process_warmup" and warmup true, and no block, rep or order_pos.
    "process_warmup": {"cases": ((256, 32), (4096, 64)), "repeats": 2},
    "reps": 12,     # measured repetitions per block
    "staging_chunks": 4,
    "num_layers": 28,
    "tiers": ("host", "disk"),
    "backends": ("baseline", "batched"),
    # every (path, backend, mode) sample of one repetition; equal to kvlab.m3_bench.COMBOS
    "combos": (("full", "none", "plain"),
               ("host", "baseline", "plain"), ("host", "batched", "plain"),
               ("disk", "baseline", "plain"), ("disk", "batched", "plain"),
               ("host", "baseline", "synced"), ("host", "batched", "synced"),
               ("disk", "baseline", "synced"), ("disk", "batched", "synced")),
    "adoption_min_chunks": 4,  # protocol §8 (3): required to be established for >= 4 chunks
    # files that executed the measurement; every session must record them, with identical bytes
    "executed_sources": ("kvlab/tiered_cache.py", "kvlab/m3_bench.py", "kvlab/perf_harness.py",
                         "kvlab/perf.py", "scripts/m3_measure.py"),
    # the separate call-count session (§7): optional, but validated when present
    "profile_session": {"id": "prof", "cases": ((256, 32), (1536, 128), (4096, 64)),
                        "units": ("restore", "request")},
}
SPECS = {FROZEN_M3_EVAL["name"]: FROZEN_M3_EVAL}
DEFAULT_SPEC = FROZEN_M3_EVAL["name"]

SESSION_SECTIONS = ("meta", "source", "restore_config", "samples", "correctness", "ab_checks",
                    "restore_memory", "failures", "blocks", "excluded")
# Bench.correctness (kvlab/perf_harness.py) fields; strict_pass is re-derived from the details
CORRECTNESS_FIELDS = ("L", "S", "host_restore_len", "host_sources", "disk_restore_len", "disk_sources",
                      "restored_kv_bytes_equal", "strict_host_eq_disk_bitwise",
                      "strict_host_eq_resident_bitwise", "strict_pass", "report_only_full_vs_restored")
# M3Bench.ab_check (kvlab/m3_bench.py) boolean fields per tier
AB_FIELDS = ("restore_len_sources_admission_equal", "kv_bytes_equal_after_prefill",
             "first_logits_bitwise_equal", "greedy_tokens_equal", "step_logits_bitwise_equal")
MAX_LISTED = 5  # sample-level problems listed individually per session; the rest are counted


def _sample_problems(s: dict, num_layers: int) -> list[str]:
    from kvlab import m3_bench as mb  # lazy: torch import only (no CUDA init, no weights)
    return mb.sample_problems(s, num_layers)


def _warmup_field_problems(s: dict) -> list[str]:
    """Fields a kept warmup record must have; the analysis lists every warmup in
    samples_chronological.csv but never uses its timing."""
    ph = s.get("phases_ns")
    checks = (("t_wall", isinstance(s.get("t_wall"), (int, float)) and math.isfinite(s["t_wall"])),
              ("e2e_ns", isinstance(s.get("e2e_ns"), int) and s["e2e_ns"] > 0),
              ("restore_len", isinstance(s.get("restore_len"), int)),
              ("sources", isinstance(s.get("sources"), list)),
              ("stored_chunks", isinstance(s.get("stored_chunks"), list)),
              ("phases_ns.restore", s.get("mode") != "synced" or (isinstance(ph, dict) and isinstance(ph.get("restore"), int))))
    return [f for f, ok in checks if not ok]


def _case(L, S):
    return (L, S) if isinstance(L, int) and isinstance(S, int) else None


def correctness_detail(e: dict, L: int, n: int) -> dict:
    """The strict checks of Bench.correctness, re-derived from its recorded detail fields."""
    kv = e["restored_kv_bytes_equal"] if isinstance(e["restored_kv_bytes_equal"], dict) else {}
    return {"host_restore_len": e["host_restore_len"] == L, "disk_restore_len": e["disk_restore_len"] == L,
            "host_sources": e["host_sources"] == ["host"] * n, "disk_sources": e["disk_sources"] == ["disk"] * n,
            "host_kv_bytes_equal": kv.get("host") is True, "disk_kv_bytes_equal": kv.get("disk") is True,
            "host_eq_disk_bitwise": e["strict_host_eq_disk_bitwise"] is True,
            "host_eq_resident_bitwise": e["strict_host_eq_resident_bitwise"] is True}


def ab_detail(t: dict, tier: str, L: int, n: int) -> dict:
    """The A/B checks of M3Bench.ab_check for one tier. It also checks that both requests actually
    restored the whole prefix from that tier; two misses would compare two recomputes."""
    return {**{f: t[f] is True for f in AB_FIELDS},
            "restored_whole_prefix_from_tier": t["restore_len"] == L and t["sources"] == [tier] * n}


def _new_counts(spec):
    n_s, n_c = len(spec["sessions"]), len(spec["cases"])
    pw = spec["process_warmup"]
    exp = {"measured_samples": n_s * n_c * spec["blocks"] * spec["reps"] * len(spec["combos"]),
           "block_warmup_samples": n_s * n_c * spec["blocks"] * spec["warmup"] * len(spec["combos"]),
           "process_warmup_samples": n_s * len(pw["cases"]) * len(spec["combos"]) * pw["repeats"],
           "correctness_checks": n_s * n_c * len(spec["backends"]),
           "ab_checks": n_s * n_c * len(spec["tiers"])}
    return {k: {"expected": v, "present": 0, "passed": 0} for k, v in exp.items()}


def _check_session(sid, rep, spec, P, fail, counts):
    """Append problems (P) and failed checks (fail) for one evaluation session. Returns the
    recomputed per-session acceptance of its correctness and A/B checks."""
    missing = [k for k in SESSION_SECTIONS if k not in rep]
    if missing:
        P(f"missing sections {missing}")
        return False
    meta, cases = rep["meta"], [tuple(c) for c in spec["cases"]]
    n_blocks, n_rep = spec["blocks"] * len(cases), spec["warmup"] + spec["reps"]
    seed = spec["sessions"][sid]

    # ---- recorded configuration vs the spec
    args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
    try:
        rec_cases = sorted(tuple(int(x) for x in c.split(":")) for c in str(args.get("cases")).split(","))
    except ValueError:
        rec_cases = None
    if rec_cases != sorted(cases):
        P(f"recorded cases {args.get('cases')!r} differ from the spec {cases}")
    want = {"session": sid, "blocks": spec["blocks"], "warmup": spec["warmup"], "reps": spec["reps"],
            "seed": seed, "staging_chunks": spec["staging_chunks"], "profile": False}
    for k, v in want.items():
        if args.get(k) != v:
            P(f"recorded argument {k}={args.get(k)!r}, spec {v!r}")
    if meta.get("session") != sid or meta.get("seed") != seed:
        P(f"meta session/seed {meta.get('session')!r}/{meta.get('seed')!r}, spec {sid!r}/{seed}")
    head = rep["source"].get("git_head") if isinstance(rep["source"], dict) else None
    if head != spec["git_head"]:
        P(f"git_head {head!r}, spec {spec['git_head']}")
    if not isinstance(rep["restore_config"], dict) or \
            rep["restore_config"].get("restore_staging_chunks") != spec["staging_chunks"]:
        P(f"restore_config {rep['restore_config']!r}, spec staging_chunks {spec['staging_chunks']}")
    if rep["failures"] != []:
        P(f"recorded block failures {rep['failures']!r}")

    # ---- block plan: every spec case exactly `blocks` times
    order = [_case(*x) if isinstance(x, list) and len(x) == 2 else None for x in meta.get("block_order") or []]
    if len(order) != n_blocks or Counter(order) != Counter({c: spec["blocks"] for c in cases}):
        P(f"block_order {meta.get('block_order')!r} is not {spec['blocks']} blocks of each spec case")
        return False  # sample identities cannot be placed without the block plan
    blocks = [(b.get("block"), b.get("L"), b.get("S")) if isinstance(b, dict) else None for b in rep["blocks"]]
    if blocks != [(i, *order[i]) for i in range(n_blocks)]:
        P(f"block records {blocks} do not match the block order")

    # ---- sample identities: (block, rep, path, backend, mode), exactly once each, for measured
    # and block-warmup repetitions alike; process warmups by (case, combination) multiplicity.
    # Warmups are evidence of the frozen procedure; they are never used in any statistic.
    measured, warm, positions, n_not_warmup = {}, {}, {}, 0
    pw, pw_seen, pw_last, first_m3 = spec["process_warmup"], Counter(), -1, None
    for i, s in enumerate(rep["samples"]):
        if not isinstance(s, dict):
            P(f"sample {i} is not a record")
            continue
        n_not_warmup += not s.get("warmup")
        if s.get("stage") == "process_warmup":
            pw_last = i
            ident = (_case(s.get("L"), s.get("S")), (s.get("path"), s.get("backend"), s.get("mode")))
            if s.get("warmup") is not True:
                P(f"sample {i}: process-warmup sample not flagged as warmup ({s.get('warmup')!r})")
            elif any(k in s for k in ("block", "rep", "order_pos")):
                P(f"sample {i}: process-warmup sample carries block/rep/order_pos fields")
            elif s.get("session") != sid or ident[0] not in pw["cases"] or ident[1] not in spec["combos"]:
                P(f"sample {i}: process-warmup session/case/combination {s.get('session')!r} {ident} not in the spec")
            else:
                pw_seen[ident] += 1
                counts["process_warmup_samples"]["present"] += 1
                bad = _warmup_field_problems(s)
                counts["process_warmup_samples"]["passed"] += not bad
                if bad:
                    P(f"sample {i}: malformed process-warmup record: {bad}")
            continue
        if s.get("stage") == "m3" and first_m3 is None:
            first_m3 = i
        if s.get("stage") != "m3":
            P(f"sample {i}: unknown stage {s.get('stage')!r}")
            continue
        b, r, combo = s.get("block"), s.get("rep"), (s.get("path"), s.get("backend"), s.get("mode"))
        if not (isinstance(b, int) and 0 <= b < n_blocks and isinstance(r, int) and 0 <= r < n_rep):
            P(f"sample {i}: block/rep {b!r}/{r!r} outside the plan ({n_blocks} blocks x {n_rep} reps)")
            continue
        if combo not in spec["combos"]:
            P(f"sample {i}: path/backend/mode {combo} not in the spec")
            continue
        if s.get("session") != sid or _case(s.get("L"), s.get("S")) != order[b]:
            P(f"sample {i}: session/case {s.get('session')!r} {(s.get('L'), s.get('S'))} != {sid} {order[b]} of block {b}")
            continue
        if s.get("warmup") is not (r < spec["warmup"]):
            P(f"sample {i}: warmup flag {s.get('warmup')!r} for repetition {r} (warmup reps: {spec['warmup']})")
            continue
        key = (b, r, *combo)
        target = warm if r < spec["warmup"] else measured
        if key in target:
            P(f"duplicate {'block-warmup' if target is warm else 'measured'} sample identity "
              f"(block, rep, path, backend, mode) = {key}")
            continue
        target[key] = s
        positions.setdefault((b, r), []).append(s.get("order_pos"))
        if target is warm:
            counts["block_warmup_samples"]["present"] += 1
            bad = _warmup_field_problems(s)
            counts["block_warmup_samples"]["passed"] += not bad
            if bad:
                P(f"sample {i}: malformed block-warmup record {key}: {bad}")
    planned = [(b, r, *c) for b in range(n_blocks) for r in range(spec["warmup"], n_rep) for c in spec["combos"]]
    absent = [k for k in planned if k not in measured]
    if absent:
        by_case = dict(Counter(order[k[0]] for k in absent))
        P(f"{len(absent)} of {len(planned)} planned measured samples missing, so their pairs are incomplete "
          f"(by case {by_case}; first {absent[0]})")
    planned_warm = [(b, r, *c) for b in range(n_blocks) for r in range(spec["warmup"]) for c in spec["combos"]]
    absent = [k for k in planned_warm if k not in warm]
    if absent:
        P(f"{len(absent)} of {len(planned_warm)} planned block-warmup samples missing "
          f"(by case {dict(Counter(order[k[0]] for k in absent))}; first {absent[0]})")
    want_pw = Counter({(c, combo): pw["repeats"] for c in pw["cases"] for combo in spec["combos"]})
    if pw_seen != want_pw:
        short = {k: f"{pw_seen[k]}/{n}" for k, n in want_pw.items() if pw_seen[k] < n}
        extra = {k: f"{n}/{want_pw[k]}" for k, n in pw_seen.items() if n > want_pw[k]}
        P(f"process-warmup records differ from the plan ({sum(want_pw.values())} = {len(pw['cases'])} cases x "
          f"{len(spec['combos'])} combinations x {pw['repeats']}): missing {short or 'none'}; surplus {extra or 'none'}")
    if first_m3 is not None and pw_last > first_m3:
        P(f"process-warmup record {pw_last} comes after the first block sample {first_m3}")
    for (b, r), pos in positions.items():
        if len(set(pos)) != len(pos) or not set(pos) <= set(range(len(spec["combos"]))):
            P(f"block {b} rep {r}: order positions {pos} are not distinct positions 0..{len(spec['combos']) - 1}")

    # ---- per-sample validity (M2/M3 rules) and the fields the analysis reads
    n_rule, n_field = 0, 0
    for key, s in measured.items():
        rule = _sample_problems(s, spec["num_layers"])
        mem = s.get("mem") if isinstance(s.get("mem"), dict) else {}
        field = [f for f, ok in (("t_wall", isinstance(s.get("t_wall"), (int, float)) and math.isfinite(s["t_wall"])),
                                 *((f"mem.{k}", isinstance(mem.get(k), int))
                                   for k in ("allocated_before", "peak_allocated", "peak_reserved"))) if not ok]
        n_rule += bool(rule)
        n_field += bool(field)
        if (rule or field) and n_rule + n_field <= MAX_LISTED:
            P(f"sample {key} fails the validity rules, so its pair is incomplete: {rule + field}")
        counts["measured_samples"]["present"] += 1
        counts["measured_samples"]["passed"] += not rule and not field
    if n_rule + n_field > MAX_LISTED:
        P(f"{n_rule} samples fail the M2/M3 validity rules and {n_field} lack analysis fields in total")
    stored = (meta.get("n_samples"), meta.get("n_measured"), meta.get("n_excluded"), len(rep["excluded"]))
    if stored != (len(rep["samples"]), n_not_warmup, n_rule, n_rule):
        P(f"stored n_samples/n_measured/n_excluded/len(excluded) {stored} != recomputed "
          f"{(len(rep['samples']), n_not_warmup, n_rule, n_rule)}")

    # ---- correctness: one record per spec case, both backends, detailed fields re-derived
    corr_ok = _check_records(
        rep["correctness"], cases, "correctness", P,
        lambda c: _case((c.get(spec["backends"][0]) or {}).get("L"), (c.get(spec["backends"][0]) or {}).get("S")),
        lambda c, case: _check_correctness(c, case, spec, P, fail, sid, counts["correctness_checks"]))
    # ---- A/B: one record per spec case, both tiers
    ab_ok = _check_records(rep["ab_checks"], cases, "A/B", P, lambda a: _case(a.get("L"), a.get("S")),
                           lambda a, case: _check_ab(a, case, spec, P, fail, sid, counts["ab_checks"]))
    # ---- restore-only memory: one record per spec case, every tier x backend a full restore
    _check_records(rep["restore_memory"], cases, "restore_memory", P, lambda m: _case(m.get("L"), m.get("S")),
                   lambda m, case: _check_restore_memory(m, case, spec, P))
    session_ok = corr_ok and ab_ok
    if meta.get("correctness_all") is not session_ok:
        P(f"stored correctness_all={meta.get('correctness_all')!r}, but the detailed correctness and A/B "
          f"records give {session_ok}")
    return session_ok


def _check_records(records, cases, what, P, case_of, check) -> bool:
    """Exactly one record per spec case; `check(record, case)` -> passed. True only if every
    spec case has exactly one record and every one passed."""
    if not isinstance(records, list):
        P(f"{what} records are not a list")
        return False
    seen, ok = Counter(), True
    for j, rec in enumerate(records):
        case = case_of(rec) if isinstance(rec, dict) else None
        if case not in cases:
            P(f"{what}[{j}]: case {case} is not a spec case")
            ok = False
            continue
        seen[case] += 1
        if seen[case] > 1:
            P(f"duplicate {what} record for case {case}")
            ok = False
            continue
        ok &= bool(check(rec, case))
    for case in cases:
        if not seen[case]:
            P(f"no {what} record for case {case}")
            ok = False
    return ok


def _check_correctness(c, case, spec, P, fail, sid, count) -> bool:
    L, n, ok_all = case[0], case[0] // C, True
    for b in spec["backends"]:
        e = c.get(b)
        missing = [f for f in CORRECTNESS_FIELDS if not isinstance(e, dict) or f not in e]
        if missing:
            P(f"correctness {case} {b}: missing detailed fields {missing}")
            ok_all = False
            continue
        malformed = [f for f in ("strict_host_eq_disk_bitwise", "strict_host_eq_resident_bitwise", "strict_pass")
                     if not isinstance(e[f], bool)]
        if (e["L"], e["S"]) != case or malformed or not isinstance(e["restored_kv_bytes_equal"], dict) or \
                set(e["restored_kv_bytes_equal"]) != set(spec["tiers"]) or \
                not isinstance(e["report_only_full_vs_restored"], dict):
            P(f"correctness {case} {b}: malformed record (case {(e['L'], e['S'])}, non-boolean {malformed}, "
              f"restored_kv_bytes_equal {e['restored_kv_bytes_equal']!r})")
            ok_all = False
            continue
        det = correctness_detail(e, L, n)
        ok = all(det.values())
        count["present"] += 1
        count["passed"] += ok
        if not ok:
            fail.append(f"{sid}: correctness {case} {b}: failed {[k for k, v in det.items() if not v]}")
        if e["strict_pass"] is not ok:
            P(f"correctness {case} {b}: stored strict_pass={e['strict_pass']!r} disagrees with the detailed checks ({ok})")
        ok_all &= ok
    if c.get("strict_pass_all") is not ok_all:
        P(f"correctness {case}: stored strict_pass_all={c.get('strict_pass_all')!r}, detailed checks give {ok_all}")
    return ok_all


def _check_ab(a, case, spec, P, fail, sid, count) -> bool:
    L, n, ok_all = case[0], case[0] // C, True
    for tier in spec["tiers"]:
        t = a.get(tier)
        missing = [f for f in (*AB_FIELDS, "restore_len", "sources") if not isinstance(t, dict) or f not in t]
        if missing:
            P(f"A/B {case} {tier}: missing detailed fields {missing}")
            ok_all = False
            continue
        malformed = [f for f in AB_FIELDS if not isinstance(t[f], bool)]
        if malformed:
            P(f"A/B {case} {tier}: non-boolean fields {malformed}")
            ok_all = False
            continue
        det = ab_detail(t, tier, L, n)
        ok = all(det.values())
        count["present"] += 1
        count["passed"] += ok
        if not ok:
            fail.append(f"{sid}: A/B {case} {tier}: failed {[k for k, v in det.items() if not v]}")
        ok_all &= ok
    if a.get("pass") is not ok_all:
        P(f"A/B {case}: stored pass={a.get('pass')!r}, detailed checks give {ok_all}")
    return ok_all


def _check_restore_memory(m, case, spec, P) -> bool:
    n = case[0] // C
    for tier in spec["tiers"]:
        for b in spec["backends"]:
            r = m.get(f"{tier}_{b}")
            lr = r.get("last_restore") if isinstance(r, dict) and isinstance(r.get("last_restore"), dict) else {}
            if not isinstance(r, dict) or r.get("L") != case[0] or r.get("sources_ok") is not True or \
                    not isinstance(r.get("peak_above_start"), int) or not isinstance(r.get("kept_after_return"), int) or \
                    lr.get("backend") != b or lr.get("chunks_loaded") != n:
                P(f"restore_memory {case} {tier}_{b}: missing, incomplete or not a full {b} restore from {tier}")
    return True


def _check_profile(rep, spec, P):
    ps = spec["profile_session"]
    missing = [k for k in ("meta", "source", "failures", "profile") if k not in rep]
    if missing:
        P(f"missing sections {missing}")
        return
    if rep["failures"] != [] or not isinstance(rep["meta"], dict) or rep["meta"].get("n_excluded") != 0:
        P(f"failures {rep['failures']!r} / n_excluded {(rep['meta'] or {}).get('n_excluded')!r}")
    if (rep["source"] or {}).get("git_head") != spec["git_head"] or "git_diff_head_sha256" not in rep["source"]:
        P(f"source identity {(rep['source'] or {}).get('git_head')!r}, spec {spec['git_head']}")
    rows = rep["profile"] if isinstance(rep["profile"], list) else []
    keys = Counter((r.get("L"), r.get("S"), r.get("path"), r.get("backend"), r.get("unit")) for r in rows
                   if isinstance(r, dict) and isinstance(r.get("api_per_call"), dict)
                   and isinstance(r.get("device_ops_per_call"), dict))
    want = Counter((L, S, t, b, u) for L, S in ps["cases"] for t in spec["tiers"] for b in spec["backends"]
                   for u in ps["units"])
    if keys != want or len(rows) != sum(want.values()):
        P(f"profile rows {sorted(keys)} differ from the planned {len(want)} (case, tier, backend, unit) rows")


def load_and_validate(run_dir, spec=FROZEN_M3_EVAL):
    """Validate a run directory against `spec`. Returns (report, sessions, profile):
    report["status"] is "valid", "invalid" or "correctness_failed" (see the module docstring);
    sessions maps each loaded evaluation session id (spec order) to its parsed JSON; profile is
    the parsed profile session or None."""
    rd = Path(run_dir)
    problems, fail, counts = [], [], _new_counts(spec)
    sessions, prof, per_session = {}, None, {}
    found = {p.stem.removeprefix("session_"): p for p in sorted(rd.glob("session_*.json"))}
    prof_id = spec["profile_session"]["id"]
    for sid in spec["sessions"]:
        if sid not in found:
            problems.append(f"missing session file session_{sid}.json (spec {spec['name']}: {list(spec['sessions'])})")
    for sid in sorted(set(found) - set(spec["sessions"]) - {prof_id}):
        problems.append(f"unexpected session file {found[sid].name} (not in spec {spec['name']})")
    for sid in (s for s in spec["sessions"] if s in found):
        try:
            rep = json.loads(found[sid].read_text())
        except (OSError, ValueError) as e:
            problems.append(f"{sid}: unreadable session file: {e}")
            continue
        if not isinstance(rep, dict):
            problems.append(f"{sid}: session file is not a JSON object")
            continue
        sessions[sid] = rep
        n0 = len(problems)
        try:
            ok = _check_session(sid, rep, spec, lambda m, sid=sid: problems.append(f"{sid}: {m}"), fail, counts)
        except (AttributeError, KeyError, TypeError, ValueError) as e:  # malformed nested record: fail closed
            problems.append(f"{sid}: malformed session record: {e!r}")
            ok = False
        per_session[sid] = {"problems": len(problems) - n0, "correctness_and_ab_pass_recomputed": ok}
    if prof_id in found:
        try:
            prof = json.loads(found[prof_id].read_text())
            _check_profile(prof, spec, lambda m: problems.append(f"{prof_id}: {m}"))
        except (OSError, ValueError, AttributeError, TypeError) as e:
            problems.append(f"{prof_id}: unreadable or malformed profile session: {e!r}")
            prof = None
    # source identity: every file recorded by all sessions has identical bytes in all of them,
    # and the files that executed the measurement are among them
    recs = {sid: (r.get("source") or {}).get("files_sha256") for sid, r in
            [*sessions.items(), *([(prof_id, prof)] if isinstance(prof, dict) else [])]}
    if recs and all(isinstance(v, dict) for v in recs.values()):
        common = set.intersection(*(set(v) for v in recs.values()))
        for f in sorted(common):
            if len({v[f] for v in recs.values()}) > 1:
                problems.append(f"source file {f} differs between sessions")
        for f in spec["executed_sources"]:
            if f not in common:
                problems.append(f"executed source {f} is not recorded by every session")
    elif recs:
        problems.append("a session lacks source.files_sha256")

    complete = all(c["present"] == c["passed"] == c["expected"] for c in counts.values())
    status = "invalid" if problems or (not fail and not complete) else "correctness_failed" if fail else "valid"
    acceptance = status == "valid" and complete
    report = {"spec": spec["name"], "spec_source": spec["source"], "run_dir": str(rd), "status": status,
              "problems": problems, "correctness_failures": fail,
              "sessions_expected": list(spec["sessions"]), "session_files_found": sorted(found),
              "profile_session_present": prof is not None, "per_session": per_session, "counts": counts,
              "correctness_and_ab_acceptance_recomputed": acceptance}
    return report, sessions, prof
