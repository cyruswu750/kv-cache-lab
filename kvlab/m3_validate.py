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

Specs:
- `FROZEN_M3_EVAL`: the M3 evaluation, unchanged. Its provenance is the Git HEAD in the spec.
- `M3_CONFIRM_V1` (docs/M3_CONFIRM_PROTOCOL.md): the preregistered confirmation. It also
  requires:
  - the exact sample sequence the runner draws from each seed;
  - a match with the independently frozen manifest: base HEAD, diff, file bytes, commands and
    stack (kvlab/m3_confirm.py);
  - a finalized batch record whose session-file hashes match: every planned session launched,
    none killed, none launched after a nonzero exit, all ended before the deadline, and no stop
    other than a nonzero exit. A nonzero exit itself is not an evidence problem: the reports
    are judged on their contents;
  - a closed file listing of the run directory, which also rejects any profiler evidence;
  - the analysis files still having their frozen bytes.

No model weights and no CUDA are needed. Only the per-sample M2/M3 rules
(`kvlab.m3_bench.sample_problems`) are imported, lazily; that imports torch but does not
initialise CUDA or load the model.
"""

import json
import random
from collections import Counter
from pathlib import Path

from kvlab import m3_confirm as mc
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
# M3_CONFIRM_V1 (docs/M3_CONFIRM_PROTOCOL.md): the preregistered confirmation of (1024, 1) plus one
# fresh 5-chunk case. The runner and its ordering semantics are the M3 ones, unchanged. Unlike the
# frozen M3 spec, the provenance (base HEAD, diff, file bytes, stack, commands) is not written here:
# it comes from the independently frozen manifest (kvlab/m3_confirm.py), which every session file
# and the batch record must match. Keys absent from FROZEN_M3_EVAL switch on the stricter checks
# below; FROZEN_M3_EVAL is unchanged. The one shared change: a sample's t_wall must convert to a
# finite float (mc.finite), so JSON true and an out-of-range integer are malformed (decisions D31).
M3_CONFIRM_V1 = {
    "name": "M3_CONFIRM_V1",
    "source": "docs/M3_CONFIRM_PROTOCOL.md + the freeze manifest results/m3_confirm/M3_CONFIRM_V1.freeze.json",
    "git_head": None,  # from the freeze manifest
    "provenance": "freeze_manifest",
    "freeze_manifest": "results/m3_confirm/M3_CONFIRM_V1.freeze.json",
    "run_dir": "results/m3_confirm/M3_CONFIRM_V1_run",
    "python": ".venv/bin/python",
    "session_env": {"HF_HUB_OFFLINE": "1"},
    # the process environment every M3 evaluation session recorded (diag_harness.process_record)
    "required_process_env": {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None, "CUDA_LAUNCH_BLOCKING": None,
                             "PYTORCH_CUDA_ALLOC_CONF": None, "CUDA_MODULE_LOADING": None, "HF_HUB_OFFLINE": "1"},
    "budget_s": 1800,  # whole batch, monotonic, from launching the first session process
    "sessions": {"c1": 32001, "c2": 32002, "c3": 32003, "c4": 32004, "c5": 32005},
    "cases": ((1024, 1), (1280, 96)),
    "blocks": 4,
    "warmup": 2,
    # the runner's process warmup, sorted({min(cases), max(cases)}): with two cases, both of them
    "process_warmup": {"cases": ((1024, 1), (1280, 96)), "repeats": 2},
    "reps": 24,
    "staging_chunks": 4,
    "num_layers": 28,
    "tiers": FROZEN_M3_EVAL["tiers"],
    "backends": FROZEN_M3_EVAL["backends"],
    "combos": FROZEN_M3_EVAL["combos"],
    "executed_sources": FROZEN_M3_EVAL["executed_sources"],
    "profile_session": None,      # no profiler run: any profiler evidence is rejected
    "exact_sequence": True,       # every sample identity, in file order, replayed from the seed
    "count_restore_memory": True,
    "decision_rule": "confirm_all_primary",
    "primary_statistics": ("median_d", "mean_d"),
    "required_model": {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                       "dtype": "torch.bfloat16", "attn_implementation": "sdpa", "num_hidden_layers": 28},
    # the stack recorded by every M3 evaluation session (results/m3/eval_20261103T173632)
    "required_stack": {"python": "3.14.4", "torch": "2.14.1+cu130", "transformers": "5.22.1", "cuda": "13.0",
                       "gpu_name": "NVIDIA GeForce RTX 4080 Laptop GPU", "driver_version": "596.49",
                       "weights_blob": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
                       "weights_bytes": 3087467144},
    "required_packages": {
        "huggingface_hub": "2.4.0", "numpy": "2.5.4", "nvidia-cublas": "13.1.1.3", "nvidia-cuda-cupti": "13.0.85",
        "nvidia-cuda-nvrtc": "13.0.88", "nvidia-cuda-runtime": "13.0.96", "nvidia-cudnn-cu13": "9.24.0.43",
        "nvidia-cufft": "12.0.0.61", "nvidia-cufile": "1.15.1.6", "nvidia-curand": "10.4.0.35",
        "nvidia-cusolver": "12.0.4.66", "nvidia-cusparse": "12.6.3.3", "nvidia-cusparselt-cu13": "0.8.1",
        "nvidia-nccl-cu13": "2.30.7", "nvidia-nvjitlink": "13.4.92", "nvidia-nvshmem-cu13": "3.4.5",
        "nvidia-nvtx": "13.0.85", "safetensors": "0.8.0", "tokenizers": "0.23.3", "torch": "2.14.1+cu130",
        "transformers": "5.22.1", "triton": "3.8.0"},
}
SPECS = {FROZEN_M3_EVAL["name"]: FROZEN_M3_EVAL, M3_CONFIRM_V1["name"]: M3_CONFIRM_V1}
DEFAULT_SPEC = FROZEN_M3_EVAL["name"]
ROOT = Path(__file__).resolve().parent.parent

SESSION_SECTIONS = ("meta", "source", "restore_config", "samples", "correctness", "ab_checks",
                    "restore_memory", "failures", "blocks", "excluded")
# Bench.correctness (kvlab/perf_harness.py) fields; strict_pass is re-derived from the details
CORRECTNESS_FIELDS = ("L", "S", "host_restore_len", "host_sources", "disk_restore_len", "disk_sources",
                      "restored_kv_bytes_equal", "strict_host_eq_disk_bitwise",
                      "strict_host_eq_resident_bitwise", "strict_pass", "report_only_full_vs_restored")
# M3Bench.ab_check (kvlab/m3_bench.py) boolean fields per tier
AB_FIELDS = ("restore_len_sources_admission_equal", "kv_bytes_equal_after_prefill",
             "first_logits_bitwise_equal", "greedy_tokens_equal", "step_logits_bitwise_equal")
# extra sections a session must have when its provenance is checked against a freeze manifest
PROVENANCE_SECTIONS = ("environment", "packages", "weights", "model", "process")
MAX_LISTED = 5  # sample-level problems listed individually per session; the rest are counted


def _sample_problems(s: dict, num_layers: int) -> list[str]:
    from kvlab import m3_bench as mb  # lazy: torch import only (no CUDA init, no weights)
    return mb.sample_problems(s, num_layers)


def _warmup_field_problems(s: dict) -> list[str]:
    """Fields a kept warmup record must have; the analysis lists every warmup in
    samples_chronological.csv but never uses its timing."""
    ph = s.get("phases_ns")
    checks = (("t_wall", mc.finite(s.get("t_wall"))),
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
    if spec.get("count_restore_memory"):  # one record per session x case, every tier x backend
        exp["restore_memory_records"] = n_s * n_c
    return {k: {"expected": v, "present": 0, "passed": 0} for k, v in exp.items()}


def _check_session(sid, rep, spec, P, fail, counts, prov=None):
    """Append problems (P) and failed checks (fail) for one evaluation session. Returns the
    recomputed per-session acceptance of its correctness and A/B checks. `prov` (specs with a
    freeze manifest) holds the manifest the session must match."""
    missing = [k for k in SESSION_SECTIONS + (PROVENANCE_SECTIONS if prov else ()) if k not in rep]
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
    want_head = prov["manifest"].get("base_git_head") if prov else spec["git_head"]
    if head != want_head:
        P(f"git_head {head!r}, {'frozen base' if prov else 'spec'} {want_head}")
    if prov:
        _check_provenance(sid, rep, spec, prov, P)
    if spec["profile_session"] is None and ("profile" in rep or args.get("profile") is not False):
        P("unexpected profiler evidence (this experiment plans no profiler run)")
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
    if spec.get("exact_sequence"):
        _check_sequence(rep, spec, sid, order, P)

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
        field = [f for f, ok in (("t_wall", mc.finite(s.get("t_wall"))),
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
    mem_count = counts.get("restore_memory_records")

    def mem_check(m, case):
        ok = _check_restore_memory(m, case, spec, P)
        if mem_count is not None:
            mem_count["present"] += 1
            mem_count["passed"] += ok
        return ok
    _check_records(rep["restore_memory"], cases, "restore_memory", P, lambda m: _case(m.get("L"), m.get("S")),
                   mem_check)
    session_ok = corr_ok and ab_ok
    if meta.get("correctness_all") is not session_ok:
        P(f"stored correctness_all={meta.get('correctness_all')!r}, but the detailed correctness and A/B "
          f"records give {session_ok}")
    return session_ok


def expected_sequence(spec, sid):
    """The runner's sample order (scripts/m3_measure.py), replayed from the session seed:
    (block order, [(stage, L, S, block, rep, order_pos, (path, backend, mode))] in file order).
    The runner draws, from one random.Random(seed): one shuffle of the cases (repeated `blocks`
    times), then one shuffle of the combinations per repetition. The process warmup runs every
    combination twice, in COMBOS order, at sorted({min(cases), max(cases)}), before block 0."""
    rng = random.Random(spec["sessions"][sid])
    cases = [tuple(c) for c in spec["cases"]]  # the --cases order of the frozen command
    combos = [tuple(c) for c in spec["combos"]]
    order = cases[:]
    rng.shuffle(order)
    block_cases = [c for _ in range(spec["blocks"]) for c in order]
    seq = [("process_warmup", L, S, None, None, None, combo)
           for L, S in sorted({min(cases), max(cases)}) for combo in combos for _ in range(2)]
    for bi, (L, S) in enumerate(block_cases):
        for r in range(spec["warmup"] + spec["reps"]):
            cs = list(combos)
            rng.shuffle(cs)
            seq += [("m3", L, S, bi, r, pos, combo) for pos, combo in enumerate(cs)]
    return block_cases, seq


def _check_sequence(rep, spec, sid, order, P):
    """Exact identities and order: every sample, in file order, is the one the runner draws at
    that position for this seed; recorded wall times never go backwards; the per-case checks
    follow the first pass of the block order."""
    block_cases, seq = expected_sequence(spec, sid)
    if order != block_cases:
        P(f"block_order {order} is not the runner's order for seed {spec['sessions'][sid]}: {block_cases}")
    got = [(s.get("stage"), s.get("L"), s.get("S"), s.get("block"), s.get("rep"), s.get("order_pos"),
            (s.get("path"), s.get("backend"), s.get("mode"))) if isinstance(s, dict) else None for s in rep["samples"]]
    if got != seq:
        i = next((i for i, (a, b) in enumerate(zip(got, seq)) if a != b), min(len(got), len(seq)))
        P(f"sample sequence differs from the runner's order for seed {spec['sessions'][sid]} at sample {i} "
          f"(found {got[i] if i < len(got) else 'end'}, expected {seq[i] if i < len(seq) else 'end'}; "
          f"{len(got)} samples, {len(seq)} planned)")
    tw = [s.get("t_wall") if isinstance(s, dict) else None for s in rep["samples"]]
    if not all(isinstance(x, (int, float)) for x in tw) or any(b < a for a, b in zip(tw, tw[1:])):
        P("sample wall times are missing or not in file order")
    first = block_cases[:len(spec["cases"])]

    def corr_case(c):
        e = c.get(spec["backends"][0])
        return _case(e.get("L"), e.get("S")) if isinstance(e, dict) else None
    for what, recs, case_of in (("correctness", rep["correctness"], corr_case),
                                ("A/B", rep["ab_checks"], lambda a: _case(a.get("L"), a.get("S"))),
                                ("restore_memory", rep["restore_memory"], lambda m: _case(m.get("L"), m.get("S")))):
        got_cases = [case_of(x) if isinstance(x, dict) else None for x in recs] if isinstance(recs, list) else None
        if got_cases != first:
            P(f"{what} records are for cases {got_cases}, not the first pass of the block order {first}")


def _check_provenance(sid, rep, spec, prov, P):
    """The session ran the frozen command, from the frozen tree, on the required stack."""
    m = prov["manifest"]
    src = rep["source"] if isinstance(rep["source"], dict) else {}
    if src.get("git_diff_head_sha256") != m.get("git_diff_head_sha256"):
        P("git diff HEAD differs from the frozen diff")
    files, frozen = src.get("files_sha256"), m.get("files_sha256")
    if files != frozen:
        files, frozen = files if isinstance(files, dict) else {}, frozen if isinstance(frozen, dict) else {}
        P(f"source files differ from the freeze: {sorted(f for f in set(files) | set(frozen) if files.get(f) != frozen.get(f))}")
    args = rep["meta"].get("args")
    if args != mc.expected_args(spec, sid):
        P(f"recorded arguments {args!r} differ from the frozen command {mc.expected_args(spec, sid)!r}")
    env = rep["environment"] if isinstance(rep["environment"], dict) else {}
    req, stack = spec["required_stack"], m.get("stack") or {}
    for k in ("torch", "transformers", "cuda", "python"):
        if env.get(k) != req[k]:
            P(f"environment {k} {env.get(k)!r}, required {req[k]!r}")
    gs = mc.parse_gpu_static(env.get("gpu_static"))
    if env.get("gpu") != req["gpu_name"] or gs != {k: stack.get(k) for k in ("gpu_name", "driver_version")}:
        P(f"GPU {env.get('gpu')!r} / {gs}, frozen {stack.get('gpu_name')!r} driver {stack.get('driver_version')!r}")
    for k in ("kernel", "cpu"):
        if env.get(k) != stack.get(k):
            P(f"environment {k} {env.get(k)!r}, frozen {stack.get(k)!r}")
    if rep["packages"] != stack.get("packages") or rep["packages"] != spec["required_packages"]:
        P("package versions differ from the frozen and required stack")
    if rep["weights"] != stack.get("weights"):
        P("model weight files differ from the freeze")
    model = rep["model"] if isinstance(rep["model"], dict) else {}
    if {k: model.get(k) for k in spec["required_model"]} != spec["required_model"]:
        P(f"model {({k: model.get(k) for k in spec['required_model']})}, required {spec['required_model']}")
    penv = (rep["process"] or {}).get("env") if isinstance(rep["process"], dict) else None
    if penv != spec["required_process_env"]:
        P(f"process environment {penv!r}, required {spec['required_process_env']}")
    started = rep["meta"].get("started_unix")
    launched = (prov.get("batch") or {}).get("launched_unix") if isinstance(prov.get("batch"), dict) else None
    fr = m.get("frozen_unix")
    if not mc.finite(started) or not mc.finite(fr) or started < fr or not mc.finite(launched) or started < launched:
        P(f"session started at {started!r}, not after the freeze ({fr!r}) and the batch launch ({launched!r})")


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
    n, ok = case[0] // C, True
    for tier in spec["tiers"]:
        for b in spec["backends"]:
            r = m.get(f"{tier}_{b}")
            lr = r.get("last_restore") if isinstance(r, dict) and isinstance(r.get("last_restore"), dict) else {}
            if not isinstance(r, dict) or r.get("L") != case[0] or r.get("sources_ok") is not True or \
                    not isinstance(r.get("peak_above_start"), int) or not isinstance(r.get("kept_after_return"), int) or \
                    lr.get("backend") != b or lr.get("chunks_loaded") != n:
                P(f"restore_memory {case} {tier}_{b}: missing, incomplete or not a full {b} restore from {tier}")
                ok = False
    return ok


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


def _load_provenance(rd, spec, freeze, root, expect_sha, problems):
    """Freeze manifest + batch record + closed file listing of the run directory."""
    path = Path(root) / spec["freeze_manifest"] if freeze is None else Path(freeze)
    info = {"freeze_manifest": str(path), "freeze_manifest_sha256": None, "batch_record_sha256": None}
    try:
        data = path.read_bytes()
        manifest = mc.load_json_finite(data)  # NaN/Infinity would defeat every time comparison
        info["freeze_manifest_sha256"] = mc.sha256_bytes(data)
    except (OSError, ValueError) as e:
        problems.append(f"freeze manifest {path} missing or unreadable: {e}")
        return None, info
    if expect_sha is not None and info["freeze_manifest_sha256"] != expect_sha:
        problems.append(f"freeze manifest sha256 {info['freeze_manifest_sha256']} != the expected {expect_sha}")
    try:
        problems += mc.manifest_problems(manifest, spec, root)
    except (AttributeError, KeyError, TypeError, ValueError) as e:  # malformed field: fail closed
        problems.append(f"malformed freeze manifest: {e!r}")
    try:
        data = (rd / mc.BATCH_RECORD).read_bytes()
        batch = mc.load_json_finite(data)
        info["batch_record_sha256"] = mc.sha256_bytes(data)
    except (OSError, ValueError) as e:
        problems.append(f"batch record {mc.BATCH_RECORD} missing or unreadable: {e}")
        batch = None
    if batch is not None:
        try:
            problems += mc.batch_problems(batch, manifest if isinstance(manifest, dict) else {},
                                          info["freeze_manifest_sha256"], rd, spec)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            problems.append(f"malformed batch record: {e!r}")
        if isinstance(batch, dict):  # execution, reported as recorded; the evidence is judged separately
            info["execution"] = {
                "execution_status": batch.get("execution_status"), "stop": batch.get("stop"),
                "not_launched": batch.get("not_launched"),
                "returncodes": {e.get("session"): e.get("returncode") for e in batch.get("sessions") or []
                                if isinstance(e, dict)}}
    present = {p.relative_to(rd).as_posix() for p in rd.rglob("*") if not p.is_dir()}
    want = mc.run_dir_files(spec)
    for f in sorted(present - want):
        problems.append(f"unexpected file in the run directory: {f}")
    for f in sorted(want - present - {f"session_{s}.json" for s in spec["sessions"]} - {mc.BATCH_RECORD}):
        problems.append(f"missing file in the run directory: {f}")
    return {"manifest": manifest if isinstance(manifest, dict) else {}, "batch": batch}, info


def _check_batch_sessions(sessions, prov, spec, problems):
    """Sequential fresh processes; each exit code is the runner's own verdict on its record."""
    entries = {e.get("session"): e for e in (prov["batch"] or {}).get("sessions") or [] if isinstance(e, dict)}
    prev, pids = None, set()
    for sid in spec["sessions"]:
        rep = sessions.get(sid)
        if not isinstance(rep, dict) or not isinstance(rep.get("meta"), dict):
            prev = None
            continue
        m = rep["meta"]
        pid = (rep.get("process") or {}).get("pid") if isinstance(rep.get("process"), dict) else None
        if pid in pids or pid is None:
            problems.append(f"{sid}: process id {pid!r} is missing or repeats an earlier session")
        elif pid != (entries.get(sid) or {}).get("pid"):
            problems.append(f"{sid}: process id {pid} is not the process the batch launched "
                            f"({(entries.get(sid) or {}).get('pid')!r})")
        pids.add(pid)
        times = [s.get("t_wall") for s in rep.get("samples") or [] if isinstance(s, dict)]
        if prev is not None and (not mc.finite(m.get("started_unix")) or m["started_unix"] <= prev):
            problems.append(f"{sid}: started at {m.get('started_unix')!r}, before the previous session's last sample {prev}")
        prev = max((x for x in times if mc.finite(x)), default=None)
        ok_run = m.get("correctness_all") is True and rep.get("failures") == [] and m.get("n_excluded") == 0
        rc = (entries.get(sid) or {}).get("returncode")
        if rc != (0 if ok_run else 1):
            problems.append(f"{sid}: exit code {rc!r} disagrees with the session record (runner success {ok_run})")


def load_and_validate(run_dir, spec=FROZEN_M3_EVAL, *, freeze=None, root=ROOT, expect_freeze_sha256=None):
    """Validate a run directory against `spec`. Returns (report, sessions, profile):
    report["status"] is "valid", "invalid" or "correctness_failed" (see the module docstring);
    sessions maps each loaded evaluation session id (spec order) to its parsed JSON; profile is
    the parsed profile session or None. For a spec with a freeze manifest, `freeze` overrides
    the manifest path (default: the spec's, under `root`), and `expect_freeze_sha256`, if
    given, must be the manifest's SHA-256 as recorded out of band at freeze time."""
    rd = Path(run_dir)
    problems, fail, counts = [], [], _new_counts(spec)
    sessions, prof, per_session = {}, None, {}
    prov, prov_info = None, None
    if spec.get("provenance") == "freeze_manifest":
        prov, prov_info = _load_provenance(rd, spec, freeze, root, expect_freeze_sha256, problems)
        if prov is None:
            prov = {"manifest": {}, "batch": None}
    found = {p.stem.removeprefix("session_"): p for p in sorted(rd.glob("session_*.json"))}
    prof_id = spec["profile_session"]["id"] if spec["profile_session"] else None
    for sid in spec["sessions"]:
        if sid not in found:
            problems.append(f"missing session file session_{sid}.json (spec {spec['name']}: {list(spec['sessions'])})")
    for sid in sorted(set(found) - set(spec["sessions"]) - {prof_id}):
        problems.append(f"unexpected session file {found[sid].name} (not in spec {spec['name']}"
                        f"{'; this experiment plans no profiler session' if prof_id is None else ''})")
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
            ok = _check_session(sid, rep, spec, lambda m, sid=sid: problems.append(f"{sid}: {m}"), fail, counts, prov)
        except (AttributeError, KeyError, TypeError, ValueError) as e:  # malformed nested record: fail closed
            problems.append(f"{sid}: malformed session record: {e!r}")
            ok = False
        per_session[sid] = {"problems": len(problems) - n0, "correctness_and_ab_pass_recomputed": ok}
    if prov is not None:
        try:
            _check_batch_sessions(sessions, prov, spec, problems)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            problems.append(f"malformed session or batch record: {e!r}")
    if prof_id is not None and prof_id in found:
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
    # Status precedence: invalid (exit 3) > correctness_failed (exit 4) > valid. Detailed
    # correctness/A-B failures found in any parseable session are listed in
    # `correctness_failures` whatever the status, so an invalid or incomplete batch never hides them.
    status = "invalid" if problems or (not fail and not complete) else "correctness_failed" if fail else "valid"
    acceptance = status == "valid" and complete
    report = {"spec": spec["name"], "spec_source": spec["source"], "run_dir": str(rd), "status": status,
              "problems": problems, "correctness_failures": fail,
              "sessions_expected": list(spec["sessions"]), "session_files_found": sorted(found),
              "profile_session_present": prof is not None, "per_session": per_session, "counts": counts,
              "correctness_and_ab_acceptance_recomputed": acceptance}
    if prov_info is not None:
        report["provenance"] = prov_info
        report["correctness_failures_observed"] = bool(fail)
    return report, sessions, prof
