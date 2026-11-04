"""M1 evidence run: synchronous always-restore over a tiered (pinned host + disk) store.

Usage: .venv/bin/python scripts/m1_run.py --out results/m1_1/m1_run_<ts>.json
       .venv/bin/python scripts/m1_run.py --validate <report.json>   (re-check a saved report; no model)

Scenarios (fresh store each, tiny budgets so eviction happens):
  sequence   9 requests with shared prefixes; host 2 slots, disk 2 files
  faults     corrupt middle chunk, missing file, incompatible namespace,
             incompatible layout, ENOSPC on spill
Every request (including ENOSPC) goes through Run.request and gets one record:
restore length, sources, miss reason, stored chunks, model forward calls during
the engine request only (expected STEPS), and STRICT checks against a
resident-GPU replay (token/step-logit counts == STEPS, then exact tokens,
bitwise step logits, byte-identical restored K/V). Full recomputation is a
REPORT-ONLY comparison (M0 D7 metrics), never part of acceptance (D10).
Acceptance (exit 0) = validate_report(): all expected scenarios and records
present, every record strict, the intended fault observed in each fault scenario.
Disk payloads live in .kvcache/m1_run/ (git-ignored) and are deleted at the end.
No timings are recorded (performance calibration deferred).
"""

import argparse
import dataclasses
import errno
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402  (sets HF_HOME before transformers import)
import torch  # noqa: E402
import transformers  # noqa: E402
from kvlab import chunk_format as cf  # noqa: E402
from kvlab.always_restore import AlwaysRestoreEngine  # noqa: E402
from kvlab.disk_store import DiskStore  # noqa: E402
from kvlab.fixtures import BOUNDARY_TEXT, CALIBRATION_TEXT, exact_length_ids  # noqa: E402
from kvlab.host_store import HostStore  # noqa: E402
from kvlab.identity import ChunkIdentity, chain_keys, namespace_for  # noqa: E402
from kvlab.metrics import compare_logits, greedy_agreement  # noqa: E402
from kvlab.resident_reference import ResidentReference  # noqa: E402
from kvlab.tiered_cache import TieredKVCache  # noqa: E402
from kvlab.tolerances import ATOL, RTOL  # noqa: E402

C, STEPS = 256, 8
DISK_ROOT = km.PROJECT_ROOT / ".kvcache" / "m1_run"


# ---- strict per-request checks and report validation (acceptance) ---------------------------

STRICT_FLAGS = ("lengths_ok", "strict_tokens_equal_resident", "strict_all_step_logits_bitwise_equal_resident",
                "all_logits_finite", "restored_kv_ok", "forward_calls_ok")

# The evidence run's fixed shape: scenario -> number of request records.
EXPECTED_RECORDS = {"sequence": 9, "fault_corrupt_middle": 3, "fault_missing_file": 2,
                    "fault_incompatible_namespace": 2, "fault_incompatible_layout": 2, "fault_enospc": 1}
# Fault scenarios detected by a miss: (record index after the fault, expected miss reason).
EXPECTED_FAULT_MISS = {"fault_corrupt_middle": (1, "disk_payload_crc"),
                       "fault_missing_file": (1, "disk_file_missing"),
                       "fault_incompatible_namespace": (1, "disk_namespace_mismatch:attn_implementation"),
                       "fault_incompatible_layout": (1, "disk_layout_mismatch:num_layers")}


def strict_checks(tokens, step_logits, ref_tokens, ref_steps, restore_len, kv_bytes, forward_calls) -> dict:
    """Strict acceptance fields for one request. Sequence lengths are checked against STEPS
    first, so zip() can never hide a truncated sequence; then exact token equality,
    bitwise step-logit equality (torch.equal also checks shape and dtype), and, if
    anything was restored, the byte check of the restored K/V."""
    lengths = {"num_tokens": len(tokens), "num_step_logits": len(step_logits),
               "num_ref_tokens": len(ref_tokens), "num_ref_step_logits": len(ref_steps)}
    lengths_ok = all(n == STEPS for n in lengths.values())
    out = {**lengths, "expected_steps": STEPS, "lengths_ok": lengths_ok,
           "strict_tokens_equal_resident": lengths_ok and list(tokens) == list(ref_tokens),
           "strict_all_step_logits_bitwise_equal_resident":
               lengths_ok and all(torch.equal(a, b) for a, b in zip(step_logits, ref_steps)),
           "all_logits_finite": lengths_ok and all(bool(torch.isfinite(x).all()) for x in step_logits),
           "strict_restored_kv_bytes": kv_bytes or None,
           "restored_kv_ok": restore_len == 0 or bool(kv_bytes and kv_bytes.get("all_bytes_equal")),
           "forward_calls": forward_calls, "expected_forward_calls": STEPS,
           "forward_calls_ok": forward_calls == STEPS}
    out["strict_pass"] = all(out[k] for k in STRICT_FLAGS)
    return out


def validate_report(report: dict) -> list[str]:
    """Problems that make the evidence run unacceptable (empty list = acceptable).
    Re-derives every check from the recorded fields instead of trusting strict_pass alone."""
    problems = []
    scenarios = report.get("scenarios") or {}
    for name, n_expected in EXPECTED_RECORDS.items():
        sc = scenarios.get(name)
        if sc is None:
            problems.append(f"{name}: scenario missing")
            continue
        records = sc.get("records") or []
        if len(records) != n_expected:
            problems.append(f"{name}: {len(records)} request records, expected {n_expected}")
        for i, rec in enumerate(records):
            where = f"{name}[{i}] {rec.get('label')!r}"
            failed = [k for k in STRICT_FLAGS + ("strict_pass",) if rec.get(k) is not True]
            if failed:
                problems.append(f"{where}: strict checks not true: {failed}")
            for k in ("num_tokens", "num_step_logits", "num_ref_tokens", "num_ref_step_logits", "expected_steps"):
                if rec.get(k) != STEPS:
                    problems.append(f"{where}: {k}={rec.get(k)}, expected {STEPS}")
            if not (rec.get("forward_calls") == rec.get("expected_forward_calls") == STEPS):
                problems.append(f"{where}: forward_calls={rec.get('forward_calls')}, "
                                f"expected_forward_calls={rec.get('expected_forward_calls')}, expected {STEPS}")
            if rec.get("L", 0) > 0 and not (rec.get("strict_restored_kv_bytes") or {}).get("all_bytes_equal"):
                problems.append(f"{where}: restored L={rec.get('L')} without a passing K/V byte check")
        if name in EXPECTED_FAULT_MISS:
            idx, reason = EXPECTED_FAULT_MISS[name]
            got = records[idx].get("miss_reason") if len(records) > idx else None
            if got != reason:
                problems.append(f"{name}: intended fault not observed (record {idx} miss_reason={got!r}, "
                                f"expected {reason!r})")
    enospc = scenarios.get("fault_enospc")
    if enospc is not None:
        n_inj = enospc.get("enospc_injections", 0)
        n_fail = ((enospc.get("counters") or {}).get("spill_failures") or {}).get("write_oserror:ENOSPC", 0)
        if not (n_inj >= 1 and n_fail == n_inj):
            problems.append(f"fault_enospc: intended fault not observed (injections={n_inj}, "
                            f"spill_failures write_oserror:ENOSPC={n_fail})")
        if list(enospc.get("dir_after") or ()) != [0, []]:
            problems.append(f"fault_enospc: cache directory not empty after failed writes: {enospc.get('dir_after')}")
    return problems


def summarize(report: dict, elapsed_s=None) -> dict:
    """Counts derived from the actual records; acceptance = no validation problems."""
    records = [rec for sc in (report.get("scenarios") or {}).values() for rec in (sc.get("records") or [])]
    problems = validate_report(report)
    restored = [rec for rec in records if rec.get("restore_used")]
    return {
        "requests": len(records),
        "strict_records": sum("strict_pass" in rec for rec in records),
        "strict_pass_all": bool(records) and all(rec.get("strict_pass") is True for rec in records),
        "expected_requests": sum(EXPECTED_RECORDS.values()),
        "acceptance": not problems,
        "problems": problems,
        "restored_requests": len(restored),
        "report_only_full_recompute_greedy_identical":
            sum(bool(((rec.get("report_only_vs_full_recompute") or {}).get("greedy") or {}).get("identical"))
                for rec in restored),
        "elapsed_s": elapsed_s,
    }


def exit_code(summary: dict) -> int:
    return 0 if summary["acceptance"] else 1


class Run:
    def __init__(self, model, name, host_chunks, disk_files, write_fn=cf.default_write):
        self.model = model
        self.ns = namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, C)
        cb = cf.payload_len(self.ns)
        self.host = HostStore((host_chunks + 1) * cb, cb, max_entries=64, pin=True)
        self.disk = DiskStore(DISK_ROOT / name, disk_files * cf.file_size(self.ns), 64, write_fn=write_fn)
        self.tiered = TieredKVCache(self.ns, self.host, self.disk)
        self.engine = AlwaysRestoreEngine(model, self.tiered)
        self.ref = ResidentReference(model, self.ns.digest(), C)
        self.records = []

    def request(self, label, token_ids):
        seen = {}
        calls = []  # model forward calls made by the engine request only
        hook = self.model.register_forward_pre_hook(lambda *a: calls.append(1))
        try:
            r = self.engine.run(token_ids, STEPS,
                                lambda cache, L: seen.update(self.ref.check_restored(token_ids, cache, L)))
        finally:
            hook.remove()
        forward_calls = len(calls)
        # Not counted: resident replay and the report-only full recomputation below.
        ref_tokens, ref_steps = self.ref.run(token_ids, r.restore_len, r.stored_chunks, STEPS)
        strict = strict_checks(r.tokens, r.step_logits, ref_tokens, ref_steps, r.restore_len, seen, forward_calls)
        ids = torch.tensor([token_ids], device="cuda")
        full_logits, full_cache = km.prefill(self.model, ids, None, 0)
        full_tokens, full_steps = km.greedy_continue(self.model, full_cache, full_logits, len(token_ids), STEPS)
        del full_cache
        cmp_full = compare_logits(full_logits.cpu(), r.step_logits[0], ATOL, RTOL)
        g = greedy_agreement(full_tokens, r.tokens)
        div = None
        if g["first_divergence"] is not None and g["first_divergence"] < min(len(r.step_logits), len(full_steps)):
            s = g["first_divergence"]
            top = torch.topk(full_steps[s], 2)
            div = {"step": s, "full_token": full_tokens[s], "restored_token": r.tokens[s],
                   "full_top2_margin": float(top.values[0] - top.values[1]),
                   "full_logit_of_restored_token": float(full_steps[s][r.tokens[s]]),
                   "restored_logit_of_full_token": float(r.step_logits[s][full_tokens[s]]),
                   "step_max_abs_full_vs_restored": float((full_steps[s] - r.step_logits[s]).abs().max())}
        rec = {
            "label": label, "N": r.prompt_len, "eligible_L": r.eligible_restore_len, "L": r.restore_len,
            "restore_used": r.restore_used, "sources": r.sources, "miss_reason": r.miss_reason,
            "stored_chunks": r.stored_chunks,
            **strict,
            "report_only_vs_full_recompute": {
                "max_abs_err": cmp_full.get("max_abs_err"), "rel_l2_err": cmp_full.get("rel_l2_err"),
                "argmax_equal": cmp_full.get("argmax_equal"), "allclose_m0_tolerance": cmp_full["allclose"],
                "greedy": g, "first_divergence": div},
            "host_resident_logical_bytes": self.host.resident_bytes, "disk_used_bytes": self.disk.used_bytes,
            "tiers_after": [self.tiered.tier_of(k) for k in chain_keys(self.ns.digest(), token_ids, C)],
        }
        self.records.append(rec)
        g = rec["report_only_vs_full_recompute"]
        div = g["first_divergence"]
        div_txt = f"(div step {div['step']} margin {div['full_top2_margin']:.3f}) " if div else ""
        print(f"  {label:22s} N={rec['N']:4d} L={rec['L']:4d}/{rec['eligible_L']:4d} src={','.join(r.sources) or '-':15s} "
              f"miss={r.miss_reason} stored={r.stored_chunks} fwd={forward_calls}/{STEPS} "
              f"STRICT={'PASS' if rec['strict_pass'] else 'FAIL'} "
              f"| vs full: max_abs={g['max_abs_err']:.3f} greedy={g['greedy']['matching_positions']}/{STEPS} "
              f"{div_txt}| tiers={rec['tiers_after']}", flush=True)
        return r

    def path(self, key):
        return self.disk.root / f"{key.hex()}.kvc"

    def finish(self):
        out = {"records": self.records, "counters": self.tiered.counters.to_dict(), "budgets": self.tiered.budgets()}
        out["disk_files_left_after_clear"] = self.disk.clear()
        self.host.close()
        return out


def prompts(tok):
    shared = exact_length_ids(tok, BOUNDARY_TEXT, 768)
    calib = exact_length_ids(tok, CALIBRATION_TEXT, 900)
    tail_a = tok(" Summarize the logbook in one sentence.", add_special_tokens=False)["input_ids"]
    tail_b = tok(" What did the keeper do on foggy nights?", add_special_tokens=False)["input_ids"]
    return {"X": shared[:512] + calib[:100] + tail_a, "Y": shared[:512] + calib[300:420] + tail_b,
            "Z": shared[:256] + calib[500:600] + tail_b, "W": calib[:600], "L3": shared[:768] + tail_a + [11]}


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="run the GPU evidence scenarios and write a new report")
    mode.add_argument("--validate", help="re-validate a saved report (ignores its stored summary); no model")
    args = ap.parse_args()
    if args.validate:
        summary = summarize(json.loads(Path(args.validate).read_text()))
        print(json.dumps(summary, indent=1))
        sys.exit(exit_code(summary))
    if Path(args.out).exists():
        sys.exit(f"refusing to overwrite existing {args.out}")
    t0 = time.time()
    if DISK_ROOT.exists():
        shutil.rmtree(DISK_ROOT)  # our own scratch area under .kvcache/ only
    model, tok = km.load_model()
    P = prompts(tok)
    report = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__},
              "gpu": torch.cuda.get_device_name(0), "model": km.model_info(model), "chunk_size": C,
              "steps": STEPS, "namespace": namespace_for(model, km.MODEL_ID, km.MODEL_REVISION, C).fields(),
              "scenarios": {}}

    print("== sequence (host 2 slots + 1 staging, disk 2 files)")
    run = Run(model, "sequence", host_chunks=2, disk_files=2)
    for label in ["X", "Y", "W", "Z", "X", "L3", "Y", "W", "X"]:
        run.request(label, P[label])
    report["scenarios"]["sequence"] = run.finish()

    print("== fault: corrupt middle chunk (one payload bit)")
    run = Run(model, "corrupt", host_chunks=1, disk_files=8)
    run.request("L3 cold", P["L3"])
    keys = chain_keys(run.ns.digest(), P["L3"], C)
    raw = bytearray(run.path(keys[1]).read_bytes())
    raw[len(raw) // 2] ^= 0x01
    run.path(keys[1]).write_bytes(raw)
    run.request("L3 chunk1 corrupt", P["L3"])
    run.request("L3 again", P["L3"])
    report["scenarios"]["fault_corrupt_middle"] = run.finish()

    print("== fault: missing file")
    run = Run(model, "missing", host_chunks=1, disk_files=8)
    run.request("X cold", P["X"])
    run.path(chain_keys(run.ns.digest(), P["X"], C)[0]).unlink()
    run.request("Y chunk0 file deleted", P["Y"])
    report["scenarios"]["fault_missing_file"] = run.finish()

    for kind in ("namespace", "layout"):
        print(f"== fault: incompatible {kind}")
        run = Run(model, f"incompat_{kind}", host_chunks=1, disk_files=8)
        run.request("X cold", P["X"])
        path = run.path(chain_keys(run.ns.digest(), P["X"], C)[0])
        raw = path.read_bytes()
        if kind == "namespace":
            other = dataclasses.replace(run.ns, attn_implementation="eager")
            oid = ChunkIdentity.for_chunk(other.digest(), P["X"], C, 0, chain_keys(other.digest(), P["X"], C))
            with open(path, "wb") as f:
                cf.write_chunk(f, cf.encode_header(other, oid), raw[len(raw) - cf.payload_len(run.ns) - 4:-4])
        else:
            bad = bytearray(raw)
            bad[116:120] = (27).to_bytes(4, "little")
            path.write_bytes(bad)
        run.request(f"Y {kind} incompatible", P["Y"])
        report["scenarios"][f"fault_incompatible_{kind}"] = run.finish()

    print("== fault: ENOSPC on every disk write (spill)")

    injections = []

    def enospc(f, mv):
        injections.append(1)
        raise OSError(errno.ENOSPC, "No space left on device")
    run = Run(model, "enospc", host_chunks=1, disk_files=8, write_fn=enospc)
    run.request("L3 ENOSPC on spill", P["L3"])
    dir_after = list(run.disk.scan_dir_bytes())  # [bytes, names]; same form in memory and in JSON
    report["scenarios"]["fault_enospc"] = {"enospc_injections": len(injections), "dir_after": dir_after, **run.finish()}
    print(f"  ENOSPC injections {len(injections)}; spill_failures "
          f"{report['scenarios']['fault_enospc']['counters']['spill_failures']}; dir after {dir_after}")

    report["summary"] = summarize(report, elapsed_s=round(time.time() - t0, 1))
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"]))
    print(f"wrote {args.out}")
    sys.exit(exit_code(report["summary"]))


if __name__ == "__main__":
    main()
