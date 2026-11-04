"""M0.1 audit: full (A) vs GPU-resident split (B) vs CPU-restored split (C).

  A  full prompt prefill on GPU
  B  prefix prefill, cache stays on GPU (never copied), then suffix prefill
  C  same split as B, but the prefix cache is exported to CPU, the GPU cache is
     deleted, and a fresh GPU cache is restored from the CPU snapshot

All three use the same model, token ids, dtype, backend and kvlab.model.prefill.
Sections:
  paths       repeated A/B/C runs; strict bitwise B-vs-C; A-vs-B on identical
              histories via teacher forcing
  divergence  first greedy divergence: history, candidates, exact logits, bf16
              bit patterns, fp32 dot products of the pre-lm_head hidden state
  controls    negative controls (bit flips, wrong prefix, swaps, zeroed K/V,
              position offset) and which check detects each
With --fp32-diagnostic the model is upcast to fp32 after loading, as a
precision instrument only (the project baseline stays bf16); it re-runs a
subset of `paths` and the divergence candidates.

Usage:
  .venv/bin/python scripts/m0_1_audit.py --out results/m0_1/audit_bf16.json
  .venv/bin/python scripts/m0_1_audit.py --fp32-diagnostic --out results/m0_1/audit_fp32.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402  (sets HF_HOME before transformers import)
import torch  # noqa: E402
import transformers  # noqa: E402
from kvlab.boundaries import eligible_boundaries  # noqa: E402
from kvlab.experiment import _free_gpu, gpu_resident_cache  # noqa: E402
from kvlab.fixtures import BOUNDARY_TEXT, CALIBRATION_TEXT, NATURAL_PROMPTS, chat_ids, exact_length_ids  # noqa: E402
from kvlab.kv_snapshot import (KVSnapshot, compare_snapshot_to_cache, export_prefix_to_cpu,  # noqa: E402
                               restore_to_device, zeroed_copy)
from kvlab.metrics import compare_logits, greedy_agreement  # noqa: E402
from kvlab.tolerances import ATOL, RTOL  # noqa: E402

CHUNK, STEPS = 256, 8


# ---------------------------------------------------------------- paths
def decode(model, suffix, cache, start, forced=None):
    """Prefill `suffix` at `start`, then run STEPS decode steps.
    forced=None: greedy (feed own argmax). forced=list: feed forced[i] at step i+1
    (teacher forcing), so histories are identical to the forcing path.
    Returns (argmax token at each step, per-step float32 logits on CPU)."""
    logits, cache = km.prefill(model, suffix, cache, start)
    pos = start + suffix.shape[1]
    tokens, steps = [], [logits.cpu()]
    for i in range(STEPS):
        t = int(torch.argmax(logits))
        tokens.append(t)
        if i == STEPS - 1:
            break
        feed = t if forced is None else forced[i]
        logits, cache = km.prefill(model, torch.tensor([[feed]], device="cuda"), cache, pos)
        steps.append(logits.cpu())
        pos += 1
    del cache
    return tokens, steps


def path_a(model, ids, forced=None):
    return decode(model, ids, None, 0, forced)


def path_b(model, ids, L, P, forced=None):
    """Resident split. P == L: the prefix cache object itself is extended (no copy).
    P > L: prefix of P tokens computed, first L tokens copied GPU->GPU."""
    if P == L:
        _, cache = km.prefill(model, ids[:, :L], None, 0)
    else:
        cache = gpu_resident_cache(model, ids, P, L)
    return decode(model, ids[:, L:], cache, L, forced)


def make_snapshot(model, ids, P):
    _, cache = km.prefill(model, ids[:, :P], None, 0)
    snap = export_prefix_to_cpu(cache, P, CHUNK)
    check = compare_snapshot_to_cache(snap, cache, P)
    del cache
    _free_gpu()  # the source GPU cache no longer exists when C restores
    return snap, check


def path_c(model, ids, L, snap, forced=None):
    cache = restore_to_device(snap, L, model.config, "cuda")
    check = compare_snapshot_to_cache(snap, cache, L)
    tokens, steps = decode(model, ids[:, L:], cache, L, forced)
    return tokens, steps, check


def bitwise(s1, s2):
    return len(s1) == len(s2) and all(torch.equal(a, b) for a, b in zip(s1, s2))


def max_abs(a, b):
    return float((a - b).abs().max())


def fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def first_cmp(a, b):
    c = compare_logits(a, b, ATOL, RTOL)
    return {k: c[k] for k in ("max_abs_err", "rel_l2_err", "allclose", "argmax_equal", "ref_top2_margin")
            if k in c}


def top2_margins(steps):
    out = []
    for s in steps:
        v = torch.topk(s, 2).values
        out.append(float(v[0] - v[1]))
    return out


def run_paths(model, name, token_ids, repeats, partial=True):
    ids = torch.tensor([token_ids], device="cuda")
    n = ids.shape[1]
    bounds = eligible_boundaries(n, CHUNK)
    a_runs = [path_a(model, ids) for _ in range(repeats)]
    a_tok, a_steps = a_runs[0]
    rec = {"name": name, "N": n, "eligible_L": bounds, "A_tokens": a_tok,
           "A_repeat_bitwise": all(r[0] == a_tok and bitwise(r[1], a_steps) for r in a_runs[1:]),
           "A_top2_margins": top2_margins(a_steps), "splits": []}
    for L in bounds:
        variants = [("exact", L)] + ([("partial", bounds[-1])] if partial and L < bounds[-1] else [])
        for vname, P in variants:
            b_runs = [path_b(model, ids, L, P) for _ in range(repeats)]
            c_runs = []
            for _ in range(repeats):
                snap, exp_check = make_snapshot(model, ids, P)
                c_tok, c_steps, res_check = path_c(model, ids, L, snap)
                c_runs.append({"tokens": c_tok, "steps": c_steps, "export_ok": exp_check["all_bytes_equal"],
                               "restore_ok": res_check["all_bytes_equal"],
                               "tensors": exp_check["tensors_checked"] + res_check["tensors_checked"]})
                del snap
            b_tok, b_steps = b_runs[0]
            c_tok, c_steps = c_runs[0]["tokens"], c_runs[0]["steps"]
            bf_tok, bf_steps = path_b(model, ids, L, P, forced=a_tok)
            snap, _ = make_snapshot(model, ids, P)
            cf_tok, cf_steps, _ = path_c(model, ids, L, snap, forced=a_tok)
            # HF default positions/mask (no explicit position_ids/attention_mask) must match ours bitwise.
            default_eq = None
            if vname == "exact" and L == bounds[-1]:
                cache = restore_to_device(snap, L, model.config, "cuda")
                with torch.inference_mode():
                    out = model(input_ids=ids[:, L:], past_key_values=cache, use_cache=True, logits_to_keep=1)
                default_eq = bool(torch.equal(out.logits[0, -1].float().cpu(), c_steps[0]))
                del cache, out
            del snap
            _free_gpu()
            rec["splits"].append({
                "L": L, "prefix_computed": P, "variant": vname, "suffix_len": n - L,
                "B_repeat_bitwise": all(r[0] == b_tok and bitwise(r[1], b_steps) for r in b_runs[1:]),
                "C_repeat_bitwise": all(r["tokens"] == c_tok and bitwise(r["steps"], c_steps) for r in c_runs[1:]),
                "C_byte_checks_pass": all(r["export_ok"] and r["restore_ok"] for r in c_runs),
                "C_tensors_byte_checked": sum(r["tensors"] for r in c_runs),
                "B_vs_C_tokens_equal_all_repeats": all(r["tokens"] == b_tok for r in c_runs),
                "B_vs_C_all_step_logits_bitwise_all_repeats": all(bitwise(r["steps"], b_steps) for r in c_runs),
                "Bforced_vs_Cforced_bitwise": bitwise(bf_steps, cf_steps),
                "default_positions_bitwise_equal": default_eq,
                "B_tokens": b_tok,
                "A_vs_B_first": first_cmp(a_steps[0], b_steps[0]),
                "A_vs_B_greedy": greedy_agreement(a_tok, b_tok),
                "A_vs_C_greedy": greedy_agreement(a_tok, c_tok),
                "A_vs_Bforced_step_max_abs": [max_abs(x, y) for x, y in zip(a_steps, bf_steps)],
                "A_vs_Bforced_step_argmax_equal": [x == y for x, y in zip(a_tok, bf_tok)],
            })
            s = rec["splits"][-1]
            print(f"  {name} L={L} P={P} B_rep={s['B_repeat_bitwise']} C_rep={s['C_repeat_bitwise']} "
                  f"bytes={s['C_byte_checks_pass']} B==C(all steps,{repeats}x)="
                  f"{s['B_vs_C_all_step_logits_bitwise_all_repeats']} Bf==Cf={s['Bforced_vs_Cforced_bitwise']} "
                  f"default_pos={default_eq} A-B max_abs={fmt(s['A_vs_B_first'].get('max_abs_err'))} "
                  f"A-B greedy={s['A_vs_B_greedy']['matching_positions']}/{STEPS} "
                  f"forced argmax eq={sum(s['A_vs_Bforced_step_argmax_equal'])}/{STEPS}", flush=True)
    return rec


# ---------------------------------------------------------------- divergence
def bf16_bits(x: float) -> str:
    t = torch.tensor([x]).to(torch.bfloat16)
    if float(t.float()) != x:
        return "not-bf16-representable"
    return f"0x{int(t.view(torch.int16).item()) & 0xFFFF:04x}"


def capture_hidden(model, fn):
    """Run fn() while recording the lm_head input (final hidden state, last position) of every call."""
    rows = []
    handle = model.lm_head.register_forward_hook(lambda m, inp, out: rows.append(inp[0][0, -1].detach().clone()))
    try:
        fn()
    finally:
        handle.remove()
    return rows


def divergence(model, tok, name, token_ids, L, a_tok, b_tok):
    s = next(i for i, (x, y) in enumerate(zip(a_tok, b_tok)) if x != y)
    ids = torch.tensor([token_ids], device="cuda")
    W = model.lm_head.weight
    cand = [a_tok[s], b_tok[s]]
    out = {"name": name, "L": L, "step": s, "shared_history_generated_tokens": a_tok[:s],
           "shared_history_text": tok.decode(a_tok[:s]),
           "candidates": [{"id": t, "text": tok.decode([t])} for t in cand], "paths": {}}
    snap, _ = make_snapshot(model, ids, L)
    runners = {
        "A_full": lambda: path_a(model, ids, forced=a_tok),
        "B_resident_split": lambda: path_b(model, ids, L, L, forced=a_tok),
        "C_restored_split": lambda: path_c(model, ids, L, snap, forced=a_tok)[:2],
    }
    step_logits = {}
    for pname, fn in runners.items():
        result = {}
        hidden = capture_hidden(model, lambda: result.setdefault("r", fn()))
        logits = result["r"][1][s]
        step_logits[pname] = logits
        # The last STEPS lm_head calls are the suffix prefill + decode steps. Path B also runs its
        # prefix prefill inside the capture, so index from the end (audit run 1 indexed from the start).
        if len(hidden) < STEPS:
            raise RuntimeError(f"captured {len(hidden)} lm_head calls, expected >= {STEPS}")
        h = hidden[len(hidden) - STEPS + s].float()
        top = torch.topk(logits, 3)
        out["paths"][pname] = {
            "top3": [{"id": int(i), "text": tok.decode([int(i)]), "logit": float(v), "bf16_bits": bf16_bits(float(v))}
                     for v, i in zip(top.values, top.indices)],
            "candidate_logits": {str(t): float(logits[t]) for t in cand},
            "candidate_bf16_bits": {str(t): bf16_bits(float(logits[t])) for t in cand},
            "candidates_exactly_equal": bool(logits[cand[0]] == logits[cand[1]]),
            "top1_minus_top2": float(top.values[0] - top.values[1]),
            "fp32_dot_from_hidden": {str(t): float(h @ W[t].detach().float()) for t in cand},
            "lm_head_calls_captured": len(hidden),
            "argmax": int(torch.argmax(logits)),
        }
    del snap
    out["A_vs_B_step_max_abs"] = max_abs(step_logits["A_full"], step_logits["B_resident_split"])
    out["B_vs_C_step_bitwise"] = bool(torch.equal(step_logits["B_resident_split"], step_logits["C_restored_split"]))
    # Unchunked reference on the identical history: one full prefill of prompt + shared generated tokens.
    hist = torch.tensor([token_ids + a_tok[:s]], device="cuda")
    full_hist, _ = km.prefill(model, hist, None, 0)
    full_hist = full_hist.cpu()
    out["full_prefill_of_history"] = {
        "candidate_logits": {str(t): float(full_hist[t]) for t in cand},
        "argmax": int(torch.argmax(full_hist)),
        "max_abs_vs_A_decode_step": max_abs(full_hist, step_logits["A_full"]),
    }
    _free_gpu()
    return out


# ---------------------------------------------------------------- controls
def clone_snap(snap):
    return KVSnapshot(snap.chunk_size, snap.num_layers,
                      [[(k.clone(), v.clone()) for k, v in layers] for layers in snap.chunks])


def flip_lsb(snap, chunk, layer, kind):
    s = clone_snap(snap)
    t = s.chunks[chunk][layer][0 if kind == "k" else 1]
    t.view(torch.int16).view(-1)[1000] ^= 1  # lowest mantissa bit of one bf16 element
    return s


def set_element(snap, chunk, layer, kind, value):
    s = clone_snap(snap)
    t = s.chunks[chunk][layer][0 if kind == "k" else 1]
    t.view(-1)[1000] = value
    return s


def swap_layers(snap, a, b):
    s = clone_snap(snap)
    for layers in s.chunks:
        layers[a], layers[b] = layers[b], layers[a]
    return s


def swap_kv(snap, layer):
    s = clone_snap(snap)
    for layers in s.chunks:
        k, v = layers[layer]
        layers[layer] = (v, k)
    return s


def swap_chunks(snap, a, b):
    s = clone_snap(snap)
    s.chunks[a], s.chunks[b] = s.chunks[b], s.chunks[a]
    return s


def controls(model, tok, name, token_ids, L):
    ids = torch.tensor([token_ids], device="cuda")
    n = ids.shape[1]
    a_tok, a_steps = path_a(model, ids)
    b_tok, b_steps = path_b(model, ids, L, L)
    snap, _ = make_snapshot(model, ids, L)
    _, ref_cache = km.prefill(model, ids[:, :L], None, 0)  # resident reference for byte checks

    other_ids = torch.tensor([exact_length_ids(tok, CALIBRATION_TEXT, L)], device="cuda")
    one_tok = ids[:, :L].clone()
    one_tok[0, L // 2] = (one_tok[0, L // 2] + 1) % model.config.vocab_size
    variants = {
        "valid_restore": snap,
        "bitflip_lsb_one_element": flip_lsb(snap, 0, 13, "k"),
        "one_element_set_to_1e3": set_element(snap, 0, 13, "k", 1000.0),
        "wrong_prefix_other_text": make_snapshot(model, other_ids, L)[0],
        "wrong_prefix_one_token_changed": make_snapshot(model, one_tok, L)[0],
        "layers_0_1_swapped": swap_layers(snap, 0, 1),
        "k_v_swapped_layer_5": swap_kv(snap, 5),
        "zeroed": zeroed_copy(snap),
    }
    if L >= 2 * CHUNK:
        variants["chunks_0_1_swapped"] = swap_chunks(snap, 0, 1)
    rows = {}
    for vname, s in variants.items():
        byte = compare_snapshot_to_cache(s, ref_cache, L)
        c_tok, c_steps, _ = path_c(model, ids, L, s)
        cmp_a = first_cmp(a_steps[0], c_steps[0])
        rows[vname] = {
            "byte_check_vs_written_prefix": byte["all_bytes_equal"],
            "mismatched_tensors": len(byte["mismatches"]),
            "tensors_checked": byte["tensors_checked"],
            "first_logits_bitwise_equal_B": bool(torch.equal(c_steps[0], b_steps[0])),
            "all_steps_bitwise_equal_B": bitwise(c_steps, b_steps),
            "A_vs_C_first_max_abs": cmp_a.get("max_abs_err"),
            "A_vs_C_allclose_atol_1.45": cmp_a["allclose"],
            "A_vs_C_argmax_equal": cmp_a.get("argmax_equal"),
            "greedy_vs_A": greedy_agreement(a_tok, c_tok)["matching_positions"],
            "greedy_vs_B": greedy_agreement(b_tok, c_tok)["matching_positions"],
            "all_finite": all(bool(torch.isfinite(x).all()) for x in c_steps),
        }
        r = rows[vname]
        print(f"  {name} L={L} {vname:32s} bytes_ok={r['byte_check_vs_written_prefix']!s:5} "
              f"mism={r['mismatched_tensors']:3d} ==B(all steps)={r['all_steps_bitwise_equal_B']!s:5} "
              f"A-C max_abs={fmt(r['A_vs_C_first_max_abs'])} allclose(atol)={r['A_vs_C_allclose_atol_1.45']!s:5} "
              f"greedy vs A {r['greedy_vs_A']}/{STEPS} vs B {r['greedy_vs_B']}/{STEPS}", flush=True)
    del variants
    _free_gpu()

    # Position control: valid restored K/V, but suffix query positions shifted by +1.
    cache = restore_to_device(snap, L, model.config, "cuda")
    with torch.inference_mode():
        pos = torch.arange(L + 1, n + 1, device="cuda").unsqueeze(0)
        mask = torch.ones((1, n), dtype=torch.long, device="cuda")
        out = model(input_ids=ids[:, L:], attention_mask=mask, position_ids=pos, past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
    shifted = out.logits[0, -1].float().cpu()
    del cache, out
    cmp_shift = first_cmp(a_steps[0], shifted)
    cache = restore_to_device(snap, L, model.config, "cuda")
    try:
        km.prefill(model, ids[:, L:], cache, L + 1)
        guard = "NOT RAISED"
    except ValueError as e:  # the invariant check under test; anything else propagates
        guard = f"ValueError: {e}"
    del cache
    rows["suffix_positions_shifted_by_+1"] = {
        "byte_check_vs_written_prefix": True,
        "first_logits_bitwise_equal_B": bool(torch.equal(shifted, b_steps[0])),
        "A_vs_C_first_max_abs": cmp_shift.get("max_abs_err"),
        "A_vs_C_allclose_atol_1.45": cmp_shift["allclose"],
        "A_vs_C_argmax_equal": cmp_shift.get("argmax_equal"),
        "kvlab_prefill_guard": guard,
    }
    r = rows["suffix_positions_shifted_by_+1"]
    print(f"  {name} L={L} suffix_positions_shifted_by_+1        ==B={r['first_logits_bitwise_equal_B']} "
          f"A-C max_abs={fmt(r['A_vs_C_first_max_abs'])} allclose(atol)={r['A_vs_C_allclose_atol_1.45']} "
          f"argmax_eq={r['A_vs_C_argmax_equal']} guard={guard}", flush=True)
    del snap, ref_cache
    _free_gpu()
    return {"name": name, "N": n, "L": L, "rows": rows}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32-diagnostic", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0)
    model, tok = km.load_model()  # bf16 baseline, SDPA (checked inside)
    if args.fp32_diagnostic:
        model = model.float()  # exact upcast of bf16 weights; diagnostic instrument only
    dtype = str(next(model.parameters()).dtype)
    cases = {
        "nat_long": chat_ids(tok, NATURAL_PROMPTS["nat_long"]),
        "boundary_257": exact_length_ids(tok, BOUNDARY_TEXT, 257),
        "boundary_511": exact_length_ids(tok, BOUNDARY_TEXT, 511),
        "boundary_513": exact_length_ids(tok, BOUNDARY_TEXT, 513),
        "nat_xlong": chat_ids(tok, NATURAL_PROMPTS["nat_xlong"]),
    }
    report = {"mode": "fp32_diagnostic" if args.fp32_diagnostic else "bf16_baseline",
              "model_dtype": dtype, "attn_implementation": model.config._attn_implementation,
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
              "gpu": torch.cuda.get_device_name(0), "model": km.model_info(model),
              "repeats": args.repeats, "chunk_size": CHUNK, "steps": STEPS,
              "m0_tolerance_reported_only": {"atol": ATOL, "rtol": RTOL}, "paths": [], "divergences": []}
    print(f"== paths ({dtype})", flush=True)
    names = list(cases) if not args.fp32_diagnostic else ["nat_long", "boundary_511", "nat_xlong"]
    for name in names:
        rec = run_paths(model, name, cases[name], args.repeats if not args.fp32_diagnostic else 2)
        report["paths"].append(rec)
        for s in rec["splits"]:
            if s["variant"] == "exact" and not s["A_vs_B_greedy"]["identical"]:
                d = divergence(model, tok, name, cases[name], s["L"], rec["A_tokens"], s["B_tokens"])
                report["divergences"].append(d)
                print(f"  DIVERGENCE {name} L={s['L']} step={d['step']} "
                      + json.dumps({p: {"cand": v["candidate_logits"], "bits": v["candidate_bf16_bits"],
                                        "fp32dot": v["fp32_dot_from_hidden"]} for p, v in d["paths"].items()}),
                      flush=True)
    if args.fp32_diagnostic:
        # Same history as the bf16 failure: force the bf16 A tokens recorded in the original M0 evidence.
        orig = json.loads((km.PROJECT_ROOT / "results/m0/divergence_nat_long.json").read_text())
        d = divergence(model, tok, "nat_long[bf16 history]", cases["nat_long"], orig["L"],
                       orig["tokens_A"], orig["tokens_B"])
        report["divergences"].append(d)
        print("  fp32 at bf16 divergence step " + str(d["step"]) + ": "
              + json.dumps({p: {"cand": v["candidate_logits"], "argmax": v["argmax"]} for p, v in d["paths"].items()}),
              flush=True)
        report["note"] = "fp32 = exact upcast of bf16 weights; KV/logits in fp32; not a baseline change"
    print("== controls", flush=True)
    report["controls"] = [controls(model, tok, "nat_long", cases["nat_long"], 256),
                          controls(model, tok, "boundary_513", cases["boundary_513"], 512)]
    report["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {args.out} in {report['elapsed_s']} s")


if __name__ == "__main__":
    main()
