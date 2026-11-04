"""Diagnose a greedy-decode divergence between path A (full) and path B (restore).

For one prompt, prints per-step top-5 logits on both paths. It also runs
"B teacher-forced on A's tokens", which separates "near-tie at one step"
from "accumulated drift". Writes results/m0/divergence_<name>.json.

Usage: .venv/bin/python scripts/m0_diagnose_divergence.py nat_long
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kvlab.model as km  # noqa: E402
import torch  # noqa: E402
from kvlab.boundaries import max_restore_len  # noqa: E402
from kvlab.fixtures import BOUNDARY_TEXT, NATURAL_PROMPTS, chat_ids, exact_length_ids  # noqa: E402
from kvlab.kv_snapshot import export_prefix_to_cpu, restore_to_device  # noqa: E402

CHUNK, STEPS = 256, 8


def top5(logits, tok):
    v, i = torch.topk(logits, 5)
    return [(int(t), tok.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]


def teacher_forced(model, ids, cache, start, forced):
    """Prefill then feed `forced` tokens; return logits at each step."""
    logits, cache = km.prefill(model, ids, cache, start)
    out, pos = [logits], start + ids.shape[1]
    for t in forced[:-1]:
        logits, cache = km.prefill(model, torch.tensor([[t]], device="cuda"), cache, pos)
        out.append(logits)
        pos += 1
    return out


def main():
    name = sys.argv[1]
    model, tok = km.load_model()
    if name in NATURAL_PROMPTS:
        token_ids = chat_ids(tok, NATURAL_PROMPTS[name])
    else:
        token_ids = exact_length_ids(tok, BOUNDARY_TEXT, int(name.split("_")[1]))
    ids = torch.tensor([token_ids], device="cuda")
    n = ids.shape[1]
    lmax = max_restore_len(n, CHUNK)

    logits_a, cache_a = km.prefill(model, ids, None, 0)
    tokens_a, steps_a = km.greedy_continue(model, cache_a, logits_a, n, STEPS)
    del cache_a
    _, pc = km.prefill(model, ids[:, :lmax], None, 0)
    snap = export_prefix_to_cpu(pc, lmax, CHUNK)
    del pc
    cache_b = restore_to_device(snap, lmax, model.config, "cuda")
    logits_b, cache_b = km.prefill(model, ids[:, lmax:], cache_b, lmax)
    tokens_b, steps_b = km.greedy_continue(model, cache_b, logits_b, n, STEPS)
    del cache_b
    forced_b = teacher_forced(model, ids[:, lmax:], restore_to_device(snap, lmax, model.config, "cuda"),
                              lmax, tokens_a)

    rows = []
    for s in range(STEPS):
        a = steps_a[s]
        fb = forced_b[s].cpu()
        ta = torch.topk(a, 2)
        tf = torch.topk(fb, 2)
        rows.append({
            "step": s, "token_A": tokens_a[s], "token_B": tokens_b[s],
            "A_top5": top5(a, tok), "B_top5": top5(steps_b[s], tok) if s < len(steps_b) else None,
            "A_top2_margin": float(ta.values[0] - ta.values[1]),
            "B_forced_on_A_tokens_top5": top5(fb, tok),
            "B_forced_top2_margin": float(tf.values[0] - tf.values[1]),
            "B_forced_argmax_equals_A": int(tf.indices[0]) == tokens_a[s],
            "max_abs_A_vs_B_forced": float((fb - a).abs().max()),
        })
        r = rows[-1]
        print(f"step {s}: A={tokens_a[s]!s:>6} B={tokens_b[s]!s:>6} A_margin={r['A_top2_margin']:.4f} "
              f"Bforced_margin={r['B_forced_top2_margin']:.4f} Bforced_argmax==A={r['B_forced_argmax_equals_A']} "
              f"max_abs(A,Bforced)={r['max_abs_A_vs_B_forced']:.4f}")
        print(f"    A top5: {r['A_top5']}")
        print(f"    Bforced top5: {r['B_forced_on_A_tokens_top5']}")
    print("A text:", repr(tok.decode(tokens_a)))
    print("B text:", repr(tok.decode(tokens_b)))
    out = km.PROJECT_ROOT / "results" / "m0" / f"divergence_{name}.json"
    out.write_text(json.dumps({"name": name, "N": n, "L": lmax, "tokens_A": tokens_a, "tokens_B": tokens_b,
                               "text_A": tok.decode(tokens_a), "text_B": tok.decode(tokens_b),
                               "steps": rows}, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
