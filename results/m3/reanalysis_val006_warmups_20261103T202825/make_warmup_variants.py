"""Scratch copies of the five M3 evaluation sessions with altered warmup evidence (VAL-006 follow-up).
Usage: python make_warmup_variants.py <eval run dir> <scratch dir>. The source files are only read."""
import copy, json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])


def first(report, **kv):
    return next(i for i, s in enumerate(report["samples"]) if all(s.get(k) == v for k, v in kv.items()))


def no_warmups(rep):  # Codex's reproduction: drop every warmup row, adjust only meta.n_samples
    rep["samples"] = [s for s in rep["samples"] if not s.get("warmup")]
    rep["meta"]["n_samples"] = len(rep["samples"])


def block_warmup_substituted(rep):  # same row count: one block warmup replaced by a copy of another
    i = first(rep, stage="m3", block=3, rep=1, path="host", backend="batched", mode="plain")
    j = first(rep, stage="m3", block=3, rep=0, path="host", backend="batched", mode="plain")
    rep["samples"][i] = copy.deepcopy(rep["samples"][j])


def process_warmup_substituted(rep):  # same row count: one process warmup moved to another combination
    i = first(rep, stage="process_warmup", path="disk", backend="batched", mode="synced")
    j = first(rep, stage="process_warmup", path="host", backend="batched", mode="synced", L=rep["samples"][i]["L"])
    rep["samples"][i] = copy.deepcopy(rep["samples"][j])


VARIANTS = {"no_warmups": no_warmups, "block_warmup_substituted": block_warmup_substituted,
            "process_warmup_substituted": process_warmup_substituted}
for name, edit in VARIANTS.items():
    d = dst / name
    d.mkdir(parents=True)
    for i in range(1, 6):
        rep = json.loads((src / f"session_s{i}.json").read_text())
        n0 = len(rep["samples"])
        if name == "no_warmups" or i == 2:
            edit(rep)
        (d / f"session_s{i}.json").write_text(json.dumps(rep, indent=1, default=str))
        if n0 != len(rep["samples"]):
            print(f"{name} s{i}: {n0} -> {len(rep['samples'])} samples")
